import asyncio
import json
import os
import queue
import re
import threading
from datetime import datetime

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"
REPORT_DIR = r"D:\NCKU_paper_Capstone_Project\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 【v5 修正】改指向新版 MCP Server，內含依 incident_id 精確查詢的工具。
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

BERTSCORE_SAFE_LENGTH = 200
MAX_ITERATIONS = 2
VERIFICATION_FIELDS = ["incident_id", "camera_id", "event_type", "severity", "risk_level", "timestamp"]
VERIFICATION_LOG_PATH = os.path.join(REPORT_DIR, "verification_log.jsonl")
TOOL_NAME = "get_incident_by_id"

REFERENCE_REPORTS = {
    "fire": (
        "事件摘要：偵測系統於監控畫面中確認煙囪出現明火，判定為火炬異常燃燒事件。"
        "歸因判定：初步研判為火炬燃燒異常或潛在火災前兆，需立即派員現場確認。"
        "風險等級：CRITICAL，屬高風險事件，可能有火災擴大風險。"
        "建議處置：立即通知現場工安人員前往確認，並視情況啟動滅火應變程序。"
        "追蹤建議：持續監控該煙囪畫面直到火焰消失並確認現場排除風險。"
    ),
    "black_smoky": (
        "事件摘要：偵測系統確認煙囪持續排放黑煙，超過安全門檻時間。"
        "歸因判定：初步研判為燃燒控制系統失效或不完全燃燒所致。"
        "風險等級：HIGH，屬中高風險事件，需儘速排查燃燒系統。"
        "建議處置：通知製程工程師檢查燃燒控制參數，排除不完全燃燒問題。"
        "追蹤建議：持續監控排煙狀況，確認恢復正常排放後解除警報。"
    ),
    "white_smoky": (
        "事件摘要：偵測系統確認煙囪出現異常白煙排放。"
        "歸因判定：初步研判為製程洩漏或蒸氣異常溢散。"
        "風險等級：WARNING，屬中低風險事件，仍須現場確認排除疑慮。"
        "建議處置：通知現場人員檢查管線閥門與製程狀態，確認無洩漏風險。"
        "追蹤建議：持續監控排放狀況，確認恢復正常後解除警報。"
    ),
    "black_smoky+fire": (
        "事件摘要：偵測系統同時確認煙囪出現濃黑煙與明火，判定為複合型高風險事件。"
        "歸因判定：初步研判為燃燒失控導致火災或爆燃風險，情況危急。"
        "風險等級：CRITICAL，屬最高風險等級，疑似火災或爆燃事故。"
        "建議處置：立即啟動緊急應變程序，通知消防與工安主管到場處置。"
        "追蹤建議：持續監控現場狀況，直到危險完全解除並完成事後調查。"
    ),
}

_report_queue = queue.Queue()
_worker_thread = None
_worker_start_lock = threading.Lock()
_tokenizer_patched = False


def _report_worker_loop():
    """【v7 修正】原本用互斥鎖，撞到就直接放棄該次報告生成，資料庫裡明明有
    這筆事件紀錄，卻永遠不會有對應的報告或驗證紀錄，而且完全不會留下任何
    「這筆被跳過了」的痕跡，只會在終端機閃過一行訊息。改成單一背景工作
    執行緒依序從佇列取出事件處理，每個事件都保證會被生成，差別只在於
    等多久才輪到它，不會再有事件憑空消失。"""
    while True:
        incident_id, event_type = _report_queue.get()
        try:
            asyncio.run(_generate_report_async(incident_id, event_type))
        except Exception as e:
            import traceback
            print(f"⚠️ [報告生成] 處理 incident_id={incident_id} 時發生未預期錯誤：{e}")
            traceback.print_exc()
        finally:
            _report_queue.task_done()


def _ensure_worker_started():
    global _worker_thread
    with _worker_start_lock:
        if _worker_thread is None:
            _worker_thread = threading.Thread(target=_report_worker_loop, daemon=True)
            _worker_thread.start()


def _ensure_tokenizer_patch():
    global _tokenizer_patched
    if _tokenizer_patched:
        return
    import transformers
    _original_from_pretrained = transformers.AutoTokenizer.from_pretrained.__func__

    def _patched_from_pretrained(cls, *args, **kwargs):
        tokenizer = _original_from_pretrained(cls, *args, **kwargs)
        if getattr(tokenizer, "model_max_length", 0) > 100000:
            tokenizer.model_max_length = 512
        return tokenizer

    transformers.AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)
    _tokenizer_patched = True


def _build_user_prompt(incident_id: int):
    """【v5 修正】明確把「這次要查的是哪一個 incident_id」寫進 prompt，
    避免模型呼叫到查「最新一筆」的舊工具、或漏帶參數。"""
    fields_hint = "、".join(VERIFICATION_FIELDS)
    return (
        "【工安事件通報】後台狀態機已觸發告警，本次目標事件的 incident_id 為 "
        f"{incident_id}。請你【必須】立刻調用 `get_incident_by_id` 工具，"
        f"並傳入參數 incident_id={incident_id}，取得該筆事件的結構化資料，"
        "不允許自行猜測，也不可以查詢其他事件。取得資料後：\n"
        "1. 生成一份專業的中文事故初報，內容需包含：事件摘要、歸因判定、"
        "風險等級、建議處置、追蹤建議。\n"
        "2. 在報告最後，另起一段，附上一個 ```json 區塊```，"
        f"格式為包含以下欄位的 JSON 物件：{fields_hint}。"
        "這段 JSON 裡的每一個值，必須跟你剛剛從工具取得的原始資料完全一致、"
        "逐字複製，不可以自行改寫格式、翻譯、四捨五入或憑印象填寫。"
    )


def _parse_verification_block(text: str):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _strip_verification_block(text: str) -> str:
    return re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()


def _verify_fields(llm_json, ground_truth: dict):
    if llm_json is None:
        return False, ["未偵測到驗證用JSON區塊，無法確認是否忠實使用工具結果"]
    mismatches = []
    for field in VERIFICATION_FIELDS:
        gt_val = str(ground_truth.get(field, "")).strip()
        llm_val = str(llm_json.get(field, "")).strip()
        if gt_val != llm_val:
            mismatches.append(f"{field}：資料庫實際值為「{gt_val}」，但你回報為「{llm_val}」")
    return len(mismatches) == 0, mismatches


def _append_verification_log(record: dict):
    with open(VERIFICATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _generate_report_async(incident_id: int, event_type: str):
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in tools_response.tools
            ]

            messages = [{"role": "user", "content": _build_user_prompt(incident_id)}]
            ground_truth = None
            full_text = ""
            iteration_records = []
            tool_was_called = False

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools,
                        "options": {"temperature": 0.0},
                        "stream": False,
                    },
                    timeout=60.0,
                )
                message = response.json().get("message", {})

                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call["function"]["name"] in (TOOL_NAME, "get_latest_incident_log"):
                            tool_was_called = True
                            # 【v5 關鍵修正】無論模型傳入的參數是什麼，這裡一律用
                            # 呼叫端已知的正確 incident_id 去查，從根本避免「查到別的
                            # 事件」這種競態條件；同時仍然要求模型必須真的發起工具呼叫，
                            # 這才是點5要驗證的重點（有沒有真的用工具，而非查對查錯）。
                            server_result = await session.call_tool(
                                TOOL_NAME, arguments={"incident_id": incident_id}
                            )
                            tool_text = server_result.content[0].text
                            try:
                                ground_truth = json.loads(tool_text)
                            except json.JSONDecodeError:
                                ground_truth = None
                            messages.append(message)
                            messages.append({
                                "role": "tool",
                                "content": tool_text,
                                "name": TOOL_NAME,
                            })
                else:
                    full_text = message.get("content", "")

                for iteration in range(1, MAX_ITERATIONS + 1):
                    if not tool_was_called and iteration == 1:
                        is_valid, mismatches = False, ["模型未呼叫 get_incident_by_id 工具"]
                    else:
                        final_response = await client.post(
                            OLLAMA_URL_CHAT,
                            json={
                                "model": MODEL_NAME,
                                "messages": messages,
                                "options": {"temperature": 0.0},
                                "stream": False,
                            },
                            timeout=60.0,
                        )
                        full_text = final_response.json().get("message", {}).get("content", "")
                        llm_json = _parse_verification_block(full_text)
                        is_valid, mismatches = _verify_fields(llm_json, ground_truth or {})

                    iteration_records.append({
                        "iteration": iteration,
                        "is_valid": is_valid,
                        "mismatches": mismatches,
                    })

                    if is_valid:
                        break

                    if iteration < MAX_ITERATIONS:
                        correction_msg = (
                            "你上一次的回答有欄位與實際工具查詢結果不符，請根據以下差異"
                            "重新產生完整報告（包含敘述文字與結尾的 json 驗證區塊）：\n"
                            + "\n".join(mismatches)
                        )
                        messages.append({"role": "assistant", "content": full_text})
                        messages.append({"role": "user", "content": correction_msg})

    final_valid = iteration_records[-1]["is_valid"] if iteration_records else False
    _append_verification_log({
        "incident_id": incident_id,
        "event_type": event_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_was_called": tool_was_called,
        "iterations_used": len(iteration_records),
        "final_valid": final_valid,
        "history": iteration_records,
    })

    narrative_text = _strip_verification_block(full_text)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{incident_id}_{timestamp_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🚨 事故歸因初報（事件 #{incident_id}）\n\n{narrative_text}")
        f.write(f"\n\n---\n**[系統驗證紀錄]** 經過 {len(iteration_records)} 輪，"
                 f"最終欄位驗證結果：{'通過 ✅' if final_valid else '未完全通過 ⚠️'}")

    print(f"\n📄 [報告生成完成] 已存至: {report_path}")
    print(f"🔍 [工具忠實度驗證] 共 {len(iteration_records)} 輪，"
          f"最終結果：{'通過' if final_valid else '未通過'}")
    if not final_valid and iteration_records:
        print(f"   最後一輪不符欄位：{iteration_records[-1]['mismatches']}")

    reference_text = REFERENCE_REPORTS.get(event_type)
    if reference_text and narrative_text.strip():
        try:
            _ensure_tokenizer_patch()
            from bert_score import score
            safe_candidate = narrative_text[:BERTSCORE_SAFE_LENGTH]
            safe_reference = reference_text[:BERTSCORE_SAFE_LENGTH]

            # 【v7 修正】上一版發現：套件在找不到 baseline 檔案時，不會丟出例外，
            # 只會印一個 UserWarning，然後直接跳過校準、悄悄回傳跟原始分數
            # 完全相同的數字。原本用 try/except 的判斷方式完全攔不到這種情況，
            # 會誤導成「已校準」但其實數值根本沒變。這裡改用 warnings 模組主動
            # 攔截這個警告文字，只要偵測到「Baseline not Found」，就明確標示
            # 校準不可用，而不是誤植一組看似校準過、實際上原封不動的數字。
            import warnings
            baseline_missing = False
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                rP, rR, rF1 = score(
                    [safe_candidate],
                    [safe_reference],
                    lang="zh",
                    model_type="hfl/chinese-roberta-wwm-ext",
                    num_layers=12,
                    device="cuda",
                    nthreads=1,
                    batch_size=1,
                    rescale_with_baseline=True,
                )
                for w in caught_warnings:
                    if "Baseline not Found" in str(w.message):
                        baseline_missing = True

            P, R, F1 = score(
                [safe_candidate],
                [safe_reference],
                lang="zh",
                model_type="hfl/chinese-roberta-wwm-ext",
                num_layers=12,
                device="cuda",
                nthreads=1,
                batch_size=1,
            )
            print(
                f"📊 [BERTScore 原始分數] Precision={P.item():.4f}  "
                f"Recall={R.item():.4f}  F1={F1.item():.4f}"
            )
            if baseline_missing:
                print(
                    "ℹ️ [BERTScore 校準] hfl/chinese-roberta-wwm-ext 沒有對應的"
                    " baseline 校準檔案，套件已自動跳過校準，"
                    "以上僅為未校準的原始分數。"
                )
            else:
                print(
                    f"📊 [BERTScore 校準後分數] Precision={rP.item():.4f}  "
                    f"Recall={rR.item():.4f}  F1={rF1.item():.4f}"
                )
                print(
                    "   （校準後分數扣掉了 BERT 詞向量空間的系統性偏差，"
                    "更能反映報告與參考答案之間真正的語意差異程度）"
                )
            print(
                f"   （註：評分僅取報告前 {BERTSCORE_SAFE_LENGTH} 字元，"
                "且已排除結尾的驗證用JSON區塊，只評分敘述正文）"
            )
        except Exception as e:
            import traceback
            print(f"⚠️ [BERTScore 評分失敗]: {e}")
            traceback.print_exc()
    else:
        print(f"⚠️ 事件類型「{event_type}」尚無對應的參考報告，略過 BERTScore 評分。")


def trigger_report_generation_async(incident_id: int, event_type: str):
    """把報告生成請求放進佇列，交給單一背景工作執行緒依序處理。
    這個函式本身會立刻返回、不會阻塞偵測迴圈；跟舊版最大差異是
    「不會再有事件因為撞到別的事件正在生成而被直接丟棄」。"""
    _ensure_worker_started()
    _report_queue.put((incident_id, event_type))
    print(
        f"\n📥 [報告佇列] incident_id={incident_id} 已排入報告生成佇列"
        f"（佇列中尚有 {_report_queue.unfinished_tasks} 筆待處理，含本筆）。"
    )


def get_pending_report_count() -> int:
    """回傳目前佇列中「還沒處理完」的報告數量（包含正在處理中的那一筆），
    供偵測端在結束時輪詢，判斷是否所有報告都已生成完畢。"""
    return _report_queue.unfinished_tasks


if __name__ == "__main__":
    asyncio.run(_generate_report_async(incident_id=1, event_type="black_smoky"))
