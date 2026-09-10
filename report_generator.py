import asyncio
import json
import os
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

# 【v4 修正】原本 report_generator_v3.py 呼叫的是 "mcp_server.py"，
# 但那支檔案有 import bug（見 mcp_server_v2.py 的說明），一律改指向修正後的版本。
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

BERTSCORE_SAFE_LENGTH = 200

# 【點5、6新增】迭代閉環的最大重試次數，對應 ADK LoopAgent 的 max_iterations 概念。
MAX_ITERATIONS = 2

# 驗證用結構化區塊必須包含的欄位，會跟 MCP 工具實際回傳的資料逐一比對。
VERIFICATION_FIELDS = ["incident_id", "camera_id", "event_type", "severity", "risk_level", "timestamp"]

VERIFICATION_LOG_PATH = os.path.join(REPORT_DIR, "verification_log.jsonl")

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

_report_lock = threading.Lock()

_tokenizer_patched = False


def _ensure_tokenizer_patch():
    """沿用 v3 已驗證有效的修正：強制把 tokenizer 的 model_max_length 改成 512，
    避開 hfl/chinese-roberta-wwm-ext 預設值溢位導致的 OverflowError。"""
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


def _build_user_prompt():
    """【點5核心】在原本的指令之外，明確要求 LLM 在報告最後附上一段結構化 JSON，
    JSON 裡的每個欄位值必須跟它剛剛透過工具查到的原始資料完全一致、逐字複製，
    不可以重新排版、四捨五入或憑印象改寫。這段 JSON 就是我們拿來驗證「工具結果
    有沒有被忠實使用」的依據，而不是靠人工肉眼檢查報告內容像不像。"""
    fields_hint = "、".join(VERIFICATION_FIELDS)
    return (
        "【工安事件通報】後台狀態機已觸發告警，請你【必須】立刻調用 "
        "`get_latest_incident_log` 工具取得最新事件的結構化資料，"
        "不允許自行猜測。取得資料後：\n"
        "1. 生成一份專業的中文事故初報，內容需包含：事件摘要、歸因判定、"
        "風險等級、建議處置、追蹤建議。\n"
        "2. 在報告最後，另起一段，附上一個 ```json 區塊```，"
        f"格式為包含以下欄位的 JSON 物件：{fields_hint}。"
        "這段 JSON 裡的每一個值，必須跟你剛剛從工具取得的原始資料完全一致、"
        "逐字複製，不可以自行改寫格式、翻譯、四捨五入或憑印象填寫。"
    )


def _parse_verification_block(text: str):
    """從 LLM 輸出中擷取 ```json ... ``` 區塊並解析成 dict，失敗回傳 None。"""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _strip_verification_block(text: str) -> str:
    """把驗證用 JSON 區塊從正文移除，避免污染 BERTScore 的語意評分
    （JSON 格式的文字跟中文敘述报告放在一起比對語意相似度沒有意義）。"""
    return re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()


def _verify_fields(llm_json, ground_truth: dict):
    """逐欄位比對 LLM 自報的 JSON 跟系統實際查到的資料庫原始值是否一致。
    回傳 (是否全部通過, 不符欄位清單)。"""
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

            messages = [{"role": "user", "content": _build_user_prompt()}]
            ground_truth = None
            full_text = ""
            iteration_records = []

            async with httpx.AsyncClient() as client:
                # 第一次呼叫：要求模型決定是否調用工具
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
                        if tool_call["function"]["name"] == "get_latest_incident_log":
                            server_result = await session.call_tool(
                                "get_latest_incident_log", arguments={}
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
                                "name": "get_latest_incident_log",
                            })
                else:
                    # 模型完全沒有呼叫工具，直接視為驗證失敗（點5要驗證的正是這種情況）
                    full_text = message.get("content", "")

                # 【點6核心】驗證-重試迴圈：exit_condition 為「所有關鍵欄位比對通過」
                for iteration in range(1, MAX_ITERATIONS + 1):
                    if ground_truth is None and iteration == 1 and full_text:
                        # 第一輪就沒呼叫工具，記錄失敗但仍照原文往下走一次重試
                        is_valid, mismatches = False, ["模型未呼叫 get_latest_incident_log 工具"]
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
                f"📊 [BERTScore] Precision={P.item():.4f}  "
                f"Recall={R.item():.4f}  F1={F1.item():.4f}"
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
    if not _report_lock.acquire(blocking=False):
        print("\n⚠️ [報告生成] 上一份報告仍在生成中，跳過本次觸發。")
        return

    def run():
        try:
            asyncio.run(_generate_report_async(incident_id, event_type))
        finally:
            _report_lock.release()

    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    asyncio.run(_generate_report_async(incident_id=1, event_type="black_smoky"))
