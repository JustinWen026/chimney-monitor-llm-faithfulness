import asyncio
import os
import threading
from datetime import datetime

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"
REPORT_DIR = r"D:\NCKU_paper_Capstone_Project\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

BERTSCORE_SAFE_LENGTH = 200

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

# ==============================================================================
# 【BERTScore 根本修正】
# 真正病灶：hfl/chinese-roberta-wwm-ext 的 tokenizer 設定沒有明確指定最大長度，
# Hugging Face 會用一個代表「無上限」的超大整數頂替，這個數字傳進底層 enable_truncation
# 時會直接溢位。use_fast_tokenizer=False 這個參數在目前版本沒有確實生效，
# 所以改用更直接的做法：攔截 AutoTokenizer.from_pretrained 這個載入函式，
# 载入完成後強制把 model_max_length 改成合理的 512，不管走哪條內部路徑都會
# 先經過這層修正，徹底避開溢位問題。
# ==============================================================================
_tokenizer_patched = False


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

            messages = [{
                "role": "user",
                "content": (
                    "【工安事件通報】後台狀態機已觸發告警，請你【必須】立刻調用 "
                    "`get_latest_incident_log` 工具取得最新事件的結構化資料，"
                    "不允許自行猜測，取得資料後生成一份專業的中文事故初報，"
                    "內容需包含：事件摘要、歸因判定、風險等級、建議處置、追蹤建議。"
                ),
            }]

            full_text = ""
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
                        if tool_call["function"]["name"] == "get_latest_incident_log":
                            server_result = await session.call_tool(
                                "get_latest_incident_log", arguments={}
                            )
                            messages.append(message)
                            messages.append({
                                "role": "tool",
                                "content": server_result.content[0].text,
                                "name": "get_latest_incident_log",
                            })
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
                            full_text = (
                                final_response.json().get("message", {}).get("content", "")
                            )
                else:
                    full_text = message.get("content", "")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{incident_id}_{timestamp_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🚨 事故歸因初報（事件 #{incident_id}）\n\n{full_text}")

    print(f"\n📄 [報告生成完成] 已存至: {report_path}")

    reference_text = REFERENCE_REPORTS.get(event_type)
    if reference_text and full_text.strip():
        try:
            _ensure_tokenizer_patch()
            from bert_score import score
            safe_candidate = full_text[:BERTSCORE_SAFE_LENGTH]
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
                "受套件長文本限制所致，通常涵蓋事件摘要與歸因判定部分）"
            )
        except Exception as e:
            import traceback
            print(f"⚠️ [BERTScore 評分失敗]: {e}")
            print("完整錯誤追蹤（若問題持續，請把以下內容整段貼給我排查）：")
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
