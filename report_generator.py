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

# ==============================================================================
# 各事件類型的標準參考報告（給 BERTScore 當作 ground truth 使用）
#
# 這是「你認為理想中的報告應該長怎樣」的人工範本，語意上要涵蓋：
# 事件摘要 / 歸因判定 / 風險等級 / 建議處置 / 追蹤建議 五個要素。
# 建議你之後照自己想要的措辭調整這幾段文字，寫得越貼近你理想中的標準答案，
# BERTScore 算出來的分數才越有意義（分數本質上是「跟這份參考文本的語意相似度」）。
# ==============================================================================
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


async def _generate_report_async(incident_id: int, event_type: str):
    """透過 MCP 取得真實結構化事件資料，呼叫本地 LLM 生成事故初報"""
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

    # 有對應事件類型的參考報告，就順便跑一次 BERTScore 評分
    reference_text = REFERENCE_REPORTS.get(event_type)
    if reference_text and full_text.strip():
        try:
            from bert_score import score
            # 保險截斷，避免超長文本觸發套件的長度溢位問題
            safe_candidate = full_text[:800]
            safe_reference = reference_text[:800]
            P, R, F1 = score(
                [safe_candidate],
                [safe_reference],
                lang="zh",
                model_type="hfl/chinese-roberta-wwm-ext",
                num_layers=12,
                device="cuda",
            )
            print(
                f"📊 [BERTScore] Precision={P.item():.4f}  "
                f"Recall={R.item():.4f}  F1={F1.item():.4f}"
            )
        except Exception as e:
            print(f"⚠️ [BERTScore 評分失敗]: {e}")
    else:
        print(f"⚠️ 事件類型「{event_type}」尚無對應的參考報告，略過 BERTScore 評分。")


def trigger_report_generation_async(incident_id: int, event_type: str):
    """外部接口：非阻塞方式在背景生成報告，不影響主推論迴圈 FPS"""
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
    # 獨立測試：假設資料庫裡已經有至少一筆事件紀錄
    asyncio.run(_generate_report_async(incident_id=1, event_type="black_smoky"))
