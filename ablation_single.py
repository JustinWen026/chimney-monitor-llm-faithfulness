"""
ablation_study_v10.py
----------------------
針對目前 v10 架構（report_generator_v7 的工具驗證 + 迭代閉環機制）設計的消融實驗。

跟舊版 ablation_study_v6.py 的核心差異：
  - 舊版比較「有無 MCP 結構化輸入」對輸出格式完整度的影響，評分方式是
    關鍵字命中 + 內容長度，測的是「像不像一份報告」。
  - 這版比較「有無強制工具呼叫 + 欄位驗證 + 迭代重試」對「報告內容是否
    忠實反映資料庫真實記錄」的影響，這才是本研究真正要證明的主張：
    LLM 生成內容的可信度可以被驗證、也可以透過迭代機制被修正。

兩組使用同一筆真實資料庫事件（而非虛構數據）當作 ground truth，
確保比較是公平的、且跟正式系統一致。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python ablation_study_v10.py                 # 自動抓資料庫最新一筆事件
    python ablation_study_v10.py 46               # 指定 incident_id=46
"""

import asyncio
import json
import re
import sqlite3
import statistics
import sys
import time
from datetime import datetime

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from db_manager import DB_PATH

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"

# 兩組各跑幾次。原本 v6 用 N=30，這裡先保守設 N=10（Proposed 組每次都要
# 起一個 MCP 子行程 + 最多兩輪 LLM 呼叫，比舊版單純一次 generate 慢很多），
# 兩組共 20 次，跑完大概抓幾分鐘到十幾分鐘，時間允許再自行調高。
N_RUNS = 10
MAX_ITERATIONS = 2

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
VERIFICATION_FIELDS = ["incident_id", "camera_id", "event_type", "severity", "risk_level", "timestamp"]

# 事件類型對照中文關鍵詞，供 Baseline 組的自由文字輸出比對用
# （Baseline 組沒有強制輸出結構化 JSON，只能用關鍵字比對是否有提到正確類型）
EVENT_TYPE_KEYWORDS = {
    "fire": ["火災", "明火", "火焰", "起火"],
    "black_smoky": ["黑煙"],
    "white_smoky": ["白煙"],
    "black_smoky+fire": ["黑煙", "火"],
}

# 幻覺觸發詞：這些都是「系統裡實際上沒有提供」的細節類型，模型如果生成
# 這些內容，代表是自己編造的，不是根據真實資料而來。
HALLUCINATION_TRIGGERS = [
    "PM2.5", "SO2", "NOx", "ppm", "mg/m³", "風速", "wind speed",
    "CAM-03", "CAM-04", "email", "safety_manager", "巡檢員姓名",
]


def fetch_ground_truth(incident_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"資料庫找不到 incident_id={incident_id}，請換一個已存在的事件編號")
    return {
        "incident_id": row["id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "event_type": row["event_type"],
        "severity": row["severity"],
        "risk_level": row["risk_level"],
    }


def fetch_latest_incident_id() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM incidents ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise ValueError("資料庫目前沒有任何事件紀錄，請先讓系統累積至少一筆告警後再執行本實驗。")
    return row[0]


def score_hallucination(text: str) -> int:
    return sum(1 for kw in HALLUCINATION_TRIGGERS if kw.lower() in text.lower())


# ── Baseline 組：模擬「沒有工具、沒有驗證」的舊做法 ──────────────────────────

def build_baseline_prompt(gt: dict) -> str:
    return (
        "你是石化園區的安全專家。後台系統偵測到以下工安事件，"
        "請直接生成一份中文事故初報，內容包含事件摘要、歸因判定、"
        "風險等級、建議處置、追蹤建議：\n"
        f"[LOG] {gt['timestamp']} 攝影機 {gt['camera_id']} 偵測到異常，"
        f"事件類型 {gt['event_type']}，嚴重度 {gt['severity']}，"
        f"風險等級 {gt['risk_level']}，事件編號 {gt['incident_id']}。"
    )


def score_grounding_baseline(text: str, gt: dict) -> float:
    """Baseline 組沒有強制輸出結構化資料，只能用關鍵字比對輸出裡有沒有
    正確提到這幾個關鍵欄位，作為跟 Proposed 組的欄位驗證可比較的近似指標。"""
    checks = [
        gt["camera_id"] in text,
        str(gt["incident_id"]) in text,
        str(gt["risk_level"]) in text,
        any(kw in text for kw in EVENT_TYPE_KEYWORDS.get(gt["event_type"], [])),
    ]
    return sum(checks) / len(checks)


async def run_baseline_once(gt: dict) -> dict:
    prompt = build_baseline_prompt(gt)
    start = time.time()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.3},
                "stream": False,
            },
            timeout=60.0,
        )
    elapsed = time.time() - start
    text = resp.json().get("message", {}).get("content", "")
    return {
        "condition": "baseline",
        "elapsed": elapsed,
        "grounding_score": score_grounding_baseline(text, gt),
        "hallucination_count": score_hallucination(text),
        "iterations_used": 1,
    }


# ── Proposed 組：完整複刻 report_generator_v7 的工具驗證＋迭代邏輯 ──────────

def _build_proposed_prompt(incident_id: int) -> str:
    fields_hint = "、".join(VERIFICATION_FIELDS)
    return (
        f"【工安事件通報】本次目標事件的 incident_id 為 {incident_id}。"
        "請你【必須】立刻調用 `get_incident_by_id` 工具，"
        f"並傳入參數 incident_id={incident_id}，取得該筆事件的結構化資料，"
        "不允許自行猜測。取得資料後：\n"
        "1. 生成一份專業的中文事故初報，內容需包含：事件摘要、歸因判定、"
        "風險等級、建議處置、追蹤建議。\n"
        "2. 在報告最後，另起一段，附上一個 ```json 區塊```，"
        f"格式為包含以下欄位的 JSON 物件：{fields_hint}。"
        "這段 JSON 裡的每一個值，必須跟你剛剛從工具取得的原始資料完全一致、"
        "逐字複製，不可以自行改寫格式、翻譯、四捨五入或憑印象填寫。"
    )


async def run_proposed_once(gt: dict) -> dict:
    incident_id = gt["incident_id"]
    start = time.time()
    tool_was_called = False
    full_text = ""
    iteration_valid_flags = []

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema},
                }
                for t in tools_response.tools
            ]

            messages = [{"role": "user", "content": _build_proposed_prompt(incident_id)}]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools,
                        "options": {"temperature": 0.3},
                        "stream": False,
                    },
                    timeout=60.0,
                )
                message = response.json().get("message", {})

                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call["function"]["name"] == "get_incident_by_id":
                            tool_was_called = True
                            server_result = await session.call_tool(
                                "get_incident_by_id", arguments={"incident_id": incident_id}
                            )
                            tool_text = server_result.content[0].text
                            messages.append(message)
                            messages.append({
                                "role": "tool",
                                "content": tool_text,
                                "name": "get_incident_by_id",
                            })

                for iteration in range(1, MAX_ITERATIONS + 1):
                    if not tool_was_called and iteration == 1:
                        is_valid, mismatches = False, ["模型未呼叫 get_incident_by_id 工具"]
                    else:
                        final_response = await client.post(
                            OLLAMA_URL_CHAT,
                            json={
                                "model": MODEL_NAME,
                                "messages": messages,
                                "options": {"temperature": 0.3},
                                "stream": False,
                            },
                            timeout=60.0,
                        )
                        full_text = final_response.json().get("message", {}).get("content", "")
                        match = re.search(r"```json\s*(\{.*?\})\s*```", full_text, re.DOTALL)
                        llm_json = None
                        if match:
                            try:
                                llm_json = json.loads(match.group(1))
                            except json.JSONDecodeError:
                                llm_json = None

                        if llm_json is None:
                            is_valid, mismatches = False, ["缺少或無法解析驗證用JSON區塊"]
                        else:
                            mismatches = [
                                f for f in VERIFICATION_FIELDS
                                if str(gt.get(f, "")).strip() != str(llm_json.get(f, "")).strip()
                            ]
                            is_valid = len(mismatches) == 0

                    iteration_valid_flags.append(is_valid)
                    if is_valid:
                        break
                    if iteration < MAX_ITERATIONS:
                        messages.append({"role": "assistant", "content": full_text})
                        messages.append({
                            "role": "user",
                            "content": f"你上一次的回答有欄位不符：{mismatches}，請重新產生完整報告。",
                        })

    elapsed = time.time() - start
    narrative_text = re.sub(r"```json\s*\{.*?\}\s*```", "", full_text, flags=re.DOTALL)
    final_valid = iteration_valid_flags[-1] if iteration_valid_flags else False
    return {
        "condition": "proposed",
        "elapsed": elapsed,
        "grounding_score": 1.0 if final_valid else 0.0,
        "hallucination_count": score_hallucination(narrative_text),
        "iterations_used": len(iteration_valid_flags),
        "tool_was_called": tool_was_called,
    }


# ── 統計彙整與主流程 ─────────────────────────────────────────────────────────

def summarize(results: list) -> dict:
    def avg(k):
        return statistics.mean(r[k] for r in results)

    def std(k):
        return statistics.stdev(r[k] for r in results) if len(results) > 1 else 0.0

    return {
        "grounding_mean": avg("grounding_score"),
        "grounding_std": std("grounding_score"),
        "hallucination_mean": avg("hallucination_count"),
        "hallucination_std": std("hallucination_count"),
        "elapsed_mean": avg("elapsed"),
        "elapsed_std": std("elapsed"),
        "iterations_mean": avg("iterations_used"),
    }


async def main():
    if len(sys.argv) > 1:
        incident_id = int(sys.argv[1])
    else:
        incident_id = fetch_latest_incident_id()
        print(f"未指定 incident_id，自動使用資料庫最新一筆事件：#{incident_id}")

    gt = fetch_ground_truth(incident_id)

    print(f"\n{'=' * 70}")
    print(f"   消融實驗 v10 — 工具驗證 + 迭代閉環對報告忠實度的影響")
    print(f"   Ground Truth 事件：#{gt['incident_id']}  {gt['camera_id']}  "
          f"{gt['event_type']}  風險等級{gt['risk_level']}")
    print(f"   模型: {MODEL_NAME}  |  每組 N={N_RUNS}，隨機順序")
    print(f"{'=' * 70}")

    conditions = (["baseline"] * N_RUNS) + (["proposed"] * N_RUNS)
    import random
    random.shuffle(conditions)

    baseline_results, proposed_results = [], []
    for i, condition in enumerate(conditions):
        label = "Baseline（無工具/無驗證）" if condition == "baseline" else "Proposed（工具+驗證+迭代）"
        print(f"▶ Run {i + 1:02d}/{len(conditions)} [{label}]", flush=True)
        if condition == "baseline":
            r = await run_baseline_once(gt)
            baseline_results.append(r)
        else:
            r = await run_proposed_once(gt)
            proposed_results.append(r)
        print(
            f"   ✓ 時間={r['elapsed']:.2f}s | 忠實度={r['grounding_score']:.0%} | "
            f"幻覺數={r['hallucination_count']} | 迭代輪數={r['iterations_used']}"
        )

    b = summarize(baseline_results)
    p = summarize(proposed_results)

    print(f"\n\n{'=' * 78}")
    print(f"          消融實驗結果對比（N={N_RUNS} each）")
    print(f"          Ground Truth 事件：#{gt['incident_id']}（{gt['event_type']}）")
    print(f"{'=' * 78}")
    print(f"{'指標':<20} {'Baseline（無工具/無驗證）':<28} {'Proposed（工具+驗證+迭代）'}")
    print("-" * 78)
    print(f"{'★ 欄位忠實度':<20} {b['grounding_mean']:.1%}±{b['grounding_std']:.1%}"
          f"{'':<15} {p['grounding_mean']:.1%}±{p['grounding_std']:.1%}")
    print(f"{'★ 平均幻覺數':<20} {b['hallucination_mean']:.2f}±{b['hallucination_std']:.2f}"
          f"{'':<18} {p['hallucination_mean']:.2f}±{p['hallucination_std']:.2f}")
    print(f"{'平均耗時':<20} {b['elapsed_mean']:.2f}±{b['elapsed_std']:.2f} s"
          f"{'':<15} {p['elapsed_mean']:.2f}±{p['elapsed_std']:.2f} s")
    print(f"{'平均迭代輪數':<20} {b['iterations_mean']:.2f}"
          f"{'':<24} {p['iterations_mean']:.2f}")
    print(f"{'=' * 78}\n")
    print("💡 欄位忠實度：報告內容裡的關鍵欄位（攝影機、事件類型、風險等級、")
    print("   事件編號）是否跟資料庫真實記錄一致。Baseline 組因為沒有結構化")
    print("   驗證機制，用關鍵字比對做近似評分；Proposed 組是精確的欄位比對。")
    print("   平均耗時的差異是可預期的 trade-off：Proposed 組多了工具呼叫與")
    print("   驗證迴圈，用時間換取忠實度與可驗證性，這正是本研究的核心論點。\n")

    output = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "n_runs_each": N_RUNS,
        "ground_truth_incident": gt,
        "baseline_summary": b,
        "proposed_summary": p,
        "raw_baseline": baseline_results,
        "raw_proposed": proposed_results,
    }
    out_path = r"D:\NCKU_paper_Capstone_Project\reports\ablation_results_v10.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {out_path}\n")
    except Exception:
        # 若不在 Windows 專案路徑下執行（例如先在其他環境測試語法），
        # 退回存在目前工作目錄，不讓實驗結果因路徑問題而遺失。
        fallback_path = "ablation_results_v10.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {fallback_path}（預設路徑不可寫入，已退回目前目錄）\n")


if __name__ == "__main__":
    asyncio.run(main())
