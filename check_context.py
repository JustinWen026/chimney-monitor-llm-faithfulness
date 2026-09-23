"""
check_context.py
----------------
檢查消融實驗的 prompt 是否曾被 Ollama 靜默截斷。

做法：用實驗腳本本身的 prompt 函式重建每一種呼叫的完整對話，各送給 Ollama 兩次：
  (A) 不指定 num_ctx（= 當初實驗的設定，使用 Ollama 預設值）
  (B) 指定很大的 num_ctx（得到完整、未截斷的 token 數）
若 (A) 的 prompt token 數明顯少於 (B)，代表當初那一種呼叫被截斷了。
每次只生成 1 個 token（num_predict=1），所以很快，不影響任何實驗結果。

放在專案資料夾（與 ablation_study_v14.py、ablation_study_multi_v2.py、
mcp_server_v3.py、db_manager_v3.py 同一層）執行：
    python check_context.py
"""

import asyncio
import json
import uuid

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import ablation_single as v14
import ablation_concurrent as mv2

try:
    import ablation_concurrent as mv3
except Exception:
    mv3 = None

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"
BIG_CTX = 16384            # 用來量「完整」長度的 context
OUTPUT_MARGIN = 1000       # 預留給模型輸出的 token（報告約 500～800 tokens）
SINGLE_ID = 54
MULTI_IDS = [49, 54, 48]
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

# 模擬「已經產生過的內容」長度（初步判斷約 150 字、一份報告約 1000 字）
PRELIM_TEXT = "初步情境判斷：" + "三筆事件均已由工具查得資料，" * 10
REPORT_TEXT = "### 綜合報告\n" + "事件摘要與歸因判定、風險等級、建議處置、追蹤建議。" * 40


def tool_call_msg(ids):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "get_incident_by_id",
                                         "arguments": {"incident_id": i}}} for i in ids]}


async def build_cases(session, tools):
    async def tool_result(i):
        r = await session.call_tool("get_incident_by_id", arguments={"incident_id": i})
        return {"role": "tool", "content": r.content[0].text, "name": "get_incident_by_id"}

    cases = []
    gt = v14.fetch_ground_truth(SINGLE_ID)
    single_tool = await tool_result(SINGLE_ID)
    cases += [
        ("單事故 Baseline", [{"role": "user", "content": v14.build_baseline_prompt(gt)}], None),
        ("單事故 Tool-only 第一次呼叫(含tools)",
         [{"role": "user", "content": v14._build_tool_only_prompt(SINGLE_ID, gt)}], tools),
        ("單事故 Tool-only 寫報告",
         [{"role": "user", "content": v14._build_tool_only_prompt(SINGLE_ID, gt)},
          tool_call_msg([SINGLE_ID]), single_tool], None),
        ("單事故 Proposed 寫報告",
         [{"role": "user", "content": v14._build_proposed_prompt(SINGLE_ID, gt)},
          tool_call_msg([SINGLE_ID]), single_tool], None),
        ("單事故 Proposed 重試(最長情況)",
         [{"role": "user", "content": v14._build_proposed_prompt(SINGLE_ID, gt)},
          tool_call_msg([SINGLE_ID]), single_tool,
          {"role": "assistant", "content": REPORT_TEXT},
          {"role": "user", "content": "你上一次的回答有欄位不符：['event_type']，請重新產生完整報告。"}], None),
    ]

    gts = [mv2.fetch_ground_truth_by_id(i) for i in MULTI_IDS]
    multi_tools = [await tool_result(i) for i in MULTI_IDS]
    v2_report_req = ("現在請根據稍早查詢到的所有事件資料，生成涵蓋所有事件的完整正式報告，"
                     "並在報告最後附上一個 ```json 區塊```。")
    v2_prop = [{"role": "user", "content": mv2._build_proposed_prompt_multi(MULTI_IDS)},
               tool_call_msg(MULTI_IDS), *multi_tools,
               {"role": "user", "content": "在正式生成報告前，請先用不超過150字，寫一段初步情境判斷。"},
               {"role": "assistant", "content": PRELIM_TEXT},
               {"role": "user", "content": v2_report_req}]
    cases += [
        ("多事故 v2 Baseline", [{"role": "user", "content": mv2.build_baseline_prompt(gts)}], None),
        ("多事故 v2 Tool-only 寫報告",
         [{"role": "user", "content": mv2._build_tool_only_prompt_multi(MULTI_IDS)},
          tool_call_msg(MULTI_IDS), *multi_tools], None),
        ("多事故 v2 Proposed 寫報告", v2_prop, None),
        ("多事故 v2 Proposed 重試(最長情況)",
         v2_prop + [{"role": "assistant", "content": REPORT_TEXT},
                    {"role": "user", "content": "你上一次的回答有以下問題：[...]，請重新產生完整報告。"},
                    {"role": "user", "content": v2_report_req}], None),
    ]

    if mv3 is not None:
        v3_prop = [{"role": "user", "content": mv3.build_proposed_prompt(gts)},
                   tool_call_msg(MULTI_IDS), *multi_tools,
                   {"role": "user", "content": "在正式生成報告前，請先用不超過150字，寫一段初步情境判斷。"},
                   {"role": "assistant", "content": PRELIM_TEXT},
                   {"role": "user", "content": mv3.report_request_prompt()}]
        cases += [
            ("多事故 v3 Tool-only 寫報告",
             [{"role": "user", "content": mv3.build_tool_only_prompt(gts)},
              tool_call_msg(MULTI_IDS), *multi_tools], None),
            ("多事故 v3 Proposed 重試(最長情況)",
             v3_prop + [{"role": "assistant", "content": REPORT_TEXT},
                        {"role": "user", "content": "你上一次的回答有以下問題：[...]，請修正。"},
                        {"role": "user", "content": mv3.report_request_prompt()}], None),
        ]
    return cases


async def count_tokens(client, messages, tools, num_ctx):
    # 開頭加一段隨機字串，避免 Ollama 的 prompt 快取讓計數變少
    msgs = [{"role": "system", "content": f"run {uuid.uuid4().hex}"}] + messages
    options = {"temperature": 0, "num_predict": 1}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = {"model": MODEL, "messages": msgs, "options": options, "stream": False}
    if tools:
        payload["tools"] = tools
    r = await client.post(f"{OLLAMA}/api/chat", json=payload, timeout=300.0)
    return r.json().get("prompt_eval_count")


async def main():
    async with httpx.AsyncClient() as client:
        version = (await client.get(f"{OLLAMA}/api/version")).json().get("version")
        show = (await client.post(f"{OLLAMA}/api/show", json={"model": MODEL})).json()
    params = show.get("parameters", "") or ""
    model_max = next((v for k, v in (show.get("model_info") or {}).items()
                      if k.endswith("context_length")), None)
    print(f"Ollama 版本：{version}")
    print(f"模型 {MODEL}：模型本身最大 context = {model_max}")
    print(f"Modelfile 參數：{params.strip() or '(無)'}")
    print(f"量化版本：{(show.get('details') or {}).get('quantization_level')}\n")

    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                       "parameters": t.inputSchema}}
                     for t in (await session.list_tools()).tools]
            cases = await build_cases(session, tools)

    results = []
    async with httpx.AsyncClient() as client:
        print("第一輪：不指定 num_ctx（與當初實驗相同）...")
        default_counts = [await count_tokens(client, m, t, None) for _, m, t in cases]
        print(f"第二輪：num_ctx={BIG_CTX}（量完整長度）...\n")
        full_counts = [await count_tokens(client, m, t, BIG_CTX) for _, m, t in cases]

    print(f"{'呼叫類型':<34}{'預設下tokens':>12}{'完整tokens':>12}   判定")
    print("-" * 78)
    for (name, _, _), d, f in zip(cases, default_counts, full_counts):
        if d is None or f is None:
            verdict = "無法取得計數"
        elif d < f - 20:
            verdict = f"⚠️ 被截斷（少了 {f - d} tokens）"
        else:
            verdict = "OK"
        results.append({"case": name, "default_prompt_tokens": d, "full_prompt_tokens": f, "verdict": verdict})
        print(f"{name:<34}{str(d):>12}{str(f):>12}   {verdict}")

    biggest = max((f or 0) for f in full_counts)
    print(f"\n最長的 prompt：{biggest} tokens；加上約 {OUTPUT_MARGIN} tokens 的輸出，需要 context ≥ {biggest + OUTPUT_MARGIN}。")
    print("若「預設下tokens」全部等於「完整tokens」，prompt 本身沒被截斷；")
    print("但如果最長 prompt + 輸出 超過預設 context，模型寫到一半仍可能被截掉，請一併回報。")
    json.dump({"ollama_version": version, "model_max_context": model_max, "modelfile_parameters": params,
               "results": results}, open("context_check_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n結果已存到 context_check_result.json")


if __name__ == "__main__":
    asyncio.run(main())
