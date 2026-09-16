"""
ablation_study_multi_v2.py
----------------------------
【跟 ablation_study_multi（第一版）的差異】

第一版有一個結構性bug：驗證失敗、要求模型「重新產生完整報告」的那次
API呼叫沒有帶tools參數，如果失敗原因是「漏查了某筆事件」，模型此時已經
沒有能力真的去補查，重試變成「叫它道歉，但沒給它修正的工具」。v2把
「查工具」跟「寫報告」合併進同一個迭代迴圈，確保每一輪重試模型都真的
有機會呼叫工具補齊缺漏的資料，讓迭代重試機制名符其實。

v2 另外新增：
  - mismatches_history：完整記錄每一輪驗證失敗的具體原因，方便直接
    在論文裡引用「修正前錯在哪、修正後有沒有真的修好」的具體案例。
  - 輸出檔名自動帶時間戳記，不會再每次執行就覆蓋掉上一次的結果。

【跟 ablation_study_v14/v17 系列的核心差異，這部分v2沿用不變】

v14/v17 系列測的是「一次只給一筆事件,LLM 能不能忠實複製這一筆的欄位」。
這版測的是完全不同的問題：「同時給 LLM 好幾筆事件,它會不會把不同事件的
欄位搞混、張冠李戴」——這更貼近真實工安場景，因為現實中很可能同時有
好幾支攝影機觸發告警，系統得同時處理、彙整成一份綜合報告，而不是永遠
乾淨地一次只面對一筆事件。

這版同時保留 v17 已經驗證過的三個難度提升手段（擴充驗證欄位、拉開推理
距離、相似干擾值），temperature 固定 0.8，疊加上「多筆事件同時處理」
這第四個難度來源，測試在這種更貼近真實情境的壓力下，迭代重試機制
會不會被真正觸發。

三組對照設計（Baseline / Tool-only / Proposed）與分組計分的幻覺量測邏輯，
沿用 v14/v17 的既有做法不變，差異只在於「這次要處理的不是一筆事件，
是同時 N_INCIDENTS 筆」。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python ablation_study_multi_v2.py                  # 自動抓資料庫最新 N 筆事件
    python ablation_study_multi_v2.py 52,54,55          # 指定 incident_id 清單（逗號分隔），
                                                          # 建議挑三種不同事件類型，
                                                          # 不要挑到同類型的雙胞胎事件
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

# 三組各跑幾次。多筆事件同時處理的 prompt 長、呼叫次數也比單筆版多，
# 一次跑完會比 v17 慢不少，先用小 N 測過流程再拉大。
N_RUNS = 50
MAX_ITERATIONS = 2

# 同時處理幾筆事件。3 筆是一個合理的起點：夠讓「搞混」有機會發生，
# 又不會讓 prompt 長到超出合理範圍。
N_INCIDENTS = 3

# 【沿用v17】固定使用較高溫度，疊加難度測試迭代重試會不會被觸發。
TEMPERATURE = 0.8

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

# 【沿用v17】8個驗證欄位，而不是最初版本的6個。
VERIFICATION_FIELDS = [
    "incident_id", "camera_id", "event_type", "severity",
    "risk_level", "timestamp", "duration_seconds", "status",
]

EVENT_TYPE_KEYWORDS = {
    "fire": ["火災", "火灾", "明火", "火焰", "起火", "fire"],
    "black_smoky": ["黑煙", "黑烟", "black_smoky", "black smoke"],
    "white_smoky": ["白煙", "白烟", "白色煙霧", "白色烟雾", "white_smoky", "white smoke"],
    "black_smoky+fire": ["黑煙", "黑烟", "black_smoky", "複合", "复合", "black_smoky+fire"],
}

FAKE_WIND_SPEED = "12.4 m/s"
FAKE_EMAIL = "safety_backup@petrochem.com"
FAKE_TS_1 = "13:25:00"
FAKE_TS_2 = "13:47:02"
FAKE_FRAME_RATE = "29.8 FPS"

# 共用（不隨事件變動）的雜訊分組，跟 v17 相同。
SHARED_HALLUCINATION_TRIGGER_GROUPS = {
    "假風速": [FAKE_WIND_SPEED],
    "假空氣品質數據": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
    "假email通知": [FAKE_EMAIL],
    "假時間戳雜訊": [FAKE_TS_1, FAKE_TS_2],
    "假影像處理參數": [FAKE_FRAME_RATE, "contrast"],
}


# ── 資料庫存取：一次抓多筆事件 ─────────────────────────────────────────────

def fetch_ground_truth_by_id(incident_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"資料庫找不到 incident_id={incident_id}")
    return {
        "incident_id": row["id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "event_type": row["event_type"],
        "severity": row["severity"],
        "risk_level": row["risk_level"],
        "duration_seconds": row["duration"],
        "status": row["status"],
    }


def fetch_default_incident_ids(n: int) -> list:
    """自動抓資料庫最新 n 筆、且 id 各不相同的事件。"""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM incidents ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
    if len(rows) < n:
        raise ValueError(
            f"資料庫目前只有 {len(rows)} 筆事件，不足以抓取 {n} 筆做多事件測試，"
            "請先讓系統累積更多告警紀錄，或降低 N_INCIDENTS。"
        )
    return [row[0] for row in rows]


# ── 幻覺計分：分組計分，含每筆事件各自的相似干擾攝影機編號 ────────────────

def _get_trigger_groups(gts: list) -> dict:
    groups = dict(SHARED_HALLUCINATION_TRIGGER_GROUPS)
    for gt in gts:
        groups[f"假相似干擾攝影機編號_{gt['incident_id']}"] = [f"{gt['camera_id']}B"]
    return groups


def score_hallucination(text: str, gts: list) -> int:
    text_lower = text.lower()
    return sum(
        1 for keywords in _get_trigger_groups(gts).values()
        if any(kw.lower() in text_lower for kw in keywords)
    )


def get_hallucination_groups_hit(text: str, gts: list) -> list:
    text_lower = text.lower()
    return [
        name for name, keywords in _get_trigger_groups(gts).items()
        if any(kw.lower() in text_lower for kw in keywords)
    ]


# ── 建立多事件雜訊日誌 ──────────────────────────────────────────────────────

def build_noisy_log_multi(gts: list) -> str:
    """把 N 筆事件的真實資料，跟一組共用雜訊、以及「每筆事件各自的相似
    干擾攝影機編號」混在同一段日誌裡。多筆事件夾雜在一起本身就是一種
    額外的難度來源：模型必須自己分辨哪一行日誌屬於哪一筆事件，不能靠
    「反正只有一筆」的捷徑蒙混過去。"""
    lines = [
        f"[INFO] {FAKE_TS_1} 系統心跳正常，Zone 巡檢無異常回報。",
        f"[WARN] 現場風速感測器讀值為 {FAKE_WIND_SPEED}，偏強陣風，可能影響煙流擴散方向。",
        f"[INFO] 空氣品質監測：SO2 濃度 0.08 ppm，NOx 濃度 0.15 ppm，均在正常範圍。",
    ]
    for gt in gts:
        decoy_camera_id = f"{gt['camera_id']}B"
        lines.append(
            f"[INFO] {decoy_camera_id}（備援機位，非事發攝影機）短暫回報畫面雜訊，"
            "經覆核為感測器雜訊，與本次事件無關。"
        )
        lines.append(
            f"[CRITICAL] {gt['timestamp']} 狀態機觸發告警。攝影機 {gt['camera_id']} "
            f"偵測到 {gt['event_type']} 異常，嚴重度 {gt['severity']}，"
            f"風險等級 {gt['risk_level']}，持續時間 {gt['duration_seconds']} 秒，"
            f"目前狀態 {gt['status']}，事件編號 {gt['incident_id']}。"
        )
    lines.append(f"[INFO] {FAKE_TS_2} 系統已自動寄送通知至 {FAKE_EMAIL}。")
    lines.append(f"[DEBUG] 背景相減演算法運行穩定，frame rate {FAKE_FRAME_RATE}。")
    return "\n".join(lines)


# ── Baseline 組：無工具、無驗證，一次生成涵蓋所有事件的報告 ─────────────────

def build_baseline_prompt(gts: list) -> str:
    noisy_log = build_noisy_log_multi(gts)
    return (
        "你是石化園區的安全專家。以下是後台系統的原始日誌，內含多筆事件混雜"
        "在一起，請從中判讀出所有工安事件，針對每一筆事件分別生成事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議：\n"
        + noisy_log
    )


def score_grounding_baseline(text: str, gts: list) -> float:
    """對每一筆事件各自檢查三項核心事實（攝影機、風險等級、事件類型）
    是否被正確提及，回傳所有事件、所有檢查項目的平均命中率。"""
    all_checks = []
    for gt in gts:
        all_checks.append(gt["camera_id"] in text)
        all_checks.append(str(gt["risk_level"]) in text)
        all_checks.append(any(kw in text for kw in EVENT_TYPE_KEYWORDS.get(gt["event_type"], [])))
    return sum(all_checks) / len(all_checks) if all_checks else 0.0


async def run_baseline_once(gts: list) -> dict:
    prompt = build_baseline_prompt(gts)
    start = time.time()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": TEMPERATURE},
                "stream": False,
            },
            timeout=180.0,
        )
    elapsed = time.time() - start
    text = resp.json().get("message", {}).get("content", "")
    return {
        "condition": "baseline",
        "elapsed": elapsed,
        "grounding_score": score_grounding_baseline(text, gts),
        "hallucination_count": score_hallucination(text, gts),
        "hallucination_keywords_hit": get_hallucination_groups_hit(text, gts),
        "iterations_used": 1,
        "output_text": text,
    }


# ── Proposed 組：多次工具呼叫 + 拉開推理距離 + 逐筆逐欄位驗證 + 迭代重試 ────

def _build_proposed_prompt_multi(incident_ids: list) -> str:
    fields_hint = "、".join(VERIFICATION_FIELDS)
    ids_str = "、".join(str(i) for i in incident_ids)
    return (
        f"【工安事件通報】後台系統同時偵測到多筆工安事件，目標事件的 "
        f"incident_id 分別為：{ids_str}。以下是後台系統的原始日誌，"
        "內含多筆事件混雜在一起，可能包含雜訊或無關資訊：\n\n"
        f"請你【必須】針對每一個 incident_id 分別呼叫 `get_incident_by_id` "
        "工具（總共應呼叫多次，每次傳入不同的 incident_id），取得每一筆事件"
        "各自的權威結構化資料（若與日誌內容有出入，一律以工具查詢結果為準，"
        "不可採信日誌裡其他無關的雜訊內容，也不可以把不同事件的資料搞混）。"
        "取得所有事件的資料後：\n"
        "1. 生成一份涵蓋所有事件的綜合報告，針對每一筆事件分別說明事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議。\n"
        "2. 在報告最後，另起一段，附上一個 ```json 區塊```，內容為一個 JSON "
        f"陣列，陣列中每個元素對應一筆事件，格式為包含以下欄位的 JSON 物件："
        f"{fields_hint}。陣列元素數量必須等於事件筆數，每個元素裡的每一個值，"
        "都必須跟該筆事件對應工具取得的原始資料完全一致、逐字複製，不可以"
        "自行改寫格式、翻譯、四捨五入、憑印象填寫，也不可以把不同事件的"
        "欄位值張冠李戴。"
    )


def _verify_multi(llm_json_array, gts: list):
    """比對模型輸出的 JSON 陣列跟真實的多筆 ground truth。除了逐欄位比對，
    也檢查事件筆數對不對、有沒有漏掉或重複某個 incident_id——這是多事件
    情境下特有的錯誤模式，單筆驗證測不到。"""
    if not isinstance(llm_json_array, list):
        return False, ["輸出的JSON不是陣列格式"]

    mismatches = []
    if len(llm_json_array) != len(gts):
        mismatches.append(f"事件筆數不符：預期{len(gts)}筆，實際輸出{len(llm_json_array)}筆")

    gt_by_id = {gt["incident_id"]: gt for gt in gts}
    seen_ids = set()
    for entry in llm_json_array:
        entry_id = entry.get("incident_id")
        try:
            entry_id_int = int(entry_id)
        except (TypeError, ValueError):
            mismatches.append(f"某筆輸出的incident_id無法辨識：{entry_id}")
            continue
        if entry_id_int in seen_ids:
            mismatches.append(f"incident_id={entry_id_int} 重複出現")
            continue
        seen_ids.add(entry_id_int)

        gt = gt_by_id.get(entry_id_int)
        if gt is None:
            mismatches.append(f"incident_id={entry_id_int} 不在本次目標事件清單內（可能是編造或張冠李戴）")
            continue

        for f in VERIFICATION_FIELDS:
            if str(gt.get(f, "")).strip() != str(entry.get(f, "")).strip():
                mismatches.append(
                    f"incident_id={entry_id_int} 的欄位「{f}」：應為「{gt.get(f, '')}」，實際為「{entry.get(f, '')}」"
                )

    missing_ids = set(gt_by_id.keys()) - seen_ids
    for missing_id in missing_ids:
        mismatches.append(f"incident_id={missing_id} 完全沒有出現在輸出中")

    return len(mismatches) == 0, mismatches


async def run_proposed_once(gts: list) -> dict:
    """【重要修正】舊版有一個結構性bug：如果第一輪模型漏查了某筆事件的
    工具，系統要求「重新產生完整報告」時，那次 API 呼叫沒有帶 tools 參數，
    等於模型此時已經沒有能力真的去補查漏掉的資料，只能瞎掰或承認做不到，
    重試變成「叫它道歉，但沒給它修正的工具」。這版把「查工具」跟「寫報告」
    合併進同一個迴圈：每一輪迭代都會先確保工具查完（查不完會一直嘗試，
    上限是事件筆數+1次），查完才進入寫報告+驗證的步驟；驗證失敗要重試時，
    下一輪一樣會先確認工具，模型如果被回饋『你漏查了某筆事件』，這次
    真的有機會去把它補上，而不是被綁著手只能口頭道歉。"""
    incident_ids = [gt["incident_id"] for gt in gts]
    start = time.time()
    tool_called_ids = set()
    full_text = ""
    iteration_valid_flags = []
    mismatches_history = []

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

            messages = [{"role": "user", "content": _build_proposed_prompt_multi(incident_ids)}]
            prelim_done = False

            async with httpx.AsyncClient() as client:
                for iteration in range(1, MAX_ITERATIONS + 1):
                    # ── 步驟A：確保這一輪的工具都查完，帶著tools參數重試最多
                    #    len(incident_ids)+1次，讓模型有充分機會把所有事件查完 ──
                    for _ in range(len(incident_ids) + 1):
                        if len(tool_called_ids) >= len(incident_ids):
                            break
                        gather_response = await client.post(
                            OLLAMA_URL_CHAT,
                            json={
                                "model": MODEL_NAME,
                                "messages": messages,
                                "tools": available_tools,
                                "options": {"temperature": TEMPERATURE},
                                "stream": False,
                            },
                            timeout=180.0,
                        )
                        gather_message = gather_response.json().get("message", {})
                        if gather_message.get("tool_calls"):
                            messages.append(gather_message)
                            for tool_call in gather_message["tool_calls"]:
                                if tool_call["function"]["name"] == "get_incident_by_id":
                                    args = tool_call["function"].get("arguments", {})
                                    try:
                                        called_id_int = int(args.get("incident_id"))
                                    except (TypeError, ValueError):
                                        called_id_int = None
                                    if called_id_int is not None and called_id_int not in tool_called_ids:
                                        server_result = await session.call_tool(
                                            "get_incident_by_id", arguments={"incident_id": called_id_int}
                                        )
                                        tool_called_ids.add(called_id_int)
                                        messages.append({
                                            "role": "tool",
                                            "content": server_result.content[0].text,
                                            "name": "get_incident_by_id",
                                        })
                        else:
                            missing_now = [i for i in incident_ids if i not in tool_called_ids]
                            if not missing_now:
                                break
                            messages.append({"role": "assistant", "content": gather_message.get("content", "")})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"你還沒有查詢以下 incident_id 的資料，請繼續呼叫 "
                                    f"`get_incident_by_id` 工具查詢：{', '.join(str(i) for i in missing_now)}"
                                ),
                            })

                    all_tools_called = len(tool_called_ids) >= len(incident_ids)
                    missing_ids = [i for i in incident_ids if i not in tool_called_ids]

                    # ── 步驟B：工具都查完才寫報告；查不完就直接判失敗，
                    #    不浪費一次生成報告的呼叫 ──
                    if not all_tools_called:
                        is_valid, mismatches = False, [f"模型未查詢完所有事件，缺少：{missing_ids}"]
                    else:
                        # 【拉開推理距離，沿用v17】只在第一次做，之後重試直接
                        # 根據既有對話紀錄要求重新產生報告，不用重複問前情提要。
                        if not prelim_done:
                            prelim_prompt = (
                                "在正式生成報告前，請先用不超過150字，寫一段初步情境判斷"
                                "（純文字敘述，不需要包含JSON），簡述你從工具查詢結果看到"
                                "了這幾筆事件的重點。"
                            )
                            messages.append({"role": "user", "content": prelim_prompt})
                            prelim_response = await client.post(
                                OLLAMA_URL_CHAT,
                                json={
                                    "model": MODEL_NAME,
                                    "messages": messages,
                                    "options": {"temperature": TEMPERATURE},
                                    "stream": False,
                                },
                                timeout=180.0,
                            )
                            prelim_text = prelim_response.json().get("message", {}).get("content", "")
                            messages.append({"role": "assistant", "content": prelim_text})
                            prelim_done = True

                        fields_hint = "、".join(VERIFICATION_FIELDS)
                        messages.append({
                            "role": "user",
                            "content": (
                                "現在請根據稍早查詢到的所有事件資料，生成涵蓋所有事件的"
                                "完整正式報告，並在報告最後附上一個 ```json 區塊```，內容"
                                "為一個 JSON 陣列，陣列中每個元素對應一筆事件，格式為包含"
                                f"以下欄位的 JSON 物件：{fields_hint}。每個元素裡的每一個值"
                                "都必須跟對應事件的工具查詢結果完全一致、逐字複製，不可"
                                "張冠李戴、不可遺漏任何一筆事件。"
                            ),
                        })
                        final_response = await client.post(
                            OLLAMA_URL_CHAT,
                            json={
                                "model": MODEL_NAME,
                                "messages": messages,
                                "options": {"temperature": TEMPERATURE},
                                "stream": False,
                            },
                            timeout=180.0,
                        )
                        full_text = final_response.json().get("message", {}).get("content", "")
                        match = re.search(r"```json\s*(\[.*?\])\s*```", full_text, re.DOTALL)
                        llm_json_array = None
                        if match:
                            try:
                                llm_json_array = json.loads(match.group(1))
                            except json.JSONDecodeError:
                                llm_json_array = None

                        if llm_json_array is None:
                            is_valid, mismatches = False, ["缺少或無法解析驗證用JSON陣列區塊"]
                        else:
                            is_valid, mismatches = _verify_multi(llm_json_array, gts)

                    mismatches_history.append({"iteration": iteration, "is_valid": is_valid, "mismatches": mismatches})
                    iteration_valid_flags.append(is_valid)
                    if is_valid:
                        break
                    if iteration < MAX_ITERATIONS:
                        messages.append({"role": "assistant", "content": full_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"你上一次的回答有以下問題：{mismatches}，請重新產生完整報告"
                                "（若問題是缺少某筆事件的資料，請先呼叫工具補查）。"
                            ),
                        })

    elapsed = time.time() - start
    narrative_text = re.sub(r"```json\s*\[.*?\]\s*```", "", full_text, flags=re.DOTALL)
    final_valid = iteration_valid_flags[-1] if iteration_valid_flags else False
    return {
        "condition": "proposed",
        "elapsed": elapsed,
        "grounding_score": 1.0 if final_valid else 0.0,
        "hallucination_count": score_hallucination(narrative_text, gts),
        "hallucination_keywords_hit": get_hallucination_groups_hit(narrative_text, gts),
        "iterations_used": len(iteration_valid_flags),
        "mismatches_history": mismatches_history,
        "tools_called_count": len(tool_called_ids),
        "expected_tools_count": len(incident_ids),
        "output_text": narrative_text,
    }


# ── Tool-only 組：多次工具呼叫、不驗證、不重試 ──────────────────────────────

def _build_tool_only_prompt_multi(incident_ids: list) -> str:
    ids_str = "、".join(str(i) for i in incident_ids)
    return (
        f"【工安事件通報】後台系統同時偵測到多筆工安事件，目標事件的 "
        f"incident_id 分別為：{ids_str}。請你針對每一筆事件分別呼叫 "
        "`get_incident_by_id` 工具取得權威結構化資料（若與日誌內容有出入，"
        "一律以工具查詢結果為準，不可採信日誌裡其他無關的雜訊內容），並生成"
        "一份涵蓋所有事件的綜合中文事故初報，針對每一筆事件分別說明事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議。"
    )


async def run_tool_only_once(gts: list) -> dict:
    """中間組：對每筆事件各呼叫一次工具、不驗證、不重試，單次生成即結束。"""
    incident_ids = [gt["incident_id"] for gt in gts]
    start = time.time()
    tool_called_ids = set()
    full_text = ""

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
            messages = [{"role": "user", "content": _build_tool_only_prompt_multi(incident_ids)}]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools,
                        "options": {"temperature": TEMPERATURE},
                        "stream": False,
                    },
                    timeout=180.0,
                )
                message = response.json().get("message", {})

                if message.get("tool_calls"):
                    messages.append(message)
                    for tool_call in message["tool_calls"]:
                        if tool_call["function"]["name"] == "get_incident_by_id":
                            args = tool_call["function"].get("arguments", {})
                            called_id = args.get("incident_id")
                            try:
                                called_id_int = int(called_id)
                            except (TypeError, ValueError):
                                called_id_int = None
                            if called_id_int is not None:
                                server_result = await session.call_tool(
                                    "get_incident_by_id", arguments={"incident_id": called_id_int}
                                )
                                tool_called_ids.add(called_id_int)
                                messages.append({
                                    "role": "tool",
                                    "content": server_result.content[0].text,
                                    "name": "get_incident_by_id",
                                })

                final_response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "options": {"temperature": TEMPERATURE},
                        "stream": False,
                    },
                    timeout=180.0,
                )
                full_text = final_response.json().get("message", {}).get("content", "")

    elapsed = time.time() - start
    return {
        "condition": "tool_only",
        "elapsed": elapsed,
        "grounding_score": score_grounding_baseline(full_text, gts),
        "hallucination_count": score_hallucination(full_text, gts),
        "hallucination_keywords_hit": get_hallucination_groups_hit(full_text, gts),
        "iterations_used": 1,
        "tools_called_count": len(tool_called_ids),
        "expected_tools_count": len(incident_ids),
        "output_text": full_text,
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


async def warmup():
    print("🔥 [熱身] 載入模型至 VRAM...", flush=True)
    async with httpx.AsyncClient() as client:
        await client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "你好"}],
                "options": {"temperature": 0.0},
                "stream": False,
            },
            timeout=60.0,
        )
    print("   ✅ 模型就緒\n", flush=True)


async def main():
    if len(sys.argv) > 1:
        incident_ids = [int(x.strip()) for x in sys.argv[1].split(",")]
    else:
        incident_ids = fetch_default_incident_ids(N_INCIDENTS)
        print(f"未指定 incident_id 清單，自動使用資料庫最新 {N_INCIDENTS} 筆事件：{incident_ids}")

    gts = [fetch_ground_truth_by_id(i) for i in incident_ids]

    # 【v2新增】每次執行都用時間戳記產生獨一無二的檔名，不會再覆蓋掉
    # 上一次跑的結果，之後要回頭比較「這次跟上次差在哪」不用再自己
    # 手動搬檔案備份。
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 74}")
    print(f"   消融實驗 multi_v2 — 多筆事件同時驗證，測試是否會張冠李戴、能否觸發迭代重試")
    print(f"   Ground Truth 事件：{[(g['incident_id'], g['event_type']) for g in gts]}")
    print(f"   模型: {MODEL_NAME}  |  temperature={TEMPERATURE}  |  每組 N={N_RUNS}，共 {N_RUNS * 3} 次")
    print(f"{'=' * 74}")

    conditions = (["baseline"] * N_RUNS) + (["tool_only"] * N_RUNS) + (["proposed"] * N_RUNS)
    import random
    random.shuffle(conditions)

    await warmup()

    condition_labels = {
        "baseline": "Baseline（無工具/無驗證）",
        "tool_only": "Tool-only（多次工具呼叫/無驗證重試）",
        "proposed": "Proposed（多次工具呼叫+跨事件驗證+迭代）",
    }
    condition_funcs = {
        "baseline": run_baseline_once,
        "tool_only": run_tool_only_once,
        "proposed": run_proposed_once,
    }

    results_by_condition = {"baseline": [], "tool_only": [], "proposed": []}
    failed_runs = []
    incremental_path = (
        r"D:\NCKU_paper_Capstone_Project\reports"
        f"\\ablation_results_multi_v2_{run_timestamp}_partial.json"
    )

    for i, condition in enumerate(conditions):
        print(f"▶ Run {i + 1:02d}/{len(conditions)} [{condition_labels[condition]}]", flush=True)
        try:
            # 【修正】單次呼叫的逾時或任何其他例外，不能讓整個實驗陪葬——
            # 之前遇到的狀況是：Run01成功了，Run02逾時直接把整個
            # asyncio.run(main()) 炸掉，連Run01已經跑出來的資料都沒機會
            # 存檔就沒了。現在把每一次呼叫包起來，失敗就記錄下來、跳到
            # 下一筆，不會拖垮整批實驗。
            r = await condition_funcs[condition](gts)
            results_by_condition[condition].append(r)
            print(
                f"   ✓ 時間={r['elapsed']:.2f}s | 忠實度={r['grounding_score']:.0%} | "
                f"幻覺數={r['hallucination_count']} | 迭代輪數={r['iterations_used']}"
            )
        except Exception as e:
            failed_runs.append({"run_index": i + 1, "condition": condition, "error": str(e)})
            print(f"   ❌ 這一筆失敗，已跳過（原因：{type(e).__name__}: {e}），繼續下一筆", flush=True)

        # 每跑完一筆就即時存檔一次，就算後面真的整個崩潰，前面的資料也不會不見。
        try:
            with open(incremental_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"results_so_far": results_by_condition, "failed_runs": failed_runs},
                    f, ensure_ascii=False, indent=2,
                )
        except Exception:
            pass  # 中繼存檔失敗不影響主流程，正式結果最後還是會存一次。

    if failed_runs:
        print(f"\n⚠️ 本次實驗共有 {len(failed_runs)} 筆執行失敗，已跳過，"
              f"最終統計只計入成功的樣本數（可能造成三組樣本數不完全相等）：")
        for fr in failed_runs:
            print(f"   Run {fr['run_index']:02d} [{fr['condition']}]: {fr['error']}")
        print()

    for condition in ["baseline", "tool_only", "proposed"]:
        if not results_by_condition[condition]:
            raise RuntimeError(
                f"[{condition}] 組所有樣本都失敗了，沒有資料可以統計，"
                "請檢查 Ollama 服務是否正常運作，或考慮再拉長 timeout 後重跑。"
            )

    b = summarize(results_by_condition["baseline"])
    t = summarize(results_by_condition["tool_only"])
    p = summarize(results_by_condition["proposed"])

    print(f"\n\n{'=' * 96}")
    print(f"          多事件消融實驗結果對比（N={N_RUNS} each，共 {N_RUNS * 3} 次，同時處理{len(gts)}筆事件）")
    print(f"          Ground Truth 事件：{[g['incident_id'] for g in gts]}")
    print(f"{'=' * 96}")
    print(f"{'指標':<18} {'Baseline':<20} {'Tool-only':<20} {'Proposed':<20}")
    print("-" * 96)
    print(f"{'★ 欄位忠實度':<18} {b['grounding_mean']:.1%}±{b['grounding_std']:.1%}"
          f"{'':<10} {t['grounding_mean']:.1%}±{t['grounding_std']:.1%}"
          f"{'':<10} {p['grounding_mean']:.1%}±{p['grounding_std']:.1%}")
    print(f"{'★ 平均幻覺數':<18} {b['hallucination_mean']:.2f}±{b['hallucination_std']:.2f}"
          f"{'':<13} {t['hallucination_mean']:.2f}±{t['hallucination_std']:.2f}"
          f"{'':<13} {p['hallucination_mean']:.2f}±{p['hallucination_std']:.2f}")
    print(f"{'平均耗時':<18} {b['elapsed_mean']:.2f}±{b['elapsed_std']:.2f} s"
          f"{'':<10} {t['elapsed_mean']:.2f}±{t['elapsed_std']:.2f} s"
          f"{'':<10} {p['elapsed_mean']:.2f}±{p['elapsed_std']:.2f} s")
    print(f"{'平均迭代輪數':<18} {b['iterations_mean']:.2f}"
          f"{'':<19} {t['iterations_mean']:.2f}"
          f"{'':<19} {p['iterations_mean']:.2f}")
    print(f"{'=' * 96}\n")
    print("💡 這版最關鍵的觀察指標是 iterations_mean 有沒有終於超過1.00，")
    print("   以及 raw_proposed 裡的 mismatches 記錄裡，有沒有出現「incident_id")
    print("   不在清單內」「筆數不符」這類跨事件搞混的錯誤模式，這是單筆版本")
    print("   完全測不到的錯誤類型。\n")

    output = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_runs_each_requested": N_RUNS,
        "n_runs_each_actual": {
            "baseline": len(results_by_condition["baseline"]),
            "tool_only": len(results_by_condition["tool_only"]),
            "proposed": len(results_by_condition["proposed"]),
        },
        "failed_runs": failed_runs,
        "n_incidents": len(gts),
        "ground_truth_incidents": gts,
        "methodology_scope_note": (
            "本測試在v14/v17已驗證過的三組對照設計、分組計分幻覺量測邏輯、"
            "擴充驗證欄位、拉開推理距離、相似干擾攝影機編號、temperature=0.8"
            "等既有設計基礎上，新增『同時處理多筆事件』作為第四個難度來源，"
            "測試模型是否會在多事件情境下發生跨事件張冠李戴。樣本數仍小"
            f"（每組請求N={N_RUNS}，實際成功筆數見上方 n_runs_each_actual），"
            "若有執行失敗（如逾時）之樣本已排除在統計外，可能造成三組樣本數"
            "不完全相等，結果為初步觀察，不宣稱涵蓋真實世界所有情境。"
        ),
        "baseline_summary": b,
        "tool_only_summary": t,
        "proposed_summary": p,
        "raw_baseline": results_by_condition["baseline"],
        "raw_tool_only": results_by_condition["tool_only"],
        "raw_proposed": results_by_condition["proposed"],
    }
    out_path = (
        r"D:\NCKU_paper_Capstone_Project\reports"
        f"\\ablation_results_multi_v2_{run_timestamp}.json"
    )
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {out_path}\n")
    except Exception:
        fallback_path = f"ablation_results_multi_v2_{run_timestamp}.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {fallback_path}（預設路徑不可寫入，已退回目前目錄）\n")


if __name__ == "__main__":
    asyncio.run(main())