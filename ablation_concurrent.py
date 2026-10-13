"""
ablation_study_multi_v3.py
--------------------------
【v3 相對 multi_v2 的修改】

1. 修正 bug：multi_v2 中 Tool-only 與 Proposed 的 prompt 寫了「以下是後台系統的
   原始日誌」，但沒有真的接上日誌，所以這兩組從未接觸到雜訊。v3 讓三組收到
   完全相同的雜訊日誌（build_noisy_log_multi），多事故的雜訊比較才有意義。

2. 「先寫 150 字初步判斷」這一步改成開關 USE_PRELIM_STEP：
   - False（預設）：兩組都沒有這一步。
   - True：Tool-only 與 Proposed 都有這一步，讓兩組只差在
     「JSON 要求 + coverage 提示 + 驗證 + 重試」。
   依正式系統（report_generator）是否有這一步來決定。

3. 計分同時輸出「原始規則」與「修正後規則」（與 rescore.py 相同，規則在執行前寫死）：
   - 攝影機：真實編號後不能緊接英數字（CAM-01B 不算 CAM-01）。
   - 風險等級：數字必須緊鄰「風險等級／风险等级／risk level」等字樣。
   - 事件類型：補齊中文同義詞；檢查 fire 前先移除 "black_smoky+fire" 字串。
   - 雜訊：依來源分組；相同的誘餌攝影機編號只算一組。
   Proposed 的敘事分數直接在這裡算（移除 JSON 區塊後的文字），不需要另外的腳本。

4. 新增紀錄：
   - 每次 Ollama 呼叫的 prompt_eval_count、eval_count、done_reason（檢查截斷）；
   - Proposed 的 coverage 提示次數（「你還沒有查詢…」送出幾次）；
   - Ollama 版本、num_ctx、執行順序。

5. 明確設定 num_ctx（NUM_CTX），避免依賴 Ollama 預設值而發生靜默截斷。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python ablation_study_multi_v3.py 49,54,48
"""

import asyncio
import json
import os
import random
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

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL_CHAT = f"{OLLAMA_BASE}/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"

N_RUNS = 80
MAX_ITERATIONS = 2
N_INCIDENTS = 3
TEMPERATURE = 0.8

# 明確設定 context 長度，避免 Ollama 預設值太小而靜默截斷 prompt 開頭。
NUM_CTX = 8192
OLLAMA_OPTIONS = {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}

# 「先寫初步判斷」這一步：True 時 Tool-only 與 Proposed 都做，False 時都不做。
USE_PRELIM_STEP = False

# 執行順序的亂數種子（None = 不固定，但實際順序會存進結果檔）。
RANDOM_SEED = None

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

VERIFICATION_FIELDS = [
    "incident_id", "camera_id", "event_type", "severity",
    "risk_level", "timestamp", "duration_seconds", "status",
]

FAKE_WIND_SPEED = "12.4 m/s"
FAKE_EMAIL = "safety_backup@petrochem.com"
FAKE_TS_1 = "13:25:00"
FAKE_TS_2 = "13:47:02"
FAKE_FRAME_RATE = "29.8 FPS"

SHARED_NOISE_GROUPS = {
    "wind": [FAKE_WIND_SPEED],
    "air_quality": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
    "email": [FAKE_EMAIL],
    "timestamps": [FAKE_TS_1, FAKE_TS_2],
    "image_params": [FAKE_FRAME_RATE, "contrast"],
}


# ── 計分：原始規則（與 multi_v2 相同，保留作對照）──────────────────────────
OLD_EVENT_KW = {
    "fire": ["火災", "火灾", "明火", "火焰", "起火", "fire"],
    "black_smoky": ["黑煙", "黑烟", "black_smoky", "black smoke"],
    "white_smoky": ["白煙", "白烟", "白色煙霧", "白色烟雾", "white_smoky", "white smoke"],
    "black_smoky+fire": ["黑煙", "黑烟", "black_smoky", "複合", "复合", "black_smoky+fire"],
}


def old_checks(text, gt):
    return [
        gt["camera_id"] in text,
        str(gt["risk_level"]) in text,
        any(kw in text for kw in OLD_EVENT_KW.get(gt["event_type"], [])),
    ]


# ── 計分：修正後規則（與 rescore.py 相同）──────────────────────────────────
FIRE_TERMS = ["火災", "火灾", "明火", "火焰", "起火", "火警", "火情", "著火", "着火",
              "失火", "燃燒", "燃烧", "火勢", "火势", "火光", "火源", "火苗", "fire"]
BLACK_TERMS = ["黑煙", "黑烟", "黑色煙", "黑色烟", "black_smoky", "black smoke"]
WHITE_TERMS = ["白煙", "白烟", "白色煙", "白色烟", "white_smoky", "white smoke"]
COMPOSITE_TERMS = ["black_smoky+fire", "複合", "复合"]
BLACK_AND_FIRE = re.compile(r"黑[煙烟](?:[和與与及、跟]|以及|並伴有|并伴有|伴隨|伴随|伴有)火")
RISK_PATTERNS = [
    re.compile(r"(?:[風风][險险](?:等[級级]|[級级][別别]|[層层][級级]|程度)|"
               r"risk[\s_-]*level)[^\d\n]{0,10}?([1-5])(?![\d.])", re.IGNORECASE),
    re.compile(r"(?<![\d.])([1-5])\s*[級级]\s*[風风][險险]"),
]


def _has(text, terms):
    low = text.lower()
    return any(t.lower() in low for t in terms)


def _event_hit(text, event_type):
    stripped = re.sub(r"black_smoky\+fire", "", text, flags=re.IGNORECASE)
    if event_type == "fire":
        return _has(stripped, FIRE_TERMS)
    if event_type == "black_smoky":
        return _has(text, BLACK_TERMS)
    if event_type == "white_smoky":
        return _has(text, WHITE_TERMS)
    if event_type == "black_smoky+fire":
        return (_has(text, COMPOSITE_TERMS) or BLACK_AND_FIRE.search(text) is not None
                or (_has(text, BLACK_TERMS) and _has(stripped, FIRE_TERMS)))
    return False


def _risk_digits(text):
    found = set()
    for p in RISK_PATTERNS:
        found.update(int(m.group(1)) for m in p.finditer(text))
    return found


def new_checks(text, gt):
    return [
        re.search(re.escape(gt["camera_id"]) + r"(?![0-9A-Za-z])", text) is not None,
        int(gt["risk_level"]) in _risk_digits(text),
        _event_hit(text, gt["event_type"]),
    ]


def narrative_scores(text, gts):
    """回傳原始規則與修正後規則的敘事提及率。注意：仍是全文比對，
    無法判斷事實寫在哪一筆事件底下。"""
    old, new = [], []
    for gt in gts:
        old += old_checks(text, gt)
        new += new_checks(text, gt)
    return {
        "narrative_old": sum(old) / len(old),
        "narrative_new": sum(new) / len(new),
        "all_correct_new": all(new),
    }


def noise_groups(gts):
    groups = dict(SHARED_NOISE_GROUPS)
    for cam in sorted({gt["camera_id"] for gt in gts}):  # 相同誘餌編號只算一組
        groups[f"decoy_{cam}B"] = [f"{cam}B"]
    return groups


def noise_hits(text, gts):
    low = text.lower()
    return [name for name, kws in noise_groups(gts).items() if any(k.lower() in low for k in kws)]


# ── 資料庫 ─────────────────────────────────────────────────────────────────
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
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM incidents ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
    if len(rows) < n:
        raise ValueError(f"資料庫目前只有 {len(rows)} 筆事件，不足 {n} 筆。")
    return [row[0] for row in rows]


# ── 雜訊日誌（三組完全相同）────────────────────────────────────────────────
def build_noisy_log_multi(gts: list) -> str:
    lines = [
        f"[INFO] {FAKE_TS_1} 系統心跳正常，Zone 巡檢無異常回報。",
        f"[WARN] 現場風速感測器讀值為 {FAKE_WIND_SPEED}，偏強陣風，可能影響煙流擴散方向。",
        "[INFO] 空氣品質監測：SO2 濃度 0.08 ppm，NOx 濃度 0.15 ppm，均在正常範圍。",
    ]
    for gt in gts:
        lines.append(
            f"[INFO] {gt['camera_id']}B（備援機位，非事發攝影機）短暫回報畫面雜訊，"
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


# ── Ollama 呼叫（統一記錄 token 數與結束原因）──────────────────────────────
async def ollama_chat(client, messages, call_log, step, tools=None):
    payload = {"model": MODEL_NAME, "messages": messages, "options": OLLAMA_OPTIONS, "stream": False}
    if tools:
        payload["tools"] = tools
    resp = await client.post(OLLAMA_URL_CHAT, json=payload, timeout=180.0)
    data = resp.json()
    call_log.append({
        "step": step,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    })
    return data.get("message", {})


def call_summary(call_log):
    prompts = [c["prompt_eval_count"] for c in call_log if c["prompt_eval_count"] is not None]
    return {
        "max_prompt_tokens": max(prompts) if prompts else None,
        "any_length_stop": any(c["done_reason"] == "length" for c in call_log),
    }


async def prelim_step(client, messages, call_log):
    messages.append({"role": "user", "content": (
        "在正式生成報告前，請先用不超過150字，寫一段初步情境判斷（純文字敘述，"
        "不需要包含JSON），簡述你從工具查詢結果看到了這幾筆事件的重點。")})
    msg = await ollama_chat(client, messages, call_log, "prelim")
    messages.append({"role": "assistant", "content": msg.get("content", "")})


def tool_schema(tools_response):
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
        for t in tools_response.tools
    ]


def result_record(condition, text, gts, elapsed, call_log, **extra):
    rec = {
        "condition": condition,
        "elapsed": elapsed,
        **narrative_scores(text, gts),
        "noise_mention_count": len(noise_hits(text, gts)),
        "noise_groups_hit": noise_hits(text, gts),
        "calls": call_log,
        **call_summary(call_log),
        "output_text": text,
    }
    rec.update(extra)
    return rec


# ── Baseline ───────────────────────────────────────────────────────────────
def build_baseline_prompt(gts: list) -> str:
    return (
        "你是石化園區的安全專家。以下是後台系統的原始日誌，內含多筆事件混雜"
        "在一起，請從中判讀出所有工安事件，針對每一筆事件分別生成事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議：\n"
        + build_noisy_log_multi(gts)
    )


async def run_baseline_once(gts: list) -> dict:
    call_log = []
    start = time.time()
    async with httpx.AsyncClient() as client:
        msg = await ollama_chat(client, [{"role": "user", "content": build_baseline_prompt(gts)}],
                                call_log, "report")
    text = msg.get("content", "")
    return result_record("baseline", text, gts, time.time() - start, call_log, iterations_used=1)


# ── Tool-only ──────────────────────────────────────────────────────────────
def build_tool_only_prompt(gts: list) -> str:
    ids_str = "、".join(str(gt["incident_id"]) for gt in gts)
    return (
        "【工安事件通報】以下是後台系統的原始日誌，內含多筆事件混雜在一起，"
        "可能包含雜訊或無關資訊：\n"
        f"{build_noisy_log_multi(gts)}\n\n"
        f"目標事件的 incident_id 分別為：{ids_str}。請你針對每一筆事件分別呼叫 "
        "`get_incident_by_id` 工具取得權威結構化資料（若與日誌內容有出入，"
        "一律以工具查詢結果為準，不可採信日誌裡其他無關的雜訊內容），並生成"
        "一份涵蓋所有事件的綜合中文事故初報，針對每一筆事件分別說明事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議。"
    )


async def run_tool_only_once(gts: list) -> dict:
    """單輪工具呼叫、不驗證、不重試、沒有 coverage 提示。"""
    incident_ids = [gt["incident_id"] for gt in gts]
    call_log, tool_called_ids = [], set()
    start = time.time()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = tool_schema(await session.list_tools())
            messages = [{"role": "user", "content": build_tool_only_prompt(gts)}]
            async with httpx.AsyncClient() as client:
                msg = await ollama_chat(client, messages, call_log, "tool_call", tools=tools)
                if msg.get("tool_calls"):
                    messages.append(msg)
                    for tc in msg["tool_calls"]:
                        if tc["function"]["name"] != "get_incident_by_id":
                            continue
                        try:
                            cid = int(tc["function"].get("arguments", {}).get("incident_id"))
                        except (TypeError, ValueError):
                            continue
                        result = await session.call_tool("get_incident_by_id", arguments={"incident_id": cid})
                        tool_called_ids.add(cid)
                        messages.append({"role": "tool", "content": result.content[0].text,
                                         "name": "get_incident_by_id"})
                if USE_PRELIM_STEP:
                    await prelim_step(client, messages, call_log)
                    messages.append({"role": "user", "content": "現在請生成涵蓋所有事件的完整正式報告。"})
                final = await ollama_chat(client, messages, call_log, "report")
    text = final.get("content", "")
    return result_record(
        "tool_only", text, gts, time.time() - start, call_log,
        iterations_used=1,
        tools_called_count=len(tool_called_ids & set(incident_ids)),
        expected_tools_count=len(incident_ids),
    )


# ── Proposed ───────────────────────────────────────────────────────────────
def build_proposed_prompt(gts: list) -> str:
    ids_str = "、".join(str(gt["incident_id"]) for gt in gts)
    return (
        "【工安事件通報】以下是後台系統的原始日誌，內含多筆事件混雜在一起，"
        "可能包含雜訊或無關資訊：\n"
        f"{build_noisy_log_multi(gts)}\n\n"
        f"目標事件的 incident_id 分別為：{ids_str}。"
        "請你【必須】針對每一個 incident_id 分別呼叫 `get_incident_by_id` "
        "工具（總共應呼叫多次，每次傳入不同的 incident_id），取得每一筆事件"
        "各自的權威結構化資料（若與日誌內容有出入，一律以工具查詢結果為準，"
        "不可採信日誌裡其他無關的雜訊內容，也不可以把不同事件的資料搞混）。"
        # 以下與 multi_v2 的指示文字相同
        "取得所有事件的資料後：\n"
        "1. 生成一份涵蓋所有事件的綜合報告，針對每一筆事件分別說明事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議。\n"
        "2. 在報告最後，另起一段，附上一個 ```json 區塊```，內容為一個 JSON "
        f"陣列，陣列中每個元素對應一筆事件，格式為包含以下欄位的 JSON 物件："
        f"{'、'.join(VERIFICATION_FIELDS)}。陣列元素數量必須等於事件筆數，每個元素裡的每一個值，"
        "都必須跟該筆事件對應工具取得的原始資料完全一致、逐字複製，不可以"
        "自行改寫格式、翻譯、四捨五入、憑印象填寫，也不可以把不同事件的"
        "欄位值張冠李戴。"
    )


def report_request_prompt() -> str:
    # 與 multi_v2 相同
    return (
        "現在請根據稍早查詢到的所有事件資料，生成涵蓋所有事件的"
        "完整正式報告，並在報告最後附上一個 ```json 區塊```，內容"
        "為一個 JSON 陣列，陣列中每個元素對應一筆事件，格式為包含"
        f"以下欄位的 JSON 物件：{'、'.join(VERIFICATION_FIELDS)}。每個元素裡的每一個值"
        "都必須跟對應事件的工具查詢結果完全一致、逐字複製，不可"
        "張冠李戴、不可遺漏任何一筆事件。"
    )


def verify_multi(arr, gts):
    if not isinstance(arr, list):
        return False, ["輸出的JSON不是陣列格式"]
    mismatches = []
    if len(arr) != len(gts):
        mismatches.append(f"事件筆數不符：預期{len(gts)}筆，實際輸出{len(arr)}筆")
    gt_by_id = {gt["incident_id"]: gt for gt in gts}
    seen = set()
    for entry in arr:
        if not isinstance(entry, dict):
            mismatches.append(f"陣列元素不是物件：{entry}")
            continue
        try:
            eid = int(entry.get("incident_id"))
        except (TypeError, ValueError):
            mismatches.append(f"某筆輸出的incident_id無法辨識：{entry.get('incident_id')}")
            continue
        if eid in seen:
            mismatches.append(f"incident_id={eid} 重複出現")
            continue
        seen.add(eid)
        gt = gt_by_id.get(eid)
        if gt is None:
            mismatches.append(f"incident_id={eid} 不在本次目標事件清單內")
            continue
        for f in VERIFICATION_FIELDS:
            if str(gt.get(f, "")).strip() != str(entry.get(f, "")).strip():
                mismatches.append(f"incident_id={eid} 的欄位「{f}」：應為「{gt.get(f, '')}」，實際為「{entry.get(f, '')}」")
    for mid in set(gt_by_id) - seen:
        mismatches.append(f"incident_id={mid} 完全沒有出現在輸出中")
    return len(mismatches) == 0, mismatches


async def run_proposed_once(gts: list) -> dict:
    """coverage 迴圈（查不完會提示補查）→（選用）初步判斷 → 報告＋JSON → 驗證 → 最多重試一次。"""
    incident_ids = [gt["incident_id"] for gt in gts]
    call_log, tool_called_ids = [], set()
    valid_flags, mismatches_history = [], []
    coverage_nudges, first_round_ids = 0, None
    full_text = ""
    start = time.time()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = tool_schema(await session.list_tools())
            messages = [{"role": "user", "content": build_proposed_prompt(gts)}]
            prelim_done = False

            async with httpx.AsyncClient() as client:
                for iteration in range(1, MAX_ITERATIONS + 1):
                    for _ in range(len(incident_ids) + 1):
                        if len(tool_called_ids) >= len(incident_ids):
                            break
                        msg = await ollama_chat(client, messages, call_log, "tool_call", tools=tools)
                        if msg.get("tool_calls"):
                            messages.append(msg)
                            for tc in msg["tool_calls"]:
                                if tc["function"]["name"] != "get_incident_by_id":
                                    continue
                                try:
                                    cid = int(tc["function"].get("arguments", {}).get("incident_id"))
                                except (TypeError, ValueError):
                                    continue
                                if cid in tool_called_ids:
                                    continue
                                result = await session.call_tool("get_incident_by_id", arguments={"incident_id": cid})
                                tool_called_ids.add(cid)
                                messages.append({"role": "tool", "content": result.content[0].text,
                                                 "name": "get_incident_by_id"})
                        else:
                            missing = [i for i in incident_ids if i not in tool_called_ids]
                            if not missing:
                                break
                            coverage_nudges += 1
                            messages.append({"role": "assistant", "content": msg.get("content", "")})
                            messages.append({"role": "user", "content": (
                                "你還沒有查詢以下 incident_id 的資料，請繼續呼叫 "
                                f"`get_incident_by_id` 工具查詢：{', '.join(str(i) for i in missing)}")})
                        if first_round_ids is None:
                            first_round_ids = sorted(tool_called_ids & set(incident_ids))

                    missing = [i for i in incident_ids if i not in tool_called_ids]
                    if missing:
                        is_valid, mismatches = False, [f"模型未查詢完所有事件，缺少：{missing}"]
                    else:
                        if USE_PRELIM_STEP and not prelim_done:
                            await prelim_step(client, messages, call_log)
                            prelim_done = True
                        messages.append({"role": "user", "content": report_request_prompt()})
                        final = await ollama_chat(client, messages, call_log, f"report_iter{iteration}")
                        full_text = final.get("content", "")
                        match = re.search(r"```json\s*(\[.*?\])\s*```", full_text, re.DOTALL)
                        arr = None
                        if match:
                            try:
                                arr = json.loads(match.group(1))
                            except json.JSONDecodeError:
                                arr = None
                        if arr is None:
                            is_valid, mismatches = False, ["缺少或無法解析驗證用JSON陣列區塊"]
                        else:
                            is_valid, mismatches = verify_multi(arr, gts)

                    mismatches_history.append({"iteration": iteration, "is_valid": is_valid, "mismatches": mismatches})
                    valid_flags.append(is_valid)
                    if is_valid:
                        break
                    if iteration < MAX_ITERATIONS:
                        messages.append({"role": "assistant", "content": full_text})
                        messages.append({"role": "user", "content": (
                            f"你上一次的回答有以下問題：{mismatches}，請重新產生完整報告"
                            "（若問題是缺少某筆事件的資料，請先呼叫工具補查）。")})

    narrative = re.sub(r"```json\s*\[.*?\]\s*```", "", full_text, flags=re.DOTALL)
    return result_record(
        "proposed", narrative, gts, time.time() - start, call_log,
        structured_valid=bool(valid_flags and valid_flags[-1]),
        iterations_used=len(valid_flags),
        mismatches_history=mismatches_history,
        coverage_nudges=coverage_nudges,
        first_round_tools=first_round_ids,
        tools_called_count=len(tool_called_ids & set(incident_ids)),
        expected_tools_count=len(incident_ids),
    )


# ── 彙整與主流程 ───────────────────────────────────────────────────────────
def summarize(results: list) -> dict:
    def mean(k):
        vals = [r[k] for r in results if r.get(k) is not None]
        return statistics.mean(vals) if vals else None

    def sd(k):
        vals = [r[k] for r in results if r.get(k) is not None]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    s = {k: mean(k) for k in ["narrative_old", "narrative_new", "noise_mention_count",
                              "elapsed", "iterations_used", "max_prompt_tokens"]}
    s.update({f"{k}_sd": sd(k) for k in ["narrative_old", "narrative_new", "noise_mention_count", "elapsed"]})
    s["all_correct_new_rate"] = statistics.mean(1.0 if r["all_correct_new"] else 0.0 for r in results)
    s["runs_with_length_stop"] = sum(1 for r in results if r["any_length_stop"])
    s["max_prompt_tokens_overall"] = max((r["max_prompt_tokens"] or 0) for r in results)
    if results and "structured_valid" in results[0]:
        s["structured_valid_rate"] = statistics.mean(1.0 if r["structured_valid"] else 0.0 for r in results)
        s["coverage_nudges_mean"] = mean("coverage_nudges")
        s["runs_needing_nudge"] = sum(1 for r in results if r["coverage_nudges"] > 0)
    if results and "tools_called_count" in results[0]:
        s["runs_under_queried"] = sum(1 for r in results if r["tools_called_count"] < r["expected_tools_count"])
    return s


async def ollama_version():
    try:
        async with httpx.AsyncClient() as client:
            return (await client.get(f"{OLLAMA_BASE}/api/version", timeout=10.0)).json().get("version")
    except Exception:
        return None


async def warmup():
    print("🔥 [熱身] 載入模型至 VRAM...", flush=True)
    async with httpx.AsyncClient() as client:
        await client.post(OLLAMA_URL_CHAT, json={
            "model": MODEL_NAME, "messages": [{"role": "user", "content": "你好"}],
            "options": {"temperature": 0.0, "num_ctx": NUM_CTX}, "stream": False}, timeout=120.0)
    print("   ✅ 模型就緒\n", flush=True)


def save_json(obj, path, fallback):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return fallback


async def main():
    if len(sys.argv) > 1:
        incident_ids = [int(x.strip()) for x in sys.argv[1].split(",")]
    else:
        incident_ids = fetch_default_incident_ids(N_INCIDENTS)
        print(f"未指定 incident_id，自動使用資料庫最新 {N_INCIDENTS} 筆：{incident_ids}")
    gts = [fetch_ground_truth_by_id(i) for i in incident_ids]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = await ollama_version()

    print(f"\n{'=' * 74}\n   消融實驗 multi_v3（三組皆收到相同雜訊日誌）")
    print(f"   事件：{[(g['incident_id'], g['event_type']) for g in gts]}")
    print(f"   模型 {MODEL_NAME} | temperature={TEMPERATURE} | num_ctx={NUM_CTX} | "
          f"prelim={USE_PRELIM_STEP} | Ollama {version} | 每組 N={N_RUNS}\n{'=' * 74}")

    conditions = ["baseline"] * N_RUNS + ["tool_only"] * N_RUNS + ["proposed"] * N_RUNS
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(conditions)
    await warmup()

    funcs = {"baseline": run_baseline_once, "tool_only": run_tool_only_once, "proposed": run_proposed_once}
    results = {k: [] for k in funcs}
    failed = []
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    partial = os.path.join(base_dir, f"ablation_results_multi_v3_{stamp}_partial.json")

    for i, cond in enumerate(conditions):
        print(f"▶ Run {i + 1:03d}/{len(conditions)} [{cond}]", flush=True)
        try:
            r = await funcs[cond](gts)
            r["run_index"] = i + 1
            results[cond].append(r)
            extra = f" | JSON驗證={'通過' if r.get('structured_valid') else '未通過'}" if cond == "proposed" else ""
            print(f"   ✓ {r['elapsed']:.1f}s | 敘事(新)={r['narrative_new']:.0%} | "
                  f"雜訊={r['noise_mention_count']} | prompt tokens={r['max_prompt_tokens']}{extra}")
        except Exception as e:
            failed.append({"run_index": i + 1, "condition": cond, "error": f"{type(e).__name__}: {e}"})
            print(f"   ❌ 失敗已跳過：{type(e).__name__}: {e}", flush=True)
        try:
            with open(partial, "w", encoding="utf-8") as f:
                json.dump({"results_so_far": results, "failed_runs": failed}, f, ensure_ascii=False)
        except Exception:
            pass

    summaries = {k: summarize(v) for k, v in results.items() if v}
    print(f"\n{'=' * 74}\n   結果摘要")
    for k, s in summaries.items():
        print(f"   [{k}] 敘事(新) {s['narrative_new']:.3f}±{s['narrative_new_sd']:.3f} | "
              f"敘事(舊) {s['narrative_old']:.3f} | 雜訊 {s['noise_mention_count']:.2f}±{s['noise_mention_count_sd']:.2f} | "
              f"耗時 {s['elapsed']:.1f}s | 最大 prompt tokens {s['max_prompt_tokens_overall']} | "
              f"因長度截斷 {s['runs_with_length_stop']} 次")
    if "proposed" in summaries:
        p = summaries["proposed"]
        print(f"   [proposed] JSON 驗證通過率 {p['structured_valid_rate']:.1%} | "
              f"需要 coverage 提示的次數 {p['runs_needing_nudge']} | 平均迭代 {p['iterations_used']:.2f}")
    if failed:
        print(f"   ⚠️ 失敗 {len(failed)} 筆：{failed}")
    if any(s["max_prompt_tokens_overall"] >= NUM_CTX * 0.9 for s in summaries.values()):
        print("   ⚠️ 有 prompt 接近 num_ctx 上限，請檢查是否被截斷！")

    output = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "ablation_study_multi_v3.py",
        "model": MODEL_NAME, "ollama_version": version,
        "options": OLLAMA_OPTIONS, "use_prelim_step": USE_PRELIM_STEP,
        "random_seed": RANDOM_SEED, "execution_order": conditions,
        "n_runs_each_requested": N_RUNS,
        "n_runs_each_actual": {k: len(v) for k, v in results.items()},
        "failed_runs": failed,
        "ground_truth_incidents": gts,
        "noise_log": build_noisy_log_multi(gts),
        "summaries": summaries,
        "raw_baseline": results["baseline"],
        "raw_tool_only": results["tool_only"],
        "raw_proposed": results["proposed"],
    }
    path = save_json(output, os.path.join(base_dir, f"ablation_results_multi_v3_{stamp}.json"),
                     f"ablation_results_multi_v3_{stamp}.json")
    print(f"\n📂 結果已存至 {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
