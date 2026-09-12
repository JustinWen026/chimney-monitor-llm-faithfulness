"""
ablation_study_v17.py
----------------------
【本版定位：整條校準路徑圖的「階段一」】

在拉大樣本數、換模型、補測事件類型、改進幻覺量測方法之前，我們先確認一件事：
現有的驗證＋迭代重試機制，有沒有機會被真正考驗過？答案是沒有——不管換哪個事件、
拉多高溫度，Proposed 組的迭代輪數始終是 1.00，因為驗證任務（把工具剛回傳的幾個
短欄位值逐字複製）對 LLM 來說太簡單，天生就很難出錯。這一版的目標，就是在
「不扭曲評分標準、不為了讓數字好看而調整方法」的前提下，把測試情境設計得
更嚴苛，讓迭代重試機制真的有機會被觸發、被驗證。

完整校準路徑圖（本版是第①步）：
  ① 修好迭代觸發機制（本版）→ ② 幻覺量測改語意層級判斷 → ③ BERTScore換有
  官方baseline的模型（可與②並行）→ ④ 補測多種事件類型（複用資料庫既有事件，
  不用重錄影片）→ ⑤ 多模型比較 + 拉大樣本數（放最後，因為運算量最大，
  前面步驟沒校準好會白做工）。

本版做的三個改動（同時上，因為目標是「打造夠嚴苛的綜合壓力測試」，
不是要逐一拆解各自貢獻）：
  1. 擴充驗證欄位：從 6 個增加到 8 個（新增 duration_seconds、status），
     欄位越多，全部一字不差抄對的機率越低（機率是乘法效應）。
  2. 拉開推理距離：原本是「工具回傳資料」後緊接著「請輸出JSON」，改成
     中間插入一段「先寫初步分析」的步驟，讓最終輸出JSON的時間點離看到
     原始資料更遠，更接近真實多輪對話情境，也更容易在複製時出現偏移。
  3. 放相似干擾值：雜訊日誌裡新增一個「長得像但不是的攝影機編號」
     （例如真實是 CAM-01，雜訊放 CAM-01B），測試模型會不會搞混。

跟上一版 ablation_study_v14/v16 的核心差異，就是以上三點；三組消融的
基本設計（Baseline / Tool-only / Proposed）、分組計分的幻覺量測邏輯、
評分公平性的修正（拿掉incident_id、繁簡體變體）全部沿用不動。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python ablation_study_v17.py                 # 自動抓資料庫最新一筆事件
    python ablation_study_v17.py 54               # 指定 incident_id=54
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

# 現在是三組（Baseline / Tool-only / Proposed）各跑 N_RUNS 次，共 3×N_RUNS 次。
# N_RUNS=10 時總共 30 次，Tool-only 跟 Proposed 都要起 MCP 子行程，
# 抓 10~20 分鐘跑完。時間緊迫可先調到 5~6 測試流程是否正常，
# 正式數據再調回 10 以上。
N_RUNS = 5
MAX_ITERATIONS = 2

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
# 【v17修正】從6個欄位擴充到8個，新增 duration_seconds、status。
# 欄位越多，模型要一字不差全部抄對的機率越低（單欄位出錯機率固定的話，
# 全對機率是各欄位機率的連乘，欄位數增加會讓全對機率明顯下降），
# 藉此提高迭代重試機制被真正觸發的機會，而不是每次都第一輪就通過。
VERIFICATION_FIELDS = ["incident_id", "camera_id", "event_type", "severity", "risk_level", "timestamp", "duration_seconds", "status"]

# 事件類型對照中文關鍵詞，供 Baseline 組的自由文字輸出比對用
# （Baseline 組沒有強制輸出結構化 JSON，只能用關鍵字比對是否有提到正確類型）
EVENT_TYPE_KEYWORDS = {
    # 【修正】原本只設了繁體「白煙」這一種寫法，完全沒涵蓋模型實際會用的
    # 其他正確表達方式：簡體字（烟/雾）、換句話說的同義詞（白色煙霧）、
    # 以及最常見、最不會產生歧義的做法——直接引用工具回傳的英文原始
    # 欄位值（white_smoky）。原本的清單漏掉這些，導致大量「講對了但用詞
    # 不同」的正確輸出被誤判為「沒講對事件類型」，這才是造成 Tool-only
    # 組分數異常偏低的真正原因，不是機制本身有問題。
    "fire": ["火災", "火灾", "明火", "火焰", "起火", "fire"],
    "black_smoky": ["黑煙", "黑烟", "black_smoky", "black smoke"],
    "white_smoky": ["白煙", "白烟", "白色煙霧", "白色烟雾", "white_smoky", "white smoke"],
    "black_smoky+fire": ["黑煙", "黑烟", "black_smoky", "複合", "复合", "black_smoky+fire"],
}

# 【v12 修正】上一版的幻覺觸發詞（PM2.5、風速…）在 prompt 裡完全沒有出現過，
# 模型沒有任何素材可以誤用，兩組測出來都是 0，這不代表系統可靠，只代表
# 實驗設計沒有給模型犯錯的機會。這版改成：故意在兩組都會看到的原始日誌裡
# 植入這些假值，讓模型有真實的「雜訊」可能被誤複製或誤觸發聯想，
# 幻覺觸發詞就是這些我們自己植入、確定為假的內容——只要報告裡出現，
# 就是百分之百可判定的幻覺，不再是含糊的關鍵字猜測。
FAKE_CAMERA_ID = "CAM-09"
FAKE_WIND_SPEED = "12.4 m/s"
FAKE_SO2 = "SO2 濃度 0.08 ppm"
FAKE_NOX = "NOx 濃度 0.15 ppm"
FAKE_EMAIL = "safety_backup@petrochem.com"
FAKE_TS_1 = "13:25:00"
FAKE_TS_2 = "13:47:02"
FAKE_FRAME_RATE = "29.8 FPS"

# 【修正】原本是一份扁平清單，逐字比對加總——但雜訊日誌裡有些假資訊本來
# 就是「綁在同一句話裡」出現的（例如「SO2濃度0.08 ppm，NOx濃度0.15 ppm」），
# 模型只要把這句話整句抄一遍，一個引用動作就被算成4分，把幻覺數字灌水到
# 看起來比實際「犯錯次數」誇張。改成分組計分：同一組雜訊來源，不管命中
# 裡面幾個字眼，都只算1分，才能反映「引用了幾種不同的雜訊」而不是
# 「引用的雜訊裡剛好包含幾個可比對的字眼」。
HALLUCINATION_TRIGGER_GROUPS = {
    "假攝影機編號": [FAKE_CAMERA_ID],
    "假風速": [FAKE_WIND_SPEED],
    "假空氣品質數據": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
    "假email通知": [FAKE_EMAIL],
    "假時間戳雜訊": [FAKE_TS_1, FAKE_TS_2],
    "假影像處理參數": [FAKE_FRAME_RATE, "contrast"],
}

# 保留舊變數名稱以維持相容性（部分程式碼可能還引用它列出所有字眼，
# 例如錯誤訊息或除錯輸出，這裡攤平成同一份清單）。
HALLUCINATION_TRIGGERS = [kw for group in HALLUCINATION_TRIGGER_GROUPS.values() for kw in group]


def build_noisy_log(gt: dict) -> str:
    """建立一段摻雜真實事件資料與植入假雜訊的原始日誌，模擬真實工安系統
    後台常見的、參雜大量無關資訊的原始 log。兩組（Baseline / Proposed）
    都會看到完全一樣的這段內容，唯一差異是 Proposed 組被要求另外呼叫
    工具取得權威資料、且被告知這段日誌可能含雜訊。

    【v17新增】相似干擾攝影機編號：在真實攝影機編號旁邊，放一個長得很像
    但不是同一支的假編號（例如真實是 CAM-01，雜訊放 CAM-01B），模擬真實
    工安系統裡常見的「編號相近、容易混淆」情境，測試模型會不會把兩者搞混、
    誤把干擾編號當成事發攝影機寫進報告。"""
    decoy_camera_id = f"{gt['camera_id']}B"
    return (
        f"[INFO] {FAKE_TS_1} 系統心跳正常，Zone 巡檢無異常回報。\n"
        f"[DEBUG] {FAKE_CAMERA_ID} 影像對比度(contrast)調整，目前對比值 1.2。\n"
        f"[INFO] {decoy_camera_id}（備援機位，非本次事發攝影機）短暫回報畫面"
        "雜訊，經覆核為感測器雜訊，與本次事件無關。\n"
        f"[WARN] 現場風速感測器讀值為 {FAKE_WIND_SPEED}，偏強陣風，"
        "可能影響煙流擴散方向。\n"
        f"[INFO] 空氣品質監測：{FAKE_SO2}，{FAKE_NOX}，均在正常範圍。\n"
        f"[CRITICAL] {gt['timestamp']} 狀態機觸發告警。攝影機 {gt['camera_id']} "
        f"偵測到 {gt['event_type']} 異常，嚴重度 {gt['severity']}，"
        f"風險等級 {gt['risk_level']}，事件編號 {gt['incident_id']}。\n"
        f"[INFO] {FAKE_TS_2} 系統已自動寄送通知至 {FAKE_EMAIL}。\n"
        f"[DEBUG] 背景相減演算法運行穩定，frame rate {FAKE_FRAME_RATE}。"
    )


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
        # 【v17新增】擴充驗證欄位，duration_seconds 對應資料庫的 duration
        # 欄位（跟 mcp_server_v3.py 的 get_incident_by_id 回傳的鍵名一致），
        # status 直接沿用資料庫欄位。這兩個都是 get_incident_by_id 本來就
        # 會回傳的欄位，不需要額外修改 MCP Server。
        "duration_seconds": row["duration"],
        "status": row["status"],
    }


def fetch_latest_incident_id() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM incidents ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise ValueError("資料庫目前沒有任何事件紀錄，請先讓系統累積至少一筆告警後再執行本實驗。")
    return row[0]


def _get_trigger_groups(gt: dict = None) -> dict:
    """回傳完整的雜訊分組清單。【v17新增】干擾攝影機編號是依 ground truth
    動態產生的（例如真實是CAM-01，干擾值是CAM-01B），沒辦法寫死在模組層級
    的字典裡，所以額外用這個函式在有 gt 時動態補上這一組。"""
    groups = dict(HALLUCINATION_TRIGGER_GROUPS)
    if gt is not None:
        groups["假相似干擾攝影機編號"] = [f"{gt['camera_id']}B"]
    return groups


def score_hallucination(text: str, gt: dict = None) -> int:
    """依分組計分：同一組雜訊來源（例如假空氣品質數據裡的SO2/NOx/兩個ppm值）
    不管命中組內幾個字眼，都只算1分，避免同一次引用動作被重複灌水。
    傳入 gt 時會額外檢查 v17 新增的相似干擾攝影機編號這一組。"""
    text_lower = text.lower()
    return sum(
        1 for keywords in _get_trigger_groups(gt).values()
        if any(kw.lower() in text_lower for kw in keywords)
    )


def get_hallucination_groups_hit(text: str, gt: dict = None) -> list:
    """回傳實際命中的雜訊分組名稱（而非逐字命中清單），方便論文或除錯時
    直接看懂「引用了哪幾種雜訊」，而不是一堆容易誤解為4倍嚴重的字眼列表。"""
    text_lower = text.lower()
    return [
        group_name for group_name, keywords in _get_trigger_groups(gt).items()
        if any(kw.lower() in text_lower for kw in keywords)
    ]


# ── Baseline 組：模擬「沒有工具、沒有驗證」的舊做法 ──────────────────────────

def build_baseline_prompt(gt: dict) -> str:
    """【v12 修正】原本直接給一句乾淨的事實陳述，模型沒有機會誤判。
    改成給一段摻雜真實資料與植入假雜訊的原始日誌，模擬「沒有工具驗證、
    只能自己從雜訊日誌裡挑資訊寫報告」的舊做法真實情境。"""
    noisy_log = build_noisy_log(gt)
    return (
        "你是石化園區的安全專家。以下是後台系統的原始日誌，"
        "請從中判讀出工安事件，生成一份中文事故初報，內容包含事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議：\n"
        + noisy_log
    )


def score_grounding_baseline(text: str, gt: dict) -> dict:
    """Baseline／Tool-only 組沒有強制輸出結構化資料，只能用關鍵字比對輸出裡
    有沒有正確提到這幾個關鍵欄位。

    【v14 修正】原本把「有沒有提到內部資料庫流水號 incident_id」也算進
    這三組的忠實度分數，但這是不合理的評分標準——一份給人看的自由格式
    工安報告，自然寫法本來就不會刻意插入內部系統的 primary key，這不是
    「講錯了」，是「沒有義務講」。Proposed 組能拿到 incident_id 完全正確，
    是因為它被強制要求輸出結構化 JSON、JSON 裡本來就得填這個欄位，這是
    格式要求造成的差異，不是事實忠實度的差異，繼續用同一把尺量三組並不
    公平。這裡把 incident_id 獨立記錄成參考資訊，但不計入主要的忠實度
    分數，主分數只看三項對人類讀者有意義的事實：攝影機、風險等級、
    事件類型。"""
    core_checks = [
        gt["camera_id"] in text,
        str(gt["risk_level"]) in text,
        any(kw in text for kw in EVENT_TYPE_KEYWORDS.get(gt["event_type"], [])),
    ]
    incident_id_mentioned = str(gt["incident_id"]) in text
    return {
        "score": sum(core_checks) / len(core_checks),
        "incident_id_mentioned": incident_id_mentioned,
    }


async def run_baseline_once(gt: dict) -> dict:
    prompt = build_baseline_prompt(gt)
    start = time.time()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.8},
                "stream": False,
            },
            timeout=60.0,
        )
    elapsed = time.time() - start
    text = resp.json().get("message", {}).get("content", "")
    grounding = score_grounding_baseline(text, gt)
    return {
        "condition": "baseline",
        "elapsed": elapsed,
        "grounding_score": grounding["score"],
        "incident_id_mentioned": grounding["incident_id_mentioned"],
        "hallucination_count": score_hallucination(text, gt),
        "hallucination_keywords_hit": get_hallucination_groups_hit(text, gt),
        "iterations_used": 1,
        "output_text": text,
    }


# ── Proposed 組：完整複刻 report_generator_v7 的工具驗證＋迭代邏輯 ──────────

def _build_proposed_prompt(incident_id: int, gt: dict) -> str:
    """【v12 修正】加入跟 Baseline 組完全相同的雜訊日誌，明確告知這段日誌
    可能含雜訊、正式數據以工具查詢結果為準。這樣才是公平且有意義的對照：
    兩組看到的雜訊完全一樣，唯一差異是有沒有強制工具查證＋欄位驗證，
    藉此測試這個機制是否真的能讓模型不被旁邊的雜訊帶著走。"""
    fields_hint = "、".join(VERIFICATION_FIELDS)
    noisy_log = build_noisy_log(gt)
    return (
        f"【工安事件通報】以下是後台系統的原始日誌，可能包含雜訊或無關資訊：\n"
        f"{noisy_log}\n\n"
        f"本次目標事件的 incident_id 為 {incident_id}。"
        "請你【必須】立刻調用 `get_incident_by_id` 工具，"
        f"並傳入參數 incident_id={incident_id}，取得該筆事件的權威結構化資料"
        "（若與上方日誌內容有出入，一律以工具查詢結果為準，不可採信日誌裡"
        "其他無關的雜訊內容）。取得資料後：\n"
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

            messages = [{"role": "user", "content": _build_proposed_prompt(incident_id, gt)}]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools,
                        "options": {"temperature": 0.8},
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

                # 【v17新增：拉開推理距離】原本拿到工具結果後緊接著就要求輸出
                # 最終報告+JSON，資料跟輸出幾乎是「複製貼上」等級的接近，模型
                # 幾乎不會出錯。這裡插入一輪「先寫初步分析」的中間步驟，讓最終
                # 需要精確複製欄位值的時間點，離看到原始資料的位置更遠，更接近
                # 真實多輪對話情境，也更容易在複製過程中出現偏移、觸發驗證失敗
                # 與後續的迭代重試。
                if tool_was_called:
                    prelim_prompt = (
                        "在正式生成報告前，請先用不超過100字，寫一段初步情境判斷"
                        "（純文字敘述，不需要包含JSON），簡述你從工具查詢結果看到了什麼。"
                    )
                    messages.append({"role": "user", "content": prelim_prompt})
                    prelim_response = await client.post(
                        OLLAMA_URL_CHAT,
                        json={
                            "model": MODEL_NAME,
                            "messages": messages,
                            "options": {"temperature": 0.8},
                            "stream": False,
                        },
                        timeout=60.0,
                    )
                    prelim_text = prelim_response.json().get("message", {}).get("content", "")
                    messages.append({"role": "assistant", "content": prelim_text})
                    fields_hint_prelim = "、".join(VERIFICATION_FIELDS)
                    messages.append({
                        "role": "user",
                        "content": (
                            "很好，現在請根據上述初步判斷與稍早工具查詢到的資料，"
                            "生成完整的正式事故初報（事件摘要、歸因判定、風險等級、"
                            "建議處置、追蹤建議），並在報告最後附上一個 ```json 區塊```，"
                            f"格式為包含以下欄位的 JSON 物件：{fields_hint_prelim}。"
                            "JSON裡的每個值都必須跟工具查詢結果完全一致、逐字複製，"
                            "不可以自行改寫格式、翻譯、四捨五入或憑印象填寫。"
                        ),
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
                                "options": {"temperature": 0.8},
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
        "hallucination_count": score_hallucination(narrative_text, gt),
        "hallucination_keywords_hit": get_hallucination_groups_hit(narrative_text, gt),
        "iterations_used": len(iteration_valid_flags),
        "tool_was_called": tool_was_called,
        "output_text": narrative_text,
    }


def _build_tool_only_prompt(incident_id: int, gt: dict) -> str:
    """【v12 三組消融新增】中間組：一樣看得到雜訊日誌、一樣被要求呼叫工具
    取得權威資料，但不要求輸出驗證用 JSON、不做欄位比對、不重試。
    用來拆解「單純擁有工具查證的資料」跟「工具+強制驗證+迭代重試」
    這兩件事，各自對忠實度與幻覺率的貢獻有多少——這才是消融實驗該回答
    的問題：不是『有沒有全部機制』，是『拿掉驗證+重試這一層，還剩多少效果』。"""
    noisy_log = build_noisy_log(gt)
    return (
        f"【工安事件通報】以下是後台系統的原始日誌，可能包含雜訊或無關資訊：\n"
        f"{noisy_log}\n\n"
        f"本次目標事件的 incident_id 為 {incident_id}。"
        "請你呼叫 `get_incident_by_id` 工具取得該筆事件的權威結構化資料"
        "（若與上方日誌內容有出入，一律以工具查詢結果為準，不可採信日誌裡"
        "其他無關的雜訊內容），並生成一份中文事故初報，內容包含事件摘要、"
        "歸因判定、風險等級、建議處置、追蹤建議。"
    )


async def run_tool_only_once(gt: dict) -> dict:
    """中間組：呼叫一次工具、不驗證、不重試，單次生成即結束。

    【v17說明】這一組刻意不套用「擴充驗證欄位」與「拉開推理距離」這兩項
    v17新增的難度調整：擴充驗證欄位只影響 Proposed 組要求輸出的 JSON
    比對邏輯，Tool-only 本來就沒有結構化驗證，加了也不會有作用；拉開推理
    距離是為了增加「複製工具資料到最終JSON」這個任務出錯的機會，但
    Tool-only 從頭到尾不會重試，套用這個步驟不會改變它的行為，只會白白
    增加一次無意義的LLM呼叫。這組唯一會受到 v17 影響的地方，是三組共用的
    雜訊日誌裡新增的相似干擾攝影機編號。"""
    incident_id = gt["incident_id"]
    start = time.time()
    tool_was_called = False
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
            messages = [{"role": "user", "content": _build_tool_only_prompt(incident_id, gt)}]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools,
                        "options": {"temperature": 0.8},
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

                # 不驗證、不重試：不管有沒有呼叫工具，都只生成這一次就結束。
                final_response = await client.post(
                    OLLAMA_URL_CHAT,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "options": {"temperature": 0.8},
                        "stream": False,
                    },
                    timeout=60.0,
                )
                full_text = final_response.json().get("message", {}).get("content", "")

    elapsed = time.time() - start
    grounding = score_grounding_baseline(full_text, gt)
    return {
        "condition": "tool_only",
        "elapsed": elapsed,
        # 這組沒有強制輸出結構化 JSON，用跟 Baseline 相同的關鍵字比對法評分，
        # 確保跟 Baseline 用同一把尺，才能公平看出「有工具」本身的貢獻。
        "grounding_score": grounding["score"],
        "incident_id_mentioned": grounding["incident_id_mentioned"],
        "hallucination_count": score_hallucination(full_text, gt),
        "hallucination_keywords_hit": get_hallucination_groups_hit(full_text, gt),
        "iterations_used": 1,
        "tool_was_called": tool_was_called,
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
    """【v11修正】舊版 ablation_study_v6.py 有這個步驟，這版重寫時漏掉了，
    導致第一筆呼叫（不論抽到哪一組）要承擔 Ollama 把模型權重載入 GPU
    顯示記憶體的冷啟動時間，把那一組的平均耗時不公平地拉高。
    正式開始計時前，先用一次無關緊要的呼叫把模型載入 VRAM。"""
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
        incident_id = int(sys.argv[1])
    else:
        incident_id = fetch_latest_incident_id()
        print(f"未指定 incident_id，自動使用資料庫最新一筆事件：#{incident_id}")

    gt = fetch_ground_truth(incident_id)

    print(f"\n{'=' * 70}")
    print(f"   消融實驗 v17 — 階段一：擴充驗證欄位＋拉開推理距離＋相似干擾值，測試迭代重試能否被真正觸發")
    print(f"   Ground Truth 事件：#{gt['incident_id']}  {gt['camera_id']}  "
          f"{gt['event_type']}  風險等級{gt['risk_level']}")
    print(f"   模型: {MODEL_NAME}  |  每組 N={N_RUNS}，隨機順序，共 {N_RUNS * 3} 次")
    print(f"{'=' * 70}")

    # 【v12 三組消融】原本只比較「全有」vs「全無」，只能回答「整套機制有沒有用」，
    # 回答不了「到底是工具本身有用，還是驗證+重試這個機制本身有用」。
    # 加入 tool_only 中間組，才是名符其實的『消融』：逐步拿掉元件，看各自貢獻。
    conditions = (["baseline"] * N_RUNS) + (["tool_only"] * N_RUNS) + (["proposed"] * N_RUNS)
    import random
    random.shuffle(conditions)

    await warmup()

    condition_labels = {
        "baseline": "Baseline（無工具/無驗證）",
        "tool_only": "Tool-only（有工具/無驗證重試）",
        "proposed": "Proposed（工具+驗證+迭代）",
    }
    condition_funcs = {
        "baseline": run_baseline_once,
        "tool_only": run_tool_only_once,
        "proposed": run_proposed_once,
    }

    results_by_condition = {"baseline": [], "tool_only": [], "proposed": []}
    for i, condition in enumerate(conditions):
        print(f"▶ Run {i + 1:02d}/{len(conditions)} [{condition_labels[condition]}]", flush=True)
        r = await condition_funcs[condition](gt)
        results_by_condition[condition].append(r)
        print(
            f"   ✓ 時間={r['elapsed']:.2f}s | 忠實度={r['grounding_score']:.0%} | "
            f"幻覺數={r['hallucination_count']} | 迭代輪數={r['iterations_used']}"
        )

    b = summarize(results_by_condition["baseline"])
    t = summarize(results_by_condition["tool_only"])
    p = summarize(results_by_condition["proposed"])

    print(f"\n\n{'=' * 96}")
    print(f"          消融實驗結果對比（N={N_RUNS} each，共 {N_RUNS * 3} 次）")
    print(f"          Ground Truth 事件：#{gt['incident_id']}（{gt['event_type']}）")
    print(f"{'=' * 96}")
    print(f"{'指標':<18} {'Baseline':<20} {'Tool-only':<20} {'Proposed':<20}")
    print(f"{'':<18} {'(無工具/無驗證)':<20} {'(有工具/無驗證重試)':<20} {'(工具+驗證+迭代)':<20}")
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

    # 【v14新增】incident_id 是否被提及僅供參考，不計入主要忠實度分數
    # （理由見 score_grounding_baseline 的說明）。
    b_id_rate = sum(1 for r in results_by_condition["baseline"] if r.get("incident_id_mentioned")) / len(results_by_condition["baseline"])
    t_id_rate = sum(1 for r in results_by_condition["tool_only"] if r.get("incident_id_mentioned")) / len(results_by_condition["tool_only"])
    print(f"（參考資訊，未計入忠實度分數）內部事件編號被主動提及比例："
          f"Baseline {b_id_rate:.0%}　Tool-only {t_id_rate:.0%}　"
          "（Proposed 因強制輸出JSON，恆為100%，故不比較意義不大）\n")

    print("💡 三組看到完全相同、摻雜真實資料與植入假雜訊的原始日誌。")
    print("   Tool-only 組拆解出「單純擁有工具查證資料」的貢獻；")
    print("   Proposed 組在此之上再加上強制驗證＋迭代重試，兩者之差")
    print("   才是『驗證+重試機制本身』的貢獻，而不是工具本身的貢獻。")
    print("   幻覺數計算的是「植入的假雜訊」出現在報告裡的次數，這些值是")
    print("   我們自己造的，只要出現就可判定為雜訊污染，但無法區分模型是")
    print("   『誤把雜訊當事實』還是『正確提及並排除雜訊』——兩者都會被計入，")
    print("   這是本量測法的已知限制，請務必人工抽查 output_text 欄位確認。\n")
    print("⚠️  方法論範圍聲明：本測試為研究者自行設計之受控雜訊情境，用於")
    print("   診斷『工具驗證＋迭代閉環』機制的相對效果，樣本為單一模型")
    print("   （qwen2.5:7b-instruct）、單一事件類型、N=" + str(N_RUNS) + " 之小樣本結果，")
    print("   不宣稱涵蓋真實世界所有幻覺樣態或可推論至其他模型/情境。\n")

    output = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "n_runs_each": N_RUNS,
        "ground_truth_incident": gt,
        "methodology_scope_note": (
            "本測試為研究者自行設計之受控雜訊情境（自行植入已知為假的雜訊內容，"
            "非取自真實事故資料），用於診斷工具驗證與迭代重試機制的相對效果，"
            "樣本為單一模型（qwen2.5:7b-instruct）、單一 ground truth 事件、"
            f"每組 N={N_RUNS} 之小樣本結果。幻覺計數以關鍵字比對實作，"
            "無法區分『誤採信雜訊為事實』與『正確提及並排除雜訊』兩種不同行為，"
            "建議人工複核 output_text 欄位以確認自動化評分之有效性。"
            "本結果不宣稱涵蓋真實世界所有幻覺樣態，亦不宣稱可推論至其他模型或情境。"
        ),
        "baseline_summary": b,
        "tool_only_summary": t,
        "proposed_summary": p,
        "raw_baseline": results_by_condition["baseline"],
        "raw_tool_only": results_by_condition["tool_only"],
        "raw_proposed": results_by_condition["proposed"],
    }
    out_path = r"D:\NCKU_paper_Capstone_Project\reports\ablation_results_v17.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {out_path}\n")
    except Exception:
        # 若不在 Windows 專案路徑下執行（例如先在其他環境測試語法），
        # 退回存在目前工作目錄，不讓實驗結果因路徑問題而遺失。
        fallback_path = "ablation_results_v17.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"📂 詳細數據已存至 {fallback_path}（預設路徑不可寫入，已退回目前目錄）\n")


if __name__ == "__main__":
    asyncio.run(main())
