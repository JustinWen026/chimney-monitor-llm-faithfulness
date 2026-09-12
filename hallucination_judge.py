"""
hallucination_judge.py
------------------------
用另一個LLM當裁判，把「有沒有提到雜訊」升級成「這段話怎麼使用這個雜訊」。

背景：原本的關鍵字比對法只看「有沒有出現」，沒辦法分辨下面三種完全不同、
嚴重程度也不同的行為：
    A. 完全排除：明確講這個雜訊跟事件無關
    B. 引用但不涉及因果：提到這個(捏造的)數據，但沒有拿來解釋事件為什麼發生
    C. 誤用為因果：明確拿來當作解釋事件發生原因的依據，這是最嚴重的幻覺

三分類的判斷是研究者本人逐筆人工讀過15筆真實樣本後歸納出來的分類架構
（不是我事先假設好套上去的），對應到的具體判斷依據，可參考本專案討論記錄
裡的人工標註範例。

使用方式：
    from hallucination_judge import judge_hallucination_usage
    result = judge_hallucination_usage(output_text, triggered_groups)
    # result 範例：
    # {"假風速": "B", "假空氣品質數據": "A", "假email通知": "C"}

【重要】這支裁判函式本身的準確度，在正式取代舊版關鍵字計分之前，
必須先經過人工抽查驗證，不能寫完就直接信——這是我們討論過的第三步驟，
還沒做完之前，不要把這支函式接進正式的消融實驗腳本。
"""

import json
import re

import httpx

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
JUDGE_MODEL_NAME = "llama3.1:8b-instruct"  # 【第六輪】換成不同家族的模型當裁判，驗證是不是qwen2.5本身的能力限制

# 雜訊分組的白話說明，讓裁判LLM看得懂每個假雜訊具體是什麼，
# 不用自己去猜「假風速」這個標籤代表哪一段文字。
GROUP_DESCRIPTIONS = {
    "假風速": "現場風速感測器讀值為 12.4 m/s（這個讀數是虛構的，系統實際上沒有風速感測器）",
    "假空氣品質數據": "SO2 濃度 0.08 ppm、NOx 濃度 0.15 ppm（這些數值是虛構的，系統實際上沒有空氣品質感測器）",
    "假email通知": "系統已自動寄送通知至 safety_backup@petrochem.com（這個通知行為是虛構的，系統實際上沒有這個通知機制）",
    "假時間戳雜訊": "日誌裡出現的 13:25:00 或 13:47:02 這類時間戳（這些是無關雜訊時間點，不是本次事件的真實時間）",
    "假影像處理參數": "影像對比度調整、frame rate 29.8 FPS 這類技術參數（這些是無關的系統除錯資訊，不是事件本身的特徵）",
}


def _build_judge_prompt(output_text: str, triggered_group_names: list) -> str:
    group_lines = []
    for name in triggered_group_names:
        desc = GROUP_DESCRIPTIONS.get(name, name)
        if name.startswith("假相似干擾攝影機編號"):
            desc = "報告中提到的某個攝影機編號，其實是虛構的相似干擾編號，不是真實存在、與本次事件相關的攝影機"
        group_lines.append(f"- 「{name}」：{desc}")

    groups_block = "\n".join(group_lines)

    return (
        "你是一個嚴謹的事實查核員。以下是一份工安事故報告，報告裡有提到"
        "幾項『已知為虛構、系統中並不存在』的資訊。請你針對每一項虛構資訊，"
        "判斷這份報告有沒有把它當作解釋『本次事件為什麼會發生』的原因之一，"
        "只有兩種可能：\n\n"
        "有錯：這段話的功能是在回答「本次事件為什麼會發生」，並把這項"
        "虛構資訊列為造成事件的原因之一（不管有沒有明確講「無關」，只要"
        "同時也把它拿來當作事件發生的原因，就算有錯）\n"
        "沒錯：報告要嘛明確講這項資訊與事件無關，要嘛只是單純陳述、羅列"
        "這項資訊，完全沒有拿它來解釋事件為什麼發生\n\n"
        "【重要判斷準則，請務必遵守】\n"
        "判斷的依據是「這句話的功能是不是在解釋事件發生的原因」，"
        "不是「這句話的語氣有多篤定」。即使句子裡用了『可能』、『初步判斷』、"
        "『初步歸因』、『需進一步調查』這類保留、謹慎的語氣，只要整句話的"
        "目的是要說明「這件事可能是造成本次事件的原因」，就必須判為「有錯」，"
        "不可以因為語氣保守、不夠篤定，就自動判為「沒錯」。\n\n"
        "以下用例句示範界線：\n"
        "範例（沒錯，明確排除）：「經確認該攝影機讀數為感測器雜訊，"
        "與本次事件無關」。\n"
        "範例（沒錯，單純陳述不涉及因果）：「現場風速為12.4 m/s，可能影響"
        "煙流擴散方向」——只描述物理效應，完全沒有連結到「本次異常事件為什麼"
        "會發生」。\n"
        "範例（有錯，注意語氣保守不影響判斷）：「初步歸因可能與自然因素"
        "（如強風）有關，具體原因需進一步調查」——雖然語氣保守，但功能明確"
        "是在解釋「本次事件為什麼會發生」，因此判為「有錯」。\n\n"
        "以下是這幾項虛構資訊：\n"
        f"{groups_block}\n\n"
        "以下是報告全文：\n"
        "----------\n"
        f"{output_text}\n"
        "----------\n\n"
        "請針對上面列出的每一項虛構資訊，判斷是「有錯」還是「沒錯」，"
        "只能回答 JSON 格式，不要有其他文字說明，範例："
        '{"假風速": "有錯", "假空氣品質數據": "沒錯"}\n'
        "只需要針對本次列出的項目回答，不要自己新增項目。"
    )


CAUSE_EXTRACTION_PROMPT_TEMPLATE = (
    "以下是一份工安事故報告。請找出報告裡『用來解釋本次事件為什麼會發生』"
    "的那一段文字，原封不動地摘錄出來（不要改寫、不要摘要、不要翻譯，"
    "逐字複製原文）。如果報告裡完全沒有任何解釋原因的句子，只回答「無」。\n\n"
    "報告全文：\n----------\n{text}\n----------\n\n"
    "請直接輸出摘錄的原文，或輸出「無」，不要有任何其他說明文字。"
)

# 各雜訊分組對應的關鍵字，用來在「抽取出來的因果解釋段落」裡做機械比對。
# 這份清單需要跟消融實驗腳本裡的 HALLUCINATION_TRIGGER_GROUPS 保持一致。
GROUP_KEYWORDS = {
    "假風速": ["12.4 m/s", "風速", "风速"],
    "假空氣品質數據": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
    "假email通知": ["safety_backup@petrochem.com"],
    "假時間戳雜訊": ["13:25:00", "13:47:02"],
    "假影像處理參數": ["29.8 FPS", "contrast", "對比度", "frame rate"],
}


def extract_cause_explanation(output_text: str) -> str:
    """【第五輪：抽取式判斷法】只要求LLM做一件具體的事——把報告裡用來
    解釋事件發生原因的那段文字，原封不動摘錄出來。這是文字擷取任務，
    比要求LLM自己判斷「語氣保守程度」「算不算誤用」這種抽象後設判斷
    具體、有邊界得多。"""
    prompt = CAUSE_EXTRACTION_PROMPT_TEMPLATE.format(text=output_text)
    with httpx.Client() as client:
        response = client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": JUDGE_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0},
                "stream": False,
            },
            timeout=60.0,
        )
    return response.json().get("message", {}).get("content", "").strip()


def judge_hallucination_usage_extractive(output_text: str, triggered_group_names: list) -> dict:
    """【第五輪：抽取式判斷法】先用LLM抽取出「因果解釋段落」，再用機械式
    關鍵字比對（不靠LLM判斷）檢查每個雜訊分組的關鍵字有沒有出現在這段
    抽取出來的文字裡。判斷責任從「LLM自己下結論」轉移到「已知可靠的
    關鍵字比對」，LLM只負責找出正確的段落範圍。

    對於動態命名的分組（例如「假相似干擾攝影機編號_54」這種帶事件編號
    後綴的分組），因為沒有預先定義關鍵字，回傳「需人工複核」，不猜測。"""
    cause_text = extract_cause_explanation(output_text)
    result = {}
    no_cause_found = cause_text.strip() in ("無", "无", "", "沒有", "没有")

    for name in triggered_group_names:
        if no_cause_found:
            result[name] = "沒錯"
            continue

        if name.startswith("假相似干擾攝影機編號"):
            result[name] = "需人工複核"
            continue

        keywords = GROUP_KEYWORDS.get(name)
        if keywords is None:
            result[name] = "需人工複核"
            continue

        hit = any(kw.lower() in cause_text.lower() for kw in keywords)
        result[name] = "有錯" if hit else "沒錯"

    return result


def judge_hallucination_usage(output_text: str, triggered_group_names: list) -> dict:
    """回傳 {分組名稱: "A"/"B"/"C"} 的判斷結果。若解析失敗，該項回傳
    "解析失敗"，呼叫端應視為需要人工複核，不要直接當作某個確定分類計分。"""
    if not triggered_group_names:
        return {}

    prompt = _build_judge_prompt(output_text, triggered_group_names)

    with httpx.Client() as client:
        response = client.post(
            OLLAMA_URL_CHAT,
            json={
                "model": JUDGE_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0},  # 裁判任務要求穩定、可重現，用溫度0
                "stream": False,
            },
            timeout=60.0,
        )
    raw_text = response.json().get("message", {}).get("content", "")

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {name: "解析失敗" for name in triggered_group_names}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {name: "解析失敗" for name in triggered_group_names}

    result = {}
    for name in triggered_group_names:
        verdict = parsed.get(name, "解析失敗")
        if verdict not in ("有錯", "沒錯"):
            verdict = "解析失敗"
        result[name] = verdict
    return result


def score_hallucination_semantic(output_text: str, triggered_group_names: list) -> dict:
    """把裁判判斷結果整理成計分摘要。【第三輪：直接問「有沒有誤用因果」】
    把原本三分類的A(排除)、B(單純提及)合併成「沒錯」，只有C(誤用因果)
    算「有錯」。這次問的問題，跟前兩輪三分類裡的C類判斷是同一個問題，
    只是重新調整了分類邊界，用意是測試「換個角度問同一個問題」會不會
    有幫助——但要注意這15筆樣本裡「有錯」只佔3筆、「沒錯」佔12筆，
    整體一致率容易被多數類別撐高，解讀時務必分開看「有錯」類別本身的
    準確率，不能只看籠統的整體數字。"""
    verdicts = judge_hallucination_usage(output_text, triggered_group_names)
    return {
        "verdicts": verdicts,
        "semantic_hallucination_count": sum(1 for v in verdicts.values() if v == "有錯"),
        "excluded_count": sum(1 for v in verdicts.values() if v == "沒錯"),
        "parse_failed_count": sum(1 for v in verdicts.values() if v == "解析失敗"),
    }


if __name__ == "__main__":
    # 快速自我測試：用我們人工讀過、判斷最明確的兩個真實案例來測試裁判邏輯
    sample_excluded = (
        "此外，尽管CAM-01B备份摄像机曾短暂报告图像杂讯问题，"
        "经确认为感测器杂讯，与本次事故无关。"
    )
    sample_not_excluded = (
        "结合系统心跳和其他监控设备的正常运行，此次事件的初步归因"
        "可能与自然因素（如强风）或人为因素有关，但具体原因需进一步调查。"
    )
    print("測試1(應判為「沒錯」):")
    print(score_hallucination_semantic(sample_excluded, ["假相似干擾攝影機編號_54"]))
    print()
    print("測試2(應判為「有錯」):")
    print(score_hallucination_semantic(sample_not_excluded, ["假風速"]))