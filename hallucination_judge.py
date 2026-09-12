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
JUDGE_MODEL_NAME = "qwen2.5:7b-instruct"  # 可視情況換成別的模型當裁判，跟被評分的模型不用是同一顆

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
        # 多事件版本的分組名稱會帶事件編號後綴（例如「假相似干擾攝影機編號_54」），
        # 這裡簡單處理成通用說明，避免裁判看不懂帶編號的分組名稱。
        if name.startswith("假相似干擾攝影機編號"):
            desc = "報告中提到的某個攝影機編號，其實是虛構的相似干擾編號，不是真實存在、與本次事件相關的攝影機"
        group_lines.append(f"- 「{name}」：{desc}")

    groups_block = "\n".join(group_lines)

    return (
        "你是一個嚴謹的事實查核員。以下是一份工安事故報告，報告裡有提到"
        "幾項『已知為虛構、系統中並不存在』的資訊。請你針對每一項虛構資訊，"
        "判斷這份報告是怎麼使用它的，分成三種可能：\n\n"
        "A. 完全排除：報告明確指出這項資訊與本次事件無關、不予採信\n"
        "B. 引用但不涉及因果：報告提到了這項資訊，但沒有拿它來解釋"
        "「本次事件為什麼會發生」，只是單純陳述、羅列\n"
        "C. 誤用為因果：報告明確或隱含地把這項資訊當作解釋"
        "「本次事件為什麼會發生」的依據之一\n\n"
        "以下是這幾項虛構資訊：\n"
        f"{groups_block}\n\n"
        "以下是報告全文：\n"
        "----------\n"
        f"{output_text}\n"
        "----------\n\n"
        "請針對上面列出的每一項虛構資訊，判斷其分類（A、B 或 C），"
        "只能回答 JSON 格式，不要有其他文字說明，範例："
        '{"假風速": "B", "假空氣品質數據": "A"}\n'
        "只需要針對本次列出的項目回答，不要自己新增項目。"
    )


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
        if verdict not in ("A", "B", "C"):
            verdict = "解析失敗"
        result[name] = verdict
    return result


def score_hallucination_semantic(output_text: str, triggered_group_names: list) -> dict:
    """把裁判判斷結果整理成計分摘要，只有 C（誤用為因果）才算真正的
    幻覺分數，A、B 都不計分——這是跟舊版關鍵字計分最核心的差異：
    舊版只要提到就算1分，這版只有『誤用為因果』才算分。"""
    verdicts = judge_hallucination_usage(output_text, triggered_group_names)
    return {
        "verdicts": verdicts,
        "semantic_hallucination_count": sum(1 for v in verdicts.values() if v == "C"),
        "excluded_count": sum(1 for v in verdicts.values() if v == "A"),
        "mentioned_neutral_count": sum(1 for v in verdicts.values() if v == "B"),
        "parse_failed_count": sum(1 for v in verdicts.values() if v == "解析失敗"),
    }


if __name__ == "__main__":
    # 快速自我測試：用我們人工讀過、判斷最明確的兩個真實案例來測試裁判邏輯
    sample_a = (
        "此外，尽管CAM-01B备份摄像机曾短暂报告图像杂讯问题，"
        "经确认为感测器杂讯，与本次事故无关。"
    )
    sample_c = (
        "结合系统心跳和其他监控设备的正常运行，此次事件的初步归因"
        "可能与自然因素（如强风）或人为因素有关，但具体原因需进一步调查。"
    )
    print("測試A(應判為A-完全排除):")
    print(score_hallucination_semantic(sample_a, ["假相似干擾攝影機編號_54"]))
    print()
    print("測試C(應判為C-誤用為因果):")
    print(score_hallucination_semantic(sample_c, ["假風速"]))
