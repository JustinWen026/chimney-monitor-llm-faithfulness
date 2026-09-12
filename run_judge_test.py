"""
run_judge_test.py
-------------------
拿我們已經人工讀過、判斷過的15筆真實樣本，去測試 hallucination_judge.py
的裁判LLM，判斷得準不準。這是我們討論的「三步驟」裡的步驟3。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python run_judge_test.py

執行完會印出：
    1. 每一筆的「人工判斷」vs「裁判LLM判斷」逐一對照
    2. 整體一致率
    3. 分開統計「有錯」「沒錯」兩類的一致率（第三輪嘗試：把原本三分類的
       A排除、B單純提及合併成「沒錯」，只有C誤用因果算「有錯」，這次問
       的問題其實跟前兩輪三分類裡的C類判斷完全相同，只是重新調整分類
       邊界。這15筆裡「有錯」只佔3筆、「沒錯」佔12筆，整體一致率容易被
       多數類別撐高到80%左右卻毫無意義，解讀時務必只看「有錯」類別本身
       的準確率，不能被籠統的整體數字誤導）
"""

import json

from hallucination_judge import judge_hallucination_usage_extractive

# ── 15筆真實樣本 + 對應要判斷的雜訊分組 + 我人工讀過的判斷（基準值） ──────
# 樣本文字節錄自 ablation_results_v17.json（ground truth事件 #54, black_smoky+fire）。
# 人工判斷（human_label）是研究者跟Claude逐句討論、依上下文語意判斷出來的，
# 作為檢驗裁判LLM準不準的基準。

SAMPLES = [
    {
        "id": "baseline_run0_風速",
        "group": "假風速",
        "text": (
            "根據攝影機CAM-01的檢測結果，初步判斷此次事件為一場火災。"
            "現場風速感測器的讀值為12.4 m/s，為偏強陣風，可能影響煙流的擴散方向，"
            "但不直接導致火災的發生。此次事件歸因於現場火災。"
        ),
        "human_label": "沒錯",  # 明確講「不直接導致火災」
    },
    {
        "id": "baseline_run1_風速",
        "group": "假風速",
        "text": (
            "此次事件被归因为烟雾和火焰的检测，可能是由于设备故障、燃料泄漏或"
            "其它潜在危险因素导致。从日志信息来看，当前的环境因素如风速（12.4 m/s）"
            "可能会影响烟雾的扩散方向，但并未直接关联到此次事件的发生原因。"
        ),
        "human_label": "沒錯",
    },
    {
        "id": "baseline_run1_email",
        "group": "假email通知",
        "text": (
            "该事件触发了系统告警，并已自动发送至safety_backup@petrochem.com。"
        ),
        "human_label": "沒錯",  # 陳述為既成事實，但沒有拿來解釋事件發生原因
    },
    {
        "id": "baseline_run2_email",
        "group": "假email通知",
        "text": (
            "攝影機CAM-01檢測到疑似煙火異常，觸發了狀態機的告警，嚴重度為CRITICAL，"
            "風險等級為5，編號為54。系統已自動寄送通知至safety_backup@petrochem.com。"
        ),
        "human_label": "沒錯",
    },
    {
        "id": "baseline_run4_風速",
        "group": "假風速",
        "text": (
            "根据日志信息，摄像头CAM-01检测到的\"black_smoky+fire\"异常，"
            "表明现场可能发生了火灾或烟雾事件。日志中提到的风速为12.4 m/s，"
            "这可能会影响烟雾的扩散方向，但并未直接指出火源位置或火势具体情况。"
        ),
        "human_label": "沒錯",
    },
    {
        "id": "toolonly_run0_風速",
        "group": "假風速",
        "text": (
            "初步分析认为，现场风速为12.4 m/s，偏强的阵风可能影响烟流的扩散方向。"
            "此外，现场感测器检测到的SO2浓度为0.08 ppm，NOx浓度为0.15 ppm，"
            "均在正常范围内，未发现异常。因此，初步推测可能是由于设备老化、"
            "维护不当或外界因素（如人为因素或自然因素）导致的突发火情。"
        ),
        "human_label": "有錯",  # 「自然因素」隱性涵蓋前述風速，且用「因此」直接銜接因果推論
    },
    {
        "id": "toolonly_run0_空氣品質",
        "group": "假空氣品質數據",
        "text": (
            "现场感测器检测到的SO2浓度为0.08 ppm，NOx浓度为0.15 ppm，"
            "均在正常范围内，未发现异常。"
        ),
        "human_label": "沒錯",
    },
    {
        "id": "toolonly_run1_全部並列",
        "group": "假風速",
        "text": (
            "根据系统日志，现场风速传感器读数为12.4 m/s，可能存在强阵风影响烟流"
            "扩散方向。空气品质监测中，SO2浓度为0.08 ppm，NOx浓度为0.15 ppm，"
            "均在正常范围内。摄像机CAM-01在事件发生时的影像对比度调整至1.2。"
            "此外，系统已自动发送通知至safety_backup@petrochem.com邮箱。"
        ),
        "human_label": "沒錯",  # 整段像資訊羅列，標題是歸因判定但沒有明確因果句
    },
    {
        "id": "toolonly_run2_風速",
        "group": "假風速",
        "text": (
            "根据后台系统日志，现场风速传感器读数为12.4 m/s，可能存在强阵风，"
            "这可能会影响烟流的扩散方向。此外，空气品质监测数据显示SO2浓度为"
            "0.08 ppm，NOx浓度为0.15 ppm，均在正常范围内，暂时未发现环境污染问题。"
            "初步判定本次事件可能由于设备故障或环境因素导致，具体情况需进一步调查核实。"
        ),
        "human_label": "有錯",  # 「環境因素」隱性指涉前面的風速
    },
    {
        "id": "toolonly_run3_風速",
        "group": "假風速",
        "text": (
            "根据后台系统日志中的记录，现场风速为12.4 m/s，这可能会影响烟雾的"
            "扩散方向。此外，虽然空气品质监测显示SO2、NOx浓度均在正常范围内，"
            "但烟雾和火情的检测仍需引起高度重视。结合系统心跳和其他监控设备的"
            "正常运行，此次事件的初步归因可能与自然因素（如强风）或人为因素有关，"
            "但具体原因需进一步调查。"
        ),
        "human_label": "有錯",  # 最明確的一筆：直接寫「歸因可能與強風有關」
    },
    {
        "id": "toolonly_run4_干擾攝影機",
        "group": "假相似干擾攝影機編號",
        "text": (
            "此外，尽管CAM-01B备份摄像机曾短暂报告图像杂讯问题，"
            "经确认为感测器杂讯，与本次事故无关。"
        ),
        "human_label": "沒錯",  # 全部樣本裡排除得最乾淨的一筆
    },
    {
        "id": "toolonly_run4_空氣品質",
        "group": "假空氣品質數據",
        "text": (
            "空气品质监控显示SO2浓度为0.08 ppm，NOx浓度为0.15 ppm，均处于正常范围，"
            "因此空气品质对本次事件的影响较小。"
        ),
        "human_label": "沒錯",
    },
    {
        "id": "proposed_run0_風速",
        "group": "假風速",
        "text": (
            "現場風速讀值為12.4 m/s，偏強陣風可能影響煙流擴散方向。"
            "系統未發現其他異常報告。"
        ),
        "human_label": "沒錯",  # 只陳述效應，沒有接因果推論句
    },
    {
        "id": "proposed_run2_風速",
        "group": "假風速",
        "text": (
            "根據現場風速感測器讀值，風速偏強，可能對煙流擴散方向造成影響。"
            "攝影機CAM-01偵測到黑煙與火災異常，初步判定可能存在火災事件。"
        ),
        "human_label": "沒錯",  # 因果句「初步判定可能存在火災事件」是基於攝影機偵測，非風速
    },
    {
        "id": "proposed_run4_email",
        "group": "假email通知",
        "text": (
            "系統已自動寄送通知至safety_backup@petrochem.com。"
        ),
        "human_label": "沒錯",
    },
]


def main():
    print("=" * 70)
    print("裁判LLM 一致率測試（第五輪：抽取式判斷法，N=15）")
    print("=" * 70)

    agree_count = 0
    confusion = {}  # (human, judge) -> count
    judged_results = []
    excluded_results = []  # 「需人工複核」的樣本，不計入一致率，避免不公平地被判成錯誤

    for i, sample in enumerate(SAMPLES):
        result = judge_hallucination_usage_extractive(sample["text"], [sample["group"]])
        judge_label = result.get(sample["group"], "解析失敗")
        human_label = sample["human_label"]

        if judge_label == "需人工複核":
            excluded_results.append({**sample, "judge_label": judge_label})
            print(f"⚪ [{i+1:02d}] {sample['id']:<28} 人工={human_label}  裁判=需人工複核（不計入一致率）")
            continue

        is_agree = (judge_label == human_label)
        agree_count += int(is_agree)
        judged_results.append({**sample, "judge_label": judge_label, "is_agree": is_agree})

        key = (human_label, judge_label)
        confusion[key] = confusion.get(key, 0) + 1

        mark = "✅" if is_agree else "❌"
        print(f"{mark} [{i+1:02d}] {sample['id']:<28} 人工={human_label}  裁判={judge_label}")

    total_scored = len(judged_results)
    print("\n" + "=" * 70)
    print(f"整體一致率（排除需人工複核的{len(excluded_results)}筆）："
          f"{agree_count}/{total_scored} = {agree_count/total_scored:.1%}" if total_scored else "無可統計樣本")
    print("=" * 70)

    print("\n混淆矩陣（人工判斷, 裁判判斷）：次數")
    for key in sorted(confusion.keys()):
        print(f"  人工={key[0]} → 裁判={key[1]}：{confusion[key]} 次")

    print("\n分類別一致率：")
    for label in ["有錯", "沒錯"]:
        subset = [r for r in judged_results if r["human_label"] == label]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["is_agree"])
        print(f"  人工判斷為{label}的樣本（共{len(subset)}筆）：裁判判對 {correct}/{len(subset)} = {correct/len(subset):.1%}")


if __name__ == "__main__":
    main()