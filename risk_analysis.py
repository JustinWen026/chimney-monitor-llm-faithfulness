"""
risk_analysis_v3.py
--------------------
在 v2 的反光風險與時段分析基礎上，新增「點8：結果歸因性關聯分析」：

1. 風險等級 (severity) 與處置效率 (MTTR) 的關聯
   → 驗證「等級判斷是否真的反映了緊急程度」：CRITICAL 事件的平均處置時間
     是否確實比 WARNING 事件短（如果沒有，代表分級可能沒有實際指導現場優先順序）。

2. 煙霧濃度等級 (density_level) 與是否升級為高風險事件的關聯
   → 檢驗「濃度評分」這個自訂指標是否真的跟最終判定的風險等級一致，
     而不是各自獨立、彼此無關的兩套數字。

3. 攝影機來源 (camera_id) 與位置確認事件 (occlusion_flag) 比例的關聯
   → 找出哪一路攝影機的視角死角問題最嚴重。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python risk_analysis_v3.py
"""

import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

DB_PATH = r"D:\NCKU_paper_Capstone_Project\safety_incidents.db"
OUTPUT_CSV = r"D:\NCKU_paper_Capstone_Project\reports\misjudgment_analysis.csv"
OUTPUT_SUMMARY = r"D:\NCKU_paper_Capstone_Project\reports\misjudgment_summary.txt"
OUTPUT_CORRELATION_CSV = r"D:\NCKU_paper_Capstone_Project\reports\correlation_analysis.csv"
GLARE_BRIGHTNESS_THRESHOLD = 240

SEVERITY_ORDER = ["WARNING", "HIGH", "CRITICAL"]


def compute_glare_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(gray >= GLARE_BRIGHTNESS_THRESHOLD)), 4)


def load_incidents():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM incidents ORDER BY id ASC", conn)
    return df


def analyze_severity_vs_mttr(df, summary_lines):
    """點8-1：風險等級與處置效率的關聯"""
    summary_lines.append("五、風險等級與處置效率關聯分析")
    resolved_df = df[(df["status"] == "RESOLVED") & (df["mttr_seconds"] > 0)]

    if resolved_df.empty:
        summary_lines.append("目前尚無已簽收處置且 MTTR > 0 的事件，無法分析此關聯，建議累積更多已處置資料後重新執行。")
        summary_lines.append("")
        return None

    mttr_by_severity = resolved_df.groupby("severity")["mttr_seconds"].agg(["mean", "count"])
    mttr_by_severity = mttr_by_severity.reindex(
        [s for s in SEVERITY_ORDER if s in mttr_by_severity.index]
    )

    print("\n【風險等級 vs 平均處置時間(MTTR)】：")
    for severity, row in mttr_by_severity.iterrows():
        print(f"  {severity}: 平均 {row['mean']:.1f} 秒 (樣本數 {int(row['count'])})")

    summary_lines.append("各風險等級的平均處置時間（MTTR）如下：")
    for severity, row in mttr_by_severity.iterrows():
        summary_lines.append(f"  - {severity}：平均 {row['mean']:.1f} 秒（樣本數 {int(row['count'])}）")

    valid_rows = mttr_by_severity.dropna()
    if len(valid_rows) >= 2:
        is_monotonic_decreasing = all(
            valid_rows["mean"].iloc[i] >= valid_rows["mean"].iloc[i + 1]
            for i in range(len(valid_rows) - 1)
        )
        if is_monotonic_decreasing:
            summary_lines.append(
                "觀察到風險等級越高，平均處置時間越短，代表現場人員的簽收處置行為"
                "確實會依照系統判定的風險等級調整優先順序，佐證了本系統風險分級具備"
                "實際的決策參考價值，而不只是形式上的標籤。"
            )
        else:
            summary_lines.append(
                "目前資料顯示風險等級與處置時間並未呈現嚴格的單調關係，這可能代表"
                "樣本數仍不足、或現場人員尚未完全依系統分級調整處置優先順序，"
                "建議累積更多資料後重新檢視，並在論文中誠實記錄此限制。"
            )
    summary_lines.append("")
    return mttr_by_severity


def analyze_density_vs_severity(df, summary_lines):
    """點8-2：煙霧濃度等級與最終風險等級的關聯（交叉統計表）"""
    summary_lines.append("六、煙霧濃度等級與風險等級交叉分析")
    valid_df = df.dropna(subset=["density_level"]) if "density_level" in df.columns else pd.DataFrame()

    if valid_df.empty:
        summary_lines.append("目前尚無足夠含濃度等級資料的事件，無法進行此項交叉分析。")
        summary_lines.append("")
        return None

    crosstab = pd.crosstab(valid_df["density_level"], valid_df["severity"])
    print("\n【煙霧濃度等級 x 風險等級 交叉統計表】：")
    print(crosstab)

    summary_lines.append("濃度等級與風險等級交叉統計（列：濃度，欄：風險等級）：")
    for density_level, row in crosstab.iterrows():
        row_desc = "、".join(f"{sev}: {count}筆" for sev, count in row.items() if count > 0)
        summary_lines.append(f"  - 濃度「{density_level}」：{row_desc}")

    if "濃烈" in crosstab.index:
        heavy_row = crosstab.loc["濃烈"]
        heavy_total = heavy_row.sum()
        heavy_critical_ratio = (
            heavy_row.get("CRITICAL", 0) / heavy_total if heavy_total > 0 else 0
        )
        summary_lines.append(
            f"「濃烈」等級事件中有 {heavy_critical_ratio:.0%} 最終被判定為 CRITICAL，"
            "可作為濃度評分指標（框面積比例 + Laplacian 紋理透光度複合分數）"
            "是否能有效預示風險升級的初步佐證。"
        )
    summary_lines.append("")
    return crosstab


def analyze_camera_vs_occlusion(df, summary_lines):
    """沿用 v2 邏輯的攝影機分組反光分析之外，額外看攝影機與位置確認事件比例的關聯"""
    if "camera_id" not in df.columns or "occlusion_flag" not in df.columns:
        return None
    if df["camera_id"].nunique() <= 1:
        return None

    summary_lines.append("七、攝影機來源與位置確認事件比例關聯")
    camera_occlusion = df.groupby("camera_id")["occlusion_flag"].mean().sort_values(ascending=False)
    print("\n【各攝影機的位置確認事件比例】：")
    for camera_id, ratio in camera_occlusion.items():
        print(f"  {camera_id}: {ratio:.0%}")
        summary_lines.append(f"  - {camera_id}：{ratio:.0%} 的觸發事件屬於「需確認位置」")

    highest_camera = camera_occlusion.idxmax()
    highest_ratio = camera_occlusion.max()
    if highest_ratio > 0.3:
        summary_lines.append(
            f"{highest_camera} 的位置確認事件比例明顯偏高（{highest_ratio:.0%}），"
            "建議優先檢視該攝影機的架設角度與涵蓋範圍是否有死角。"
        )
    summary_lines.append("")
    return camera_occlusion


def main():
    print("=" * 60)
    print("誤判風險與結果歸因關聯分析（v3：新增點8關聯性分析）")
    print("=" * 60)

    df = load_incidents()
    if df.empty:
        print("⚠️ 資料庫目前沒有任何事件紀錄，請先讓系統累積一些告警資料再執行本分析。")
        return

    print(f"📊 共讀取到 {len(df)} 筆歷史事件紀錄，開始分析...\n")

    summary_lines = []
    summary_lines.append("石化園區工業煙囪 AI 安全監控系統 — 誤判風險與結果歸因關聯分析摘要")
    summary_lines.append(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"分析樣本數：{len(df)} 筆歷史告警紀錄")
    summary_lines.append("")

    # ---- 沿用 v2：時段分布 ----
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    hour_counts = df["hour"].value_counts().sort_index()

    print("【時段分布】各小時觸發事件次數：")
    for hour, count in hour_counts.items():
        print(f"  {hour:02d}:00  {'█' * count} ({count})")

    peak_hour = int(hour_counts.idxmax())
    peak_count = int(hour_counts.max())
    peak_ratio = peak_count / len(df)

    summary_lines.append("一、事件發生時段")
    summary_lines.append(
        f"事件最集中發生於 {peak_hour:02d}:00 至 {peak_hour+1:02d}:00 這個時段，"
        f"共 {peak_count} 次，佔全部事件的 {peak_ratio:.0%}。"
    )
    summary_lines.append("")

    # ---- 沿用 v2：反光風險 ----
    glare_scores = []
    for _, row in df.iterrows():
        snapshot_path = row.get("snapshot_path")
        score = compute_glare_score(snapshot_path) if snapshot_path and os.path.exists(snapshot_path) else None
        glare_scores.append(score)
    df["glare_score"] = glare_scores
    valid_df = df.dropna(subset=["glare_score"])

    summary_lines.append("二、反光風險評估")
    if valid_df.empty:
        summary_lines.append("尚無足夠截圖資料可供分析。")
    else:
        avg_glare = valid_df["glare_score"].mean()
        high_glare_count = (valid_df["glare_score"] > 0.15).sum()
        summary_lines.append(f"全部事件平均反光風險分數為 {avg_glare:.3f}。")
        summary_lines.append(f"其中有 {high_glare_count} 筆事件反光風險偏高（>0.15），建議人工複查。")
    summary_lines.append("")

    # ---- 沿用 v2：位置確認事件比例 ----
    if "occlusion_flag" in df.columns:
        occlusion_count = int((df["occlusion_flag"] == 1).sum())
        occlusion_ratio = occlusion_count / len(df)
        summary_lines.append("三、位置確認事件")
        summary_lines.append(
            f"共有 {occlusion_count} 筆事件（佔 {occlusion_ratio:.0%}）觸發時未偵測到煙囪主體。"
        )
        summary_lines.append("")

    summary_lines.append("四、（保留章節，沿用 v2 攝影機反光分組分析，詳見原始 CSV）")
    summary_lines.append("")

    # ---- 【點8新增】關聯性分析 ----
    analyze_severity_vs_mttr(df, summary_lines)
    analyze_density_vs_severity(df, summary_lines)
    analyze_camera_vs_occlusion(df, summary_lines)

    summary_lines.append("八、結論建議")
    summary_lines.append(
        "以上分析基於現有告警紀錄自動產生，樣本數若持續累積，結論會更具統計代表性。"
        "第五至七節的關聯分析可作為論文中討論「系統判定結果是否具備一致性與可信度」"
        "的量化佐證，而不只是描述系統做了什麼。"
    )

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    export_cols = [
        c for c in [
            "id", "timestamp", "hour", "camera_id", "event_type", "severity",
            "risk_level", "density_level", "occlusion_flag", "glare_score",
            "status", "mttr_seconds", "snapshot_path",
        ] if c in df.columns
    ]
    df[export_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n📄 完整原始資料表已匯出至: {OUTPUT_CSV}")
    print(f"📝 純文字摘要已匯出至: {OUTPUT_SUMMARY}")
    print("=" * 60)
    print("\n--- 摘要內容預覽 ---\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
