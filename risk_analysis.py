"""
risk_analysis_v2.py
--------------------
在原本 CSV 匯出的基礎上，新增一份純文字、白話的分析摘要，
不需要打開 Excel 或看數字表格，直接讀這份摘要就能知道結論。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python risk_analysis_v2.py
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
GLARE_BRIGHTNESS_THRESHOLD = 240


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


def main():
  print("=" * 60)
  print("誤判風險分析（功能 9）")
  print("=" * 60)

  df = load_incidents()
  if df.empty:
    print("⚠️ 資料庫目前沒有任何事件紀錄，請先讓系統累積一些告警資料再執行本分析。")
    return

  print(f"📊 共讀取到 {len(df)} 筆歷史事件紀錄，開始分析...\n")

  summary_lines = []
  summary_lines.append("石化園區工業煙囪 AI 安全監控系統 — 誤判風險與時段分析摘要")
  summary_lines.append(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  summary_lines.append(f"分析樣本數：{len(df)} 筆歷史告警紀錄")
  summary_lines.append("")

  # 分析一：時段分布
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
  if 10 <= peak_hour <= 15:
    summary_lines.append(
        "此時段落在白天日照較強的區間，建議進一步確認是否與陽光角度、"
        "金屬設備反光造成的誤判有關，而非單純異常事件本身增加。"
    )
  summary_lines.append("")

  # 分析二 + 三：反光風險
  glare_scores = []
  for _, row in df.iterrows():
    snapshot_path = row.get("snapshot_path")
    score = compute_glare_score(snapshot_path) if snapshot_path and os.path.exists(snapshot_path) else None
    glare_scores.append(score)

  df["glare_score"] = glare_scores
  valid_df = df.dropna(subset=["glare_score"])

  summary_lines.append("二、反光風險評估")
  if valid_df.empty:
    print("⚠️ 找不到任何可分析的截圖檔案，反光風險分析略過。")
    summary_lines.append("尚無足夠截圖資料可供分析，建議累積更多告警紀錄後重新執行本分析。")
  else:
    avg_glare = valid_df["glare_score"].mean()
    high_glare_count = (valid_df["glare_score"] > 0.15).sum()
    high_glare_ratio = high_glare_count / len(valid_df)

    print("【反光風險排名】反光風險分數最高的前 5 筆事件：")
    top5 = valid_df.sort_values("glare_score", ascending=False).head(5)
    for _, row in top5.iterrows():
      print(f"  事件 #{row['id']} | {row.get('camera_id', 'CAM-01')} | 反光分數: {row['glare_score']:.3f}")

    summary_lines.append(
        f"全部事件平均反光風險分數為 {avg_glare:.3f}"
        f"（數值越接近 1 代表畫面中過曝、反光區域比例越高）。"
    )
    summary_lines.append(
        f"其中有 {high_glare_count} 筆事件（佔 {high_glare_ratio:.0%}）反光風險偏高（>0.15），"
        "建議人工複查這幾筆截圖，確認是否為反光造成的誤判，而非真實煙霧或火焰。"
    )

    if "camera_id" in valid_df.columns and valid_df["camera_id"].nunique() > 1:
      camera_glare = valid_df.groupby("camera_id")["glare_score"].mean().sort_values(ascending=False)
      highest_camera = camera_glare.idxmax()
      highest_score = camera_glare.max()
      print("\n【依攝影機分組】各攝影機平均反光風險分數：")
      for camera_id, avg_score in camera_glare.items():
        risk_label = "🔴 高風險" if avg_score > 0.15 else ("🟡 中等" if avg_score > 0.05 else "🟢 低")
        print(f"  {camera_id}: 平均反光分數 {avg_score:.3f} {risk_label}")

      summary_lines.append("")
      summary_lines.append(
          f"以攝影機分組比較，{highest_camera} 的平均反光風險分數最高（{highest_score:.3f}），"
      )
      if highest_score > 0.1:
        summary_lines.append(
            f"建議優先現場檢查 {highest_camera} 拍攝範圍內是否有強反光表面"
            "（例如金屬管線、淺色塗裝），可評估調整該區域塗裝顏色或加裝遮光設施，"
            "以降低系統誤判率並提升整體監控可靠度。"
        )
      else:
        summary_lines.append("但整體反光風險仍在可接受範圍內，暫不需要特別處置。")
  summary_lines.append("")

  # 分析三：遮蔽情境比例
  if "occlusion_flag" in df.columns:
    occlusion_count = int((df["occlusion_flag"] == 1).sum())
    occlusion_ratio = occlusion_count / len(df)
    summary_lines.append("三、位置確認事件")
    summary_lines.append(
        f"共有 {occlusion_count} 筆事件（佔 {occlusion_ratio:.0%}）觸發時未偵測到煙囪主體，"
        "代表這些事件可能發生於煙囪以外的廠區位置，或攝影機視角遭遮蔽物阻擋。"
    )
    if occlusion_ratio > 0.3:
      summary_lines.append(
          "此比例偏高，建議檢視目前攝影機架設角度是否涵蓋足夠廣的監控範圍，"
          "或考慮增設攝影機補足視角死角。"
      )
    summary_lines.append("")

  summary_lines.append("四、結論建議")
  summary_lines.append(
      "以上分析基於現有告警紀錄自動產生，樣本數若持續累積，結論會更具統計代表性。"
      "建議定期（例如每兩週）重新執行本分析，追蹤反光風險與位置確認事件比例的變化趨勢。"
  )

  # 輸出檔案
  os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
  export_cols = [
      c for c in [
          "id", "timestamp", "hour", "camera_id", "event_type", "severity",
          "density_level", "occlusion_flag", "glare_score", "snapshot_path",
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
