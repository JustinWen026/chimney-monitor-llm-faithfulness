"""
risk_analysis.py
-----------------
【功能 9：誤判分析與預警建議】

這是一支獨立的離線分析腳本，不影響、不需要修改你既有的即時監控系統。
用途：讀取資料庫裡所有觸發過的事件截圖，分析：
  1. 事件發生的時段分布（是否集中在特定時間，例如強光時段）
  2. 每張截圖的反光風險評分（畫面中有多少比例是疑似過曝的死白區域）
  3. 依攝影機來源分組，找出「反覆出現高反光風險」的特定攝影機/位置

分析結果會：
  - 印出摘要統計到終端機
  - 存成一份 CSV 報表，方便你之後貼進論文附錄或簡報

使用時機：累積至少 10~20 筆告警紀錄後執行效果較好，樣本太少統計意義不大。
執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python risk_analysis.py
"""

import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

DB_PATH = r"D:\NCKU_paper_Capstone_Project\safety_incidents.db"
OUTPUT_CSV = r"D:\NCKU_paper_Capstone_Project\reports\misjudgment_analysis.csv"
GLARE_BRIGHTNESS_THRESHOLD = 240  # 與偵測端一致，判定「過曝」的像素亮度門檻


def compute_glare_score(image_path):
  """
  計算一張截圖的反光風險分數：畫面中過曝像素的佔比（0~1，越高代表反光疑慮越高）。
  這是簡易的整張圖統計，跟即時偵測時「只看單一偵測框」的邏輯不同，
  這裡是要抓「這張現場照片整體光線條件如何」，用來找出容易起反光的位置/時段。
  """
  img = cv2.imread(image_path)
  if img is None:
    return None
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hot_pixel_ratio = float(np.mean(gray >= GLARE_BRIGHTNESS_THRESHOLD))
  return round(hot_pixel_ratio, 4)


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

  # ------------------------------------------------------------------------
  # 分析一：事件時段分布
  # ------------------------------------------------------------------------
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df["hour"] = df["timestamp"].dt.hour
  hour_counts = df["hour"].value_counts().sort_index()

  print("【時段分布】各小時觸發事件次數：")
  for hour, count in hour_counts.items():
    bar = "█" * count
    print(f"  {hour:02d}:00  {bar} ({count})")

  peak_hour = hour_counts.idxmax()
  print(
      f"\n👉 事件最集中的時段是 {peak_hour:02d}:00 附近"
      f"（共 {hour_counts.max()} 次），可對照該時段陽光角度是否為主要因素。\n"
  )

  # ------------------------------------------------------------------------
  # 分析二：逐張截圖計算反光風險分數
  # ------------------------------------------------------------------------
  glare_scores = []
  for _, row in df.iterrows():
    snapshot_path = row.get("snapshot_path")
    if snapshot_path and os.path.exists(snapshot_path):
      score = compute_glare_score(snapshot_path)
    else:
      score = None
    glare_scores.append(score)

  df["glare_score"] = glare_scores
  valid_df = df.dropna(subset=["glare_score"])

  if valid_df.empty:
    print("⚠️ 找不到任何可分析的截圖檔案，反光風險分析略過。")
  else:
    print("【反光風險排名】反光風險分數最高的前 5 筆事件：")
    top5 = valid_df.sort_values("glare_score", ascending=False).head(5)
    for _, row in top5.iterrows():
      camera = row.get("camera_id", "CAM-01")
      print(
          f"  事件 #{row['id']} | {camera} | {row['timestamp']} | "
          f"反光分數: {row['glare_score']:.3f} | 截圖: {row['snapshot_path']}"
      )

    # ------------------------------------------------------------------------
    # 分析三：依攝影機分組，找出反覆高反光的位置
    # ------------------------------------------------------------------------
    if "camera_id" in valid_df.columns:
      camera_glare = (
          valid_df.groupby("camera_id")["glare_score"]
          .mean()
          .sort_values(ascending=False)
      )
      print("\n【依攝影機分組】各攝影機平均反光風險分數：")
      for camera_id, avg_score in camera_glare.items():
        risk_label = "🔴 高風險" if avg_score > 0.15 else ("🟡 中等" if avg_score > 0.05 else "🟢 低")
        print(f"  {camera_id}: 平均反光分數 {avg_score:.3f} {risk_label}")

      highest_camera = camera_glare.idxmax()
      if camera_glare.max() > 0.1:
        print(
            f"\n👉 建議重點關注：{highest_camera} 平均反光風險偏高，"
            "可考慮現場檢查該攝影機視角範圍內是否有強反光表面"
            "（例如金屬管線或淺色塗裝），評估調整塗裝顏色或加裝遮光設施。"
        )

  # ------------------------------------------------------------------------
  # 輸出 CSV 報表
  # ------------------------------------------------------------------------
  os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
  export_cols = [
      c for c in [
          "id", "timestamp", "hour", "camera_id", "event_type", "severity",
          "density_level", "occlusion_flag", "glare_score", "snapshot_path",
      ] if c in df.columns
  ]
  df[export_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
  print(f"\n📄 完整分析結果已匯出至: {OUTPUT_CSV}")
  print("=" * 60)


if __name__ == "__main__":
  main()
