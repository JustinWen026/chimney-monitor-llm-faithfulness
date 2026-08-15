from datetime import datetime
import os
import time
import cv2
from db_manager import log_incident
from discord_notifier import send_discord_alert_async
from ultralytics import YOLO

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT = r"D:\NCKU_paper_Capstone_Project"
VIDEO_PATH = os.path.join(PROJECT_ROOT, "PixVerse_Clean_Slow.mp4")
MODEL_PATH = os.path.join(
    PROJECT_ROOT, r"runs\detect\chimney_smoke_fire_yolo11s\weights\best.pt"
)
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

ALERT_THRESHOLD_SEC = 3.0  # 狀態機安全閾值 (秒)
ALERT_COOLDOWN_SEC = 60.0  # 告警防連發冷卻時間 (秒)


def resolve_anomaly_type(detected_classes):
  """
  v3：Layer 3 歸因邏輯的偵測端對應版本。

  判斷優先序（由高到低）：
      1. black_smoky AND fire 同時存在 → "black_smoky+fire"（複合風險，最高優先）
      2. fire 存在 → "fire"
      3. black_smoky 存在 → "black_smoky"
      4. white_smoky 存在 → "white_smoky"
      5. 皆無 → None

  舊版邏輯只用 next() 抓 detected_classes 裡第一個符合的類別，
  沒有考慮優先序也沒有處理複合情況，v3 修正這點。
  """
  has_fire = "fire" in detected_classes
  has_black = "black_smoky" in detected_classes
  has_white = "white_smoky" in detected_classes

  if has_fire and has_black:
    return "black_smoky+fire"
  elif has_fire:
    return "fire"
  elif has_black:
    return "black_smoky"
  elif has_white:
    return "white_smoky"
  else:
    return None


def main():
  print(
      f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11"
      " 推論引擎 (V3 複合風險判斷版)..."
  )
  model = YOLO(MODEL_PATH)
  cap = cv2.VideoCapture(VIDEO_PATH)

  if not cap.isOpened():
    print(f"❌ 錯誤：無法開啟影片，請確認路徑：{VIDEO_PATH}")
    return

  current_state = "NORMAL"
  anomaly_start_time = None
  active_anomaly_type = None
  duration = 0.0
  last_alert_time = 0.0

  frame_counter = 0
  display_pipe_fps = 0.0

  video_fps = cap.get(cv2.CAP_PROP_FPS)
  wait_ms = int(1000 / video_fps) if video_fps > 0 else 30

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # GPU 推論
    results = model(frame, device=0, verbose=False)
    annotated_frame = results[0].plot()

    t_pre = results[0].speed.get("preprocess", 0.0)
    t_inf = results[0].speed.get("inference", 0.0)
    t_post = results[0].speed.get("postprocess", 0.0)
    total_latency = t_pre + t_inf + t_post

    frame_counter += 1
    if frame_counter % 10 == 0 or frame_counter == 1:
      display_pipe_fps = (
          1000.0 / total_latency if total_latency > 0 else 0.0
      )

    detected_classes = [
        model.names[int(box.cls[0])] for box in results[0].boxes
    ]

    # v3：改用優先序 + 複合風險判斷，取代舊版單純 next() 抓第一個符合類別
    current_frame_anomaly = resolve_anomaly_type(detected_classes)
    has_anomaly = current_frame_anomaly is not None

    current_time = time.time()

    # ==============================================================================
    # ⚙️ 時序狀態機與存證紀錄
    # ==============================================================================
    if current_state == "NORMAL":
      duration = 0.0
      if has_anomaly:
        current_state = "ACCUMULATING"
        anomaly_start_time = current_time
        active_anomaly_type = current_frame_anomaly
      else:
        print(
            "\r[STATE: NORMAL] 🟢 煙囪無異常，狀態機守衛中... | 效能:"
            f" {display_pipe_fps:.1f} FPS   ",
            end="",
            flush=True,
        )

    elif current_state == "ACCUMULATING":
      if has_anomaly:
        # v3：狀態持續累積期間，若偵測到更高風險組合（例如原本只有黑煙，
        # 之後火焰也出現形成複合風險），即時更新 active_anomaly_type
        active_anomaly_type = current_frame_anomaly
        duration = current_time - anomaly_start_time
        print(
            "\r[STATE: ACCUMULATING] ⏳ 異常時間蓄能中:"
            f" {duration:.2f}s / {ALERT_THRESHOLD_SEC}s | 類型:"
            f" {active_anomaly_type}   ",
            end="",
            flush=True,
        )

        if duration >= ALERT_THRESHOLD_SEC:
          current_state = "TRIGGERED"
          print(
              f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔥 達到滿載臨界值！狀態跳轉:"
              f" TRIGGERED ({active_anomaly_type})"
          )

          # 📸 功能 3：自動截圖存證
          timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
          safe_type_name = active_anomaly_type.replace("+", "_")
          snapshot_name = f"snapshot_{timestamp_str}_{safe_type_name}.jpg"
          snapshot_full_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
          cv2.imwrite(snapshot_full_path, annotated_frame)
          print(f"📸 [存證存檔] 現場影像已保存至: {snapshot_full_path}")

          # 💾 功能 1：寫入 SQLite 工安資料庫
          incident_id = log_incident(
              event_type=active_anomaly_type,
              duration=round(duration, 1),
              snapshot_path=snapshot_full_path,
          )
          print(
              f"💾 [SQL資料庫] 已新增工安紀錄 (ID: #{incident_id},"
              f" 狀態: PENDING)"
          )

          # 🚨 Discord 即時非同步發送
          if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_discord_alert_async(
                timestamp=readable_time,
                duration=round(duration, 1),
                event_type=active_anomaly_type or "smoke",
            )
            last_alert_time = current_time
      else:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 異常中斷，狀態機重置回"
            " NORMAL"
        )
        current_state = "NORMAL"
        anomaly_start_time = None

    elif current_state == "TRIGGERED":
      if has_anomaly:
        active_anomaly_type = current_frame_anomaly
        duration = current_time - anomaly_start_time
        print(
            "\r[STATE: TRIGGERED] 🔴 警報持續中！累計異常時間:"
            f" {duration:.2f}s | 類型: {active_anomaly_type}   ",
            end="",
            flush=True,
        )
      else:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 工安危機解除，狀態機重置"
        )
        current_state = "NORMAL"
        anomaly_start_time = None

    # ==============================================================================
    # 🎨 狀態機 UI 視覺化
    # ==============================================================================
    bar_x, bar_y = 10, 40
    bar_width, bar_height = 120, 8

    if current_state == "NORMAL":
      status_color = (0, 255, 0)
      progress_ratio = 0.0
    elif current_state == "ACCUMULATING":
      status_color = (0, 165, 255)
      progress_ratio = min(duration / ALERT_THRESHOLD_SEC, 1.0)
    else:
      status_color = (0, 0, 255)
      progress_ratio = 1.0

    cv2.rectangle(
        annotated_frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        1,
    )
    fill_width = int(bar_width * progress_ratio)
    if fill_width > 0:
      cv2.rectangle(
          annotated_frame,
          (bar_x, bar_y),
          (bar_x + fill_width, bar_y + bar_height),
          status_color,
          -1,
      )

    cv2.putText(
        annotated_frame,
        f"STATE: {current_state}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        f"Time: {duration:.1f}s / {ALERT_THRESHOLD_SEC}s",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        f"FPS: {display_pipe_fps:.1f}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if current_state == "TRIGGERED":
      alert_label = (
          "[COMPOSITE CRITICAL]"
          if active_anomaly_type == "black_smoky+fire"
          else "[CRITICAL ALERT]"
      )
      cv2.putText(
          annotated_frame,
          alert_label,
          (10, 110),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.45,
          (0, 0, 255),
          2,
          cv2.LINE_AA,
      )

    cv2.imshow("NCKU Petrochemical Safety Monitor v3.0", annotated_frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
