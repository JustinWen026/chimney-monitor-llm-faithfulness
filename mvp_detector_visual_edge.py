from datetime import datetime
import os
import time
import cv2
from ultralytics import YOLO

# 匯入獨立拆分的 Discord 告警模組[cite: 8]
from discord_notifier import send_discord_alert_async  #[cite: 8]

# ==============================================================================
# 1. 系統參數配置[cite: 8]
# ==============================================================================
PROJECT_ROOT = r"D:\NCKU_paper_Capstone_Project"
VIDEO_PATH = r"D:\NCKU_paper_Capstone_Project\PixVerse_Clean_Slow.mp4"
MODEL_PATH = os.path.join(
    PROJECT_ROOT, r"runs\detect\chimney_smoke_fire_yolo11s\weights\best.pt"
)

ALERT_THRESHOLD_SEC = 3.0  # 狀態機安全閾值 (秒)[cite: 8]
ALERT_COOLDOWN_SEC = 60.0  # 告警防連發冷卻時間 (秒)[cite: 8]


def main():
  print(
      f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11"
      " 推論引擎..."
  )  #[cite: 8]
  model = YOLO(MODEL_PATH)  #[cite: 8]
  cap = cv2.VideoCapture(VIDEO_PATH)  #[cite: 8]

  if not cap.isOpened():  #[cite: 8]
    print(f"❌ 錯誤：無法開啟影片，請確認路徑：{VIDEO_PATH}")
    return

  # 狀態機與告警時序核心變數[cite: 8]
  current_state = "NORMAL"  #[cite: 8]
  anomaly_start_time = None
  active_anomaly_type = None
  duration = 0.0  #[cite: 8]
  last_alert_time = 0.0  # 記錄上一次成功觸發 Discord 告警的時間戳[cite: 8]

  # FPS 平滑緩衝[cite: 8]
  frame_counter = 0  #[cite: 8]
  display_pipe_fps = 0.0  #[cite: 8]

  print("\n" + "=" * 95)  #[cite: 8]
  print(" 🚀 工業煙囪 AI 影像安全監控系統已啟動...")
  print(
      " 說明：結合 YOLOv11 視覺辨識、時序狀態機演算法與 Discord 即時主動通報。"
  )
  print("=" * 95 + "\n")  #[cite: 8]

  video_fps = cap.get(cv2.CAP_PROP_FPS)  #[cite: 8]
  wait_ms = int(1000 / video_fps) if video_fps > 0 else 30  #[cite: 8]

  while cap.isOpened():  #[cite: 8]
    ret, frame = cap.read()  #[cite: 8]
    if not ret:  #[cite: 8]
      break  #[cite: 8]

    # GPU 推論 (RTX 5060 加速)[cite: 8]
    results = model(frame, device=0, verbose=False)  #[cite: 8]
    annotated_frame = results[0].plot()  #[cite: 8]

    # FPS 計算[cite: 8]
    t_pre = results[0].speed.get("preprocess", 0.0)  #[cite: 8]
    t_inf = results[0].speed.get("inference", 0.0)  #[cite: 8]
    t_post = results[0].speed.get("postprocess", 0.0)  #[cite: 8]
    total_latency = t_pre + t_inf + t_post  #[cite: 8]

    frame_counter += 1  #[cite: 8]
    if frame_counter % 10 == 0 or frame_counter == 1:  #[cite: 8]
      display_pipe_fps = (
          1000.0 / total_latency if total_latency > 0 else 0.0
      )  #[cite: 8]

    # 解析標籤：納入 black_smoky, white_smoky 以及 fire
    detected_classes = [
        model.names[int(box.cls[0])] for box in results[0].boxes
    ]  #[cite: 8]
    target_anomalies = ["fire", "black_smoky", "white_smoky"]

    has_anomaly = any(cls in target_anomalies for cls in detected_classes)
    current_frame_anomaly = next(
        (cls for cls in detected_classes if cls in target_anomalies), None
    )

    current_time = time.time()  #[cite: 8]

    # ==============================================================================
    # ⚙️ 時序狀態機演算法內核與運算 (含防連發機制)[cite: 8]
    # ==============================================================================
    if current_state == "NORMAL":  #[cite: 8]
      duration = 0.0  #[cite: 8]
      if has_anomaly:
        current_state = "ACCUMULATING"  #[cite: 8]
        anomaly_start_time = current_time
        active_anomaly_type = current_frame_anomaly
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 狀態機跳轉: NORMAL ➔"
            f" ACCUMULATING (偵測到: {active_anomaly_type})"
        )
      else:
        print(
            "\r[STATE: NORMAL] 🟢 煙囪無異常，狀態機守衛中... | 效能:"
            f" {display_pipe_fps:.1f} FPS   ",
            end="",
            flush=True,
        )  #[cite: 8]

    elif current_state == "ACCUMULATING":  #[cite: 8]
      if has_anomaly:
        duration = current_time - anomaly_start_time
        print(
            "\r[STATE: ACCUMULATING] ⏳ 異常時間蓄能中:"
            f" {duration:.2f}s / {ALERT_THRESHOLD_SEC}s... ",
            end="",
            flush=True,
        )  #[cite: 8]

        if duration >= ALERT_THRESHOLD_SEC:  #[cite: 8]
          # 關鍵瞬間：時序滿 3 秒，突破臨界點！[cite: 8]
          current_state = "TRIGGERED"  #[cite: 8]
          print(
              f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔥"
              " 異常達到滿載臨界值！狀態跳轉: TRIGGERED"
          )  #[cite: 8]

          # 安全通報防連發邏輯[cite: 8]
          if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:  #[cite: 8]
            readable_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )  #[cite: 8]
            send_discord_alert_async(
                timestamp=readable_time,
                duration=round(duration, 1),
                event_type=active_anomaly_type or "smoke",
            )  #[cite: 8]
            last_alert_time = current_time  #[cite: 8]
          else:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [防連發保護]"
                " 處於冷卻期內，跳過本次 Discord 卡片發送。"
            )  #[cite: 8]
      else:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 異常中斷，狀態機重置回"
            " NORMAL"
        )  #[cite: 8]
        current_state = "NORMAL"  #[cite: 8]
        anomaly_start_time = None

    elif current_state == "TRIGGERED":  #[cite: 8]
      if has_anomaly:
        duration = current_time - anomaly_start_time
        print(
            "\r[STATE: TRIGGERED] 🔴 警報持續中！累計異常時間:"
            f" {duration:.2f}s   ",
            end="",
            flush=True,
        )  #[cite: 8]
      else:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 工安危機解除，狀態機重置"
        )  #[cite: 8]
        current_state = "NORMAL"  #[cite: 8]
        anomaly_start_time = None

    # ==============================================================================
    # 🎨 狀態機演算法視覺化 (UI 疊加)[cite: 8]
    # ==============================================================================
    bar_x, bar_y = 10, 40  #[cite: 8]
    bar_width, bar_height = 120, 8  #[cite: 8]

    if current_state == "NORMAL":  #[cite: 8]
      status_color = (0, 255, 0)  # 綠色[cite: 8]
      progress_ratio = 0.0  #[cite: 8]
    elif current_state == "ACCUMULATING":  #[cite: 8]
      status_color = (0, 165, 255)  # 橘黃色[cite: 8]
      progress_ratio = min(duration / ALERT_THRESHOLD_SEC, 1.0)  #[cite: 8]
    else:  # TRIGGERED[cite: 8]
      status_color = (0, 0, 255)  # 鮮紅色[cite: 8]
      progress_ratio = 1.0  #[cite: 8]

    # 1. 繪製進度條外框[cite: 8]
    cv2.rectangle(
        annotated_frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        1,
    )  #[cite: 8]
    # 2. 動態填滿進度條[cite: 8]
    fill_width = int(bar_width * progress_ratio)  #[cite: 8]
    if fill_width > 0:  #[cite: 8]
      cv2.rectangle(
          annotated_frame,
          (bar_x, bar_y),
          (bar_x + fill_width, bar_y + bar_height),
          status_color,
          -1,
      )  #[cite: 8]

    # 3. 疊加文字資訊[cite: 8]
    cv2.putText(
        annotated_frame,
        f"STATE: {current_state}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        2,
        cv2.LINE_AA,
    )  #[cite: 8]
    cv2.putText(
        annotated_frame,
        f"Time: {duration:.1f}s / {ALERT_THRESHOLD_SEC}s",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )  #[cite: 8]
    cv2.putText(
        annotated_frame,
        f"FPS: {display_pipe_fps:.1f}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )  #[cite: 8]

    # 4. 警報狀態提示[cite: 8]
    if current_state == "TRIGGERED":  #[cite: 8]
      cv2.putText(
          annotated_frame,
          "[CRITICAL ALERT]",
          (10, 110),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.45,
          (0, 0, 255),
          2,
          cv2.LINE_AA,
      )  #[cite: 8]

    cv2.imshow(
        "NCKU Petrochemical Chimney - Advanced Algorithmic Dashboard",
        annotated_frame,
    )  #[cite: 8]

    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):  #[cite: 8]
      break  #[cite: 8]

  cap.release()  #[cite: 8]
  cv2.destroyAllWindows()  #[cite: 8]


if __name__ == "__main__":  #[cite: 8]
  main()  #[cite: 8]