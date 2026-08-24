from datetime import datetime
import os
import time
import threading
import cv2
import numpy as np
from db_manager import log_incident
from discord_notifier import send_discord_alert_async
from report_generator import trigger_report_generation_async
from ultralytics import YOLO

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT = r"D:\NCKU_paper_Capstone_Project"
MODEL_PATH = os.path.join(
    PROJECT_ROOT, r"runs\detect\chimney_smoke_fire_yolo11s\weights\best.pt"
)
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# 【功能 2：多攝影機】自行編輯這個字典即可新增/移除攝影機來源，不需要改其他程式碼
CAMERA_SOURCES = {
    "CAM-01": os.path.join(PROJECT_ROOT, "PixVerse_Clean_Slow.mp4"),
    "CAM-02": os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4"),
}

ALERT_THRESHOLD_SEC = 3.0
ALERT_COOLDOWN_SEC = 60.0

# ------------------------------------------------------------------------------
# 【功能 10：影片前處理 + 雙畫面比較】
# ENABLE_PREPROCESS：是否對送進模型判斷的畫面做前處理（對比度/亮度調整）
# SHOW_SIDE_BY_SIDE：是否在同一個視窗左右並排顯示「原始畫面」與「前處理後畫面」
#   左邊＝原始畫面（純比較用，不影響偵測）
#   右邊＝前處理後畫面（模型實際拿這張去判斷、觸發告警都以這邊為準）
# ------------------------------------------------------------------------------
ENABLE_PREPROCESS = True
SHOW_SIDE_BY_SIDE = True
PREPROCESS_ALPHA = 1.2
PREPROCESS_BETA = 20

# 【功能 8：反光防誤判】
GLARE_BRIGHTNESS_THRESHOLD = 240
GLARE_HOT_PIXEL_RATIO = 0.6
GLARE_MAX_BOX_AREA_RATIO = 0.02

print_lock = threading.Lock()


def safe_print(camera_id, message):
  with print_lock:
    print(f"[{camera_id}] {message}")


def resolve_anomaly_type(detected_classes):
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


def classify_smoke_density(box_area_ratio):
  if box_area_ratio >= 0.15:
    return "濃烈"
  elif box_area_ratio >= 0.05:
    return "中等"
  else:
    return "輕微"


def is_likely_glare(frame, box_xyxy):
  x1, y1, x2, y2 = [int(v) for v in box_xyxy]
  h, w = frame.shape[:2]
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(w, x2), min(h, y2)
  if x2 <= x1 or y2 <= y1:
    return False

  crop = frame[y1:y2, x1:x2]
  box_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
  gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  hot_pixel_ratio = np.mean(gray_crop >= GLARE_BRIGHTNESS_THRESHOLD)

  return (
      hot_pixel_ratio >= GLARE_HOT_PIXEL_RATIO
      and box_area_ratio <= GLARE_MAX_BOX_AREA_RATIO
  )


def build_occlusion_message(occlusion_flag):
  """
  【功能 1：遮蔽情境判斷 - 呈現改善】
  把「有沒有看到煙囪」這個布林值，翻譯成清楚、可行動的一句話，
  取代原本語意模糊的「疑似遮蔽情境」標籤。
  """
  if occlusion_flag:
    return (
        "📍 位置確認提醒：本次觸發時畫面中未偵測到煙囪主體，"
        "可能原因為（1）事件發生於煙囪以外之廠區位置，或"
        "（2）攝影機視角遭遮蔽物阻擋，建議現場人員確認實際起火/排煙位置。"
    )
  return ""


def process_camera_stream(camera_id, video_path, model):
  safe_print(camera_id, "[INIT] 正在初始化推論引擎...")
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    safe_print(camera_id, f"❌ 錯誤：無法開啟影片，請確認路徑：{video_path}")
    return

  current_state = "NORMAL"
  anomaly_start_time = None
  active_anomaly_type = None
  duration = 0.0
  last_alert_time = 0.0

  video_fps = cap.get(cv2.CAP_PROP_FPS)
  wait_ms = int(1000 / video_fps) if video_fps > 0 else 30

  window_name = f"NCKU Petrochemical Safety Monitor - {camera_id}"

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # 【功能 10】保留一份原始畫面純粹供比較顯示用，不影響判斷
    raw_frame = frame.copy()

    if ENABLE_PREPROCESS:
      frame = cv2.convertScaleAbs(
          frame, alpha=PREPROCESS_ALPHA, beta=PREPROCESS_BETA
      )

    # 模型判斷永遠吃「前處理後」的畫面（若 ENABLE_PREPROCESS=False，等同原始畫面）
    results = model(frame, device=0, verbose=False)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    detected_classes_all = [model.names[int(b.cls[0])] for b in boxes]
    has_chimney = "chimney" in detected_classes_all

    valid_classes = []
    anomaly_boxes = []
    for box in boxes:
      cls_name = model.names[int(box.cls[0])]
      if cls_name in ("fire", "black_smoky", "white_smoky"):
        if is_likely_glare(frame, box.xyxy[0].tolist()):
          continue
        anomaly_boxes.append((cls_name, box.xyxy[0].tolist()))
      valid_classes.append(cls_name)

    current_frame_anomaly = resolve_anomaly_type([c for c, _ in anomaly_boxes])
    has_anomaly = current_frame_anomaly is not None

    current_density_level = None
    if anomaly_boxes:
      h, w = frame.shape[:2]
      max_ratio = 0.0
      for cls_name, xyxy in anomaly_boxes:
        x1, y1, x2, y2 = xyxy
        ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
        max_ratio = max(max_ratio, ratio)
      current_density_level = classify_smoke_density(max_ratio)

    current_time = time.time()

    if current_state == "NORMAL":
      duration = 0.0
      if has_anomaly:
        current_state = "ACCUMULATING"
        anomaly_start_time = current_time
        active_anomaly_type = current_frame_anomaly
      else:
        safe_print(camera_id, "\r🟢 無異常，狀態機守衛中...")

    elif current_state == "ACCUMULATING":
      if has_anomaly:
        active_anomaly_type = current_frame_anomaly
        duration = current_time - anomaly_start_time

        if duration >= ALERT_THRESHOLD_SEC:
          current_state = "TRIGGERED"
          occlusion_flag = 1 if not has_chimney else 0
          occlusion_message = build_occlusion_message(occlusion_flag)

          safe_print(
              camera_id,
              f"🔥 達到滿載臨界值！狀態跳轉: TRIGGERED "
              f"({active_anomaly_type}, 濃度:{current_density_level})",
          )
          if occlusion_message:
            safe_print(camera_id, occlusion_message)

          timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
          safe_type_name = active_anomaly_type.replace("+", "_")
          snapshot_name = f"snapshot_{camera_id}_{timestamp_str}_{safe_type_name}.jpg"
          snapshot_full_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
          cv2.imwrite(snapshot_full_path, annotated_frame)
          safe_print(camera_id, f"📸 [存證存檔] {snapshot_full_path}")

          incident_id = log_incident(
              event_type=active_anomaly_type,
              duration=round(duration, 1),
              snapshot_path=snapshot_full_path,
              camera_id=camera_id,
              density_level=current_density_level,
              occlusion_flag=occlusion_flag,
          )
          safe_print(
              camera_id,
              f"💾 [SQL資料庫] 已新增工安紀錄 (ID: #{incident_id}, 狀態: PENDING)",
          )

          trigger_report_generation_async(
              incident_id=incident_id, event_type=active_anomaly_type
          )

          if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_discord_alert_async(
                timestamp=readable_time,
                duration=round(duration, 1),
                event_type=active_anomaly_type or "smoke",
                camera_id=camera_id,
                density_level=current_density_level,
                occlusion_flag=occlusion_flag,
            )
            last_alert_time = current_time
      else:
        safe_print(camera_id, "🟢 異常中斷，狀態機重置回 NORMAL")
        current_state = "NORMAL"
        anomaly_start_time = None

    elif current_state == "TRIGGERED":
      if has_anomaly:
        active_anomaly_type = current_frame_anomaly
        duration = current_time - anomaly_start_time
      else:
        safe_print(camera_id, "🟢 工安危機解除，狀態機重置")
        current_state = "NORMAL"
        anomaly_start_time = None

    # UI 視覺化（畫在 annotated_frame 上，這是右半邊要顯示的內容）
    bar_x, bar_y = 10, 40
    bar_width, bar_height = 120, 8
    if current_state == "NORMAL":
      status_color, progress_ratio = (0, 255, 0), 0.0
    elif current_state == "ACCUMULATING":
      status_color = (0, 165, 255)
      progress_ratio = min(duration / ALERT_THRESHOLD_SEC, 1.0)
    else:
      status_color, progress_ratio = (0, 0, 255), 1.0

    cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    fill_width = int(bar_width * progress_ratio)
    if fill_width > 0:
      cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), status_color, -1)

    cv2.putText(annotated_frame, f"[{camera_id}] STATE: {current_state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Time: {duration:.1f}s / {ALERT_THRESHOLD_SEC}s", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    if current_density_level:
      density_label_en = {"輕微": "Light", "中等": "Moderate", "濃烈": "Heavy"}.get(current_density_level, current_density_level)
      cv2.putText(annotated_frame, f"Density: {density_label_en}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if current_state == "TRIGGERED":
      alert_label = "[COMPOSITE CRITICAL]" if active_anomaly_type == "black_smoky+fire" else "[CRITICAL ALERT]"
      cv2.putText(annotated_frame, alert_label, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

    # 【功能 10】組成左右並排比較畫面
    if ENABLE_PREPROCESS and SHOW_SIDE_BY_SIDE:
      raw_display = raw_frame.copy()
      cv2.putText(raw_display, "ORIGINAL (Comparison Only)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(annotated_frame, "PREPROCESSED (Detection Basis)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
      # 確保左右高度一致才能拼接
      if raw_display.shape[0] != annotated_frame.shape[0]:
        raw_display = cv2.resize(raw_display, (annotated_frame.shape[1], annotated_frame.shape[0]))
      display_frame = cv2.hconcat([raw_display, annotated_frame])
    else:
      display_frame = annotated_frame

    cv2.imshow(window_name, display_frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyWindow(window_name)
  safe_print(camera_id, "[EXIT] 影像播放結束，釋放視窗資源。")


def main():
  print(
      f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在載入 YOLOv11 模型..."
  )
  model = YOLO(MODEL_PATH)

  camera_threads = []
  for camera_id, video_path in CAMERA_SOURCES.items():
    t = threading.Thread(
        target=process_camera_stream,
        args=(camera_id, video_path, model),
        daemon=False,
    )
    camera_threads.append(t)
    t.start()

  for t in camera_threads:
    t.join()

  cv2.destroyAllWindows()
  print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 所有攝影機串流已結束。")

  active_threads = [
      t for t in threading.enumerate()
      if t is not threading.main_thread() and t.is_alive()
  ]
  if active_threads:
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 背景任務（報告生成 / 通知）"
        " 仍在執行中，請稍候..."
    )
    while any(t.is_alive() for t in active_threads):
      time.sleep(1.0)
      active_threads = [
          t for t in threading.enumerate()
          if t is not threading.main_thread() and t.is_alive()
      ]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 所有背景任務已完成。")


if __name__ == "__main__":
  main()
