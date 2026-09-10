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

CAMERA_SOURCES = {
    "CAM-01": os.path.join(PROJECT_ROOT, "PixVerse_Clean_Slow.mp4"),
    "CAM-02": os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4"),
}

ALERT_THRESHOLD_SEC = 3.0
ALERT_COOLDOWN_SEC = 60.0

CONFIDENCE_THRESHOLD = 0.5

ENABLE_PREPROCESS = True
SHOW_SIDE_BY_SIDE = True
PREPROCESS_ALPHA = 1.1
PREPROCESS_BETA = 10

GLARE_BRIGHTNESS_THRESHOLD = 240
GLARE_HOT_PIXEL_RATIO = 0.6
GLARE_MAX_BOX_AREA_RATIO = 0.02

LAPLACIAN_REFERENCE_MAX = 500.0
DENSITY_HEAVY_THRESHOLD = 0.6
DENSITY_MODERATE_THRESHOLD = 0.35

print_lock = threading.Lock()

# 【v7 修正】原本用 threading.enumerate() 掃描「所有存活執行緒」來判斷背景任務
# 是否跑完，這個做法會被 huggingface_hub / torch 等第三方套件內部常駐、不會
# 主動結束的執行緒池誤判，導致主程式跑完偵測後卡住不動，要手動 Ctrl+C 才能繼續。
# 改成只追蹤「我們自己真正建立」的報告生成執行緒，只等這些跑完即可。
report_threads = []
report_threads_lock = threading.Lock()

# ------------------------------------------------------------------------------
# 【多視窗顯示問題修正】
# Windows 上 cv2.imshow 不是為多執行緒設計的，多支攝影機各自的執行緒
# 直接呼叫 imshow 常常只有其中一個視窗能正常顯示。
# 修正方式：處理執行緒只負責運算並把畫面放進這個共用字典，
# 實際的 imshow / waitKey / destroyWindow 全部統一交給主執行緒處理。
# ------------------------------------------------------------------------------
latest_frames = {}
frames_lock = threading.Lock()
stop_event = threading.Event()
finished_cameras = set()
finished_lock = threading.Lock()


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


def compute_density_score(frame, box_xyxy):
  x1, y1, x2, y2 = [int(v) for v in box_xyxy]
  h, w = frame.shape[:2]
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(w, x2), min(h, y2)
  if x2 <= x1 or y2 <= y1:
    return 0.0

  area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
  area_score = min(area_ratio / 0.2, 1.0)

  crop_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
  laplacian_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
  opacity_score = 1.0 - min(laplacian_var / LAPLACIAN_REFERENCE_MAX, 1.0)

  return 0.5 * area_score + 0.5 * opacity_score


def classify_smoke_density(density_score):
  if density_score >= DENSITY_HEAVY_THRESHOLD:
    return "濃烈"
  elif density_score >= DENSITY_MODERATE_THRESHOLD:
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
  if occlusion_flag:
    return (
        "📍 位置確認提醒：本次觸發時畫面中未偵測到煙囪主體，"
        "可能原因為（1）事件發生於煙囪以外之廠區位置，或"
        "（2）攝影機視角遭遮蔽物阻擋，建議現場人員確認實際起火/排煙位置。"
    )
  return ""


def process_camera_stream(camera_id, video_path, model):
  """
  【重要】這個函式只負責「運算」：讀影格、跑模型、狀態機判斷、寫資料庫。
  完全不呼叫 cv2.imshow / cv2.waitKey / cv2.destroyWindow 這類 GUI 操作，
  只把畫好的畫面放進 latest_frames 這個共用字典，交給主執行緒統一顯示。
  """
  safe_print(camera_id, "[INIT] 正在初始化推論引擎...")
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    safe_print(camera_id, f"❌ 錯誤：無法開啟影片，請確認路徑：{video_path}")
    with finished_lock:
      finished_cameras.add(camera_id)
    return

  current_state = "NORMAL"
  anomaly_start_time = None
  active_anomaly_type = None
  duration = 0.0
  last_alert_time = 0.0

  video_fps = cap.get(cv2.CAP_PROP_FPS)
  frame_interval = 1.0 / video_fps if video_fps > 0 else 1.0 / 30

  while cap.isOpened() and not stop_event.is_set():
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
      break

    raw_frame = frame.copy()

    if ENABLE_PREPROCESS:
      frame = cv2.convertScaleAbs(frame, alpha=PREPROCESS_ALPHA, beta=PREPROCESS_BETA)

    results = model(frame, device=0, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    detected_classes_all = [model.names[int(b.cls[0])] for b in boxes]
    has_chimney = "chimney" in detected_classes_all

    anomaly_boxes = []
    for box in boxes:
      cls_name = model.names[int(box.cls[0])]
      if cls_name in ("fire", "black_smoky", "white_smoky"):
        if is_likely_glare(frame, box.xyxy[0].tolist()):
          continue
        anomaly_boxes.append((cls_name, box.xyxy[0].tolist()))

    current_frame_anomaly = resolve_anomaly_type([c for c, _ in anomaly_boxes])
    has_anomaly = current_frame_anomaly is not None

    current_density_level = None
    current_density_score = None
    if anomaly_boxes:
      best_score = 0.0
      for cls_name, xyxy in anomaly_boxes:
        score_val = compute_density_score(frame, xyxy)
        best_score = max(best_score, score_val)
      current_density_score = best_score
      current_density_level = classify_smoke_density(best_score)

    current_time = time.time()

    if current_state == "NORMAL":
      duration = 0.0
      if has_anomaly:
        current_state = "ACCUMULATING"
        anomaly_start_time = current_time
        active_anomaly_type = current_frame_anomaly

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
              f"({active_anomaly_type}, 濃度:{current_density_level}, "
              f"分數:{current_density_score:.3f})",
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

          new_report_thread = trigger_report_generation_async(
              incident_id=incident_id, event_type=active_anomaly_type
          )
          if new_report_thread is not None:
            with report_threads_lock:
              report_threads.append(new_report_thread)

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
      density_label_en = {"輕微": "Light", "中等": "Moderate", "濃烈": "Heavy"}.get(
          current_density_level, current_density_level
      )
      cv2.putText(annotated_frame, f"Density: {density_label_en} ({current_density_score:.2f})", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if current_state == "TRIGGERED":
      alert_label = "[COMPOSITE CRITICAL]" if active_anomaly_type == "black_smoky+fire" else "[CRITICAL ALERT]"
      cv2.putText(annotated_frame, alert_label, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

    if ENABLE_PREPROCESS and SHOW_SIDE_BY_SIDE:
      raw_display = raw_frame.copy()
      cv2.putText(raw_display, "ORIGINAL (Comparison Only)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(annotated_frame, "PREPROCESSED (Detection Basis)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
      if raw_display.shape[0] != annotated_frame.shape[0]:
        raw_display = cv2.resize(raw_display, (annotated_frame.shape[1], annotated_frame.shape[0]))
      display_frame = cv2.hconcat([raw_display, annotated_frame])
    else:
      display_frame = annotated_frame

    # 只把畫面放進共用字典，不在這個執行緒裡呼叫任何 GUI 函式
    with frames_lock:
      latest_frames[camera_id] = display_frame

    # 依照這支影片本身的 FPS 控制節奏，不再依賴 cv2.waitKey 來計時
    elapsed = time.time() - loop_start
    sleep_time = frame_interval - elapsed
    if sleep_time > 0:
      time.sleep(sleep_time)

  cap.release()
  safe_print(camera_id, "[EXIT] 影像播放結束，釋放視窗資源。")
  with finished_lock:
    finished_cameras.add(camera_id)


def main():
  print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在載入 YOLOv11 模型...")
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

  # ------------------------------------------------------------------------
  # 【多視窗顯示問題修正】主執行緒統一負責所有視窗的顯示、按鍵偵測、關閉，
  # 這是 Windows 上唯一穩定可靠的多視窗顯示方式。
  # ------------------------------------------------------------------------
  active_windows = set()
  total_cameras = len(CAMERA_SOURCES)

  while True:
    with finished_lock:
      all_done = len(finished_cameras) >= total_cameras

    with frames_lock:
      current_frames = dict(latest_frames)

    for camera_id, frame in current_frames.items():
      window_name = f"NCKU Petrochemical Safety Monitor - {camera_id}"
      cv2.imshow(window_name, frame)
      active_windows.add(window_name)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [使用者中斷] 偵測到 q 鍵，停止所有攝影機串流...")
      stop_event.set()
      break

    if all_done:
      break

    time.sleep(0.01)

  for t in camera_threads:
    t.join()

  for window_name in active_windows:
    cv2.destroyWindow(window_name)
  cv2.destroyAllWindows()

  print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 所有攝影機串流已結束。")

  with report_threads_lock:
    pending_threads = [t for t in report_threads if t.is_alive()]

  if pending_threads:
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 還有 {len(pending_threads)} "
        "個報告生成任務仍在執行中，請稍候..."
    )
    # 逾時保護：每份報告含 LLM 生成 + BERTScore 評分，正常情況下不會超過幾分鐘，
    # 給每個執行緒最多 180 秒，超過就不再等待，避免真的遇到異常卡死時無法結束程式。
    for t in pending_threads:
      t.join(timeout=180)
    still_alive = [t for t in pending_threads if t.is_alive()]
    if still_alive:
      print(
          f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 有 {len(still_alive)} "
          "個報告生成任務逾時仍未結束，已不再等待（可能是 LLM 或 BERTScore 端異常，"
          "建議之後檢查 reports/verification_log.jsonl 確認該筆是否有正常寫入）。"
      )
    else:
      print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 所有報告生成任務已完成。")
  else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 沒有仍在執行中的報告生成任務。")


if __name__ == "__main__":
  main()
