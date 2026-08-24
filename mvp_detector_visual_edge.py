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

# ------------------------------------------------------------------------------
# 【功能 2：多攝影機】
# 這裡用字典模擬多路攝影機輸入，key 是你自己取的攝影機代號，value 是影片路徑。
# 目前你手上只有一支測試影片，所以 CAM-02 先指向同一支影片來驗證「多路同時處理」
# 的架構是否正常運作；之後如果你有第二支真實影片，把路徑換掉即可，不用改其他邏輯。
# ------------------------------------------------------------------------------
CAMERA_SOURCES = {
    "CAM-01": os.path.join(PROJECT_ROOT, "PixVerse_Clean_Slow.mp4"),
    # "CAM-02": os.path.join(PROJECT_ROOT, "另一支影片檔名.mp4"),  # 有第二支影片時取消註解
}

ALERT_THRESHOLD_SEC = 3.0
ALERT_COOLDOWN_SEC = 60.0

# ------------------------------------------------------------------------------
# 【功能 10：影片前處理】
# 開關與參數：alpha 控制對比度（1.0 = 不變），beta 控制亮度（0 = 不變）。
# 預設關閉，因為模型是用未經前處理的原始畫面訓練的，貿然開啟可能造成畫面風格
# 與訓練資料不一致。建議先用 ENABLE_PREPROCESS=False 跑一次當作對照組，
# 若你發現特定光線不佳的畫面誤判較多，再打開來看有沒有改善，兩者都測過最嚴謹。
# ------------------------------------------------------------------------------
ENABLE_PREPROCESS = False
PREPROCESS_ALPHA = 1.1   # 對比度倍率
PREPROCESS_BETA = 10     # 亮度偏移

# ------------------------------------------------------------------------------
# 【功能 8：反光防誤判】
# 判斷邏輯：反光通常是畫面裡一小塊區域「極度死白且集中」，跟真正的煙霧/火焰
# （亮度分布較大範圍、較不集中）不同。這是一個簡易啟發式規則，不是完美的
# 反光辨識模型，但能過濾掉一部分最明顯的反光誤判案例。
# ------------------------------------------------------------------------------
GLARE_BRIGHTNESS_THRESHOLD = 240   # 判定為「過曝」的像素亮度門檻（0~255）
GLARE_HOT_PIXEL_RATIO = 0.6        # 框內過曝像素佔比超過此值，視為疑似反光
GLARE_MAX_BOX_AREA_RATIO = 0.02    # 反光通常範圍小，框面積佔全畫面比例上限

# 多執行緒印出訊息時用的鎖，避免多支攝影機的 log 交錯錯亂
print_lock = threading.Lock()


def safe_print(camera_id, message):
  with print_lock:
    print(f"[{camera_id}] {message}")


def resolve_anomaly_type(detected_classes):
  """Layer 3 歸因邏輯：優先序判斷 + 複合風險偵測"""
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
  """
  【功能 8：濃淡分級】
  用偵測框面積佔全畫面的比例，當作煙霧/火焰「視覺上有多濃/多大」的簡易代理指標。
  框越大，通常代表煙霧擴散範圍越廣、或火焰規模越大。
  這不是真正量測煙霧濃度（需要專門的透光度或粒子感測器才能精確測量），
  是用既有 YOLO 框資訊做的簡易分級，建議在論文中誠實註明這個限制。
  """
  if box_area_ratio >= 0.15:
    return "濃烈"
  elif box_area_ratio >= 0.05:
    return "中等"
  else:
    return "輕微"


def is_likely_glare(frame, box_xyxy):
  """
  【功能 8：反光防誤判】
  裁切偵測框範圍，檢查是否符合「小範圍、高度集中過曝」的反光視覺特徵。
  回傳 True 代表這個框很可能是反光造成的誤判，應該從有效偵測中濾除。
  """
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


def process_camera_stream(camera_id, video_path, model):
  """
  【功能 2：多攝影機】
  把原本 main() 裡「處理單一影片來源」的完整邏輯包成一個函式，
  讓每一支攝影機都能各自獨立跑一份狀態機、各自互不干擾，
  只共用同一個 SQLite 資料庫（透過 camera_id 欄位區分是誰觸發的）。
  """
  safe_print(camera_id, f"[INIT] 正在初始化推論引擎...")
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

    # 【功能 10：影片前處理】（預設關閉，見上方 ENABLE_PREPROCESS 開關）
    if ENABLE_PREPROCESS:
      frame = cv2.convertScaleAbs(
          frame, alpha=PREPROCESS_ALPHA, beta=PREPROCESS_BETA
      )

    results = model(frame, device=0, verbose=False)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    detected_classes_all = [model.names[int(b.cls[0])] for b in boxes]
    has_chimney = "chimney" in detected_classes_all

    # 【功能 8：反光防誤判】逐一檢查每個框，濾掉疑似反光的偵測結果
    valid_classes = []
    anomaly_boxes = []  # 保留通過反光過濾的異常類別框，供濃淡分級使用
    for box in boxes:
      cls_name = model.names[int(box.cls[0])]
      if cls_name in ("fire", "black_smoky", "white_smoky"):
        if is_likely_glare(frame, box.xyxy[0].tolist()):
          continue  # 疑似反光，不計入這次判斷
        anomaly_boxes.append((cls_name, box.xyxy[0].tolist()))
      valid_classes.append(cls_name)

    current_frame_anomaly = resolve_anomaly_type(
        [c for c, _ in anomaly_boxes]
    )
    has_anomaly = current_frame_anomaly is not None

    # 【功能 8：濃淡分級】取當前異常類別中面積最大的框計算密度
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
        safe_print(
            camera_id,
            f"\r🟢 無異常，狀態機守衛中...",
        )

    elif current_state == "ACCUMULATING":
      if has_anomaly:
        active_anomaly_type = current_frame_anomaly
        duration = current_time - anomaly_start_time

        if duration >= ALERT_THRESHOLD_SEC:
          current_state = "TRIGGERED"

          # 【功能 1：遮蔽情境判斷】觸發當下若畫面中沒有偵測到 chimney 主體，
          # 標記為疑似遮蔽或非煙囪場景起火，不影響是否觸發告警（寧可錯報不漏報），
          # 只是額外附加這個情境標記，供後續人員判斷與論文分析使用。
          occlusion_flag = 1 if not has_chimney else 0
          context_note = (
              "（⚠️ 未偵測到煙囪主體，疑似廠區其他位置或視角遮蔽情境）"
              if occlusion_flag
              else ""
          )

          safe_print(
              camera_id,
              f"🔥 達到滿載臨界值！狀態跳轉: TRIGGERED "
              f"({active_anomaly_type}, 濃度:{current_density_level}) {context_note}",
          )

          timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
          safe_type_name = active_anomaly_type.replace("+", "_")
          snapshot_name = (
              f"snapshot_{camera_id}_{timestamp_str}_{safe_type_name}.jpg"
          )
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

    # UI 視覺化
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
      cv2.putText(annotated_frame, f"Density: {current_density_level}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if current_state == "TRIGGERED":
      alert_label = "[COMPOSITE CRITICAL]" if active_anomaly_type == "black_smoky+fire" else "[CRITICAL ALERT]"
      cv2.putText(annotated_frame, alert_label, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyWindow(window_name)
  safe_print(camera_id, "[EXIT] 影像播放結束，釋放視窗資源。")


def main():
  print(
      f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在載入 YOLOv11 模型"
      "（多攝影機共用同一份權重，不需要重複載入）..."
  )
  model = YOLO(MODEL_PATH)

  # 【功能 2：多攝影機】為 CAMERA_SOURCES 裡的每一路來源各開一個執行緒同時處理
  camera_threads = []
  for camera_id, video_path in CAMERA_SOURCES.items():
    t = threading.Thread(
        target=process_camera_stream,
        args=(camera_id, video_path, model),
        daemon=False,  # 這裡刻意不用 daemon，確保主程式會等待攝影機執行緒真正跑完
    )
    camera_threads.append(t)
    t.start()

  for t in camera_threads:
    t.join()

  cv2.destroyAllWindows()
  print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 所有攝影機串流已結束。")

  # 等待背景任務（報告生成、Discord 通知）完成，避免程式提前退出把任務中斷
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
