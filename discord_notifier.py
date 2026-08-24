from datetime import datetime
import os
import threading
import requests

# =====================================================================
# Discord Webhook 設定區（環境變數讀取，不硬編碼在程式碼裡）
# =====================================================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def _send_discord_worker(
    timestamp, duration, event_type="煙霧外溢 (Smoky Event)",
    camera_id="CAM-01", density_level=None, occlusion_flag=0,
):
  """實際在背景執行緒中運行的 Discord 發送工作（v3：新增攝影機來源標示）"""
  if not DISCORD_WEBHOOK_URL:
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [Discord] 尚未設定"
        " DISCORD_WEBHOOK_URL 環境變數，跳過發送。"
    )
    return

  event_map = {
      "black_smoky+fire": "🆘 複合型高風險事故：疑似火災或爆燃 (Composite Critical Event)",
      "fire": "🔥 煙囪起火燃燒 (Chimney Flare/Fire)",
      "black_smoky": "💨 嚴重黑煙排放 (Black Smoke Emission)",
      "white_smoky": "🌫️ 白煙外溢異常 (White Smoke Detected)",
  }
  event_display = event_map.get(event_type, event_type)
  is_composite = "black_smoky" in event_type and "fire" in event_type

  extra_fields = [
      {"name": "📷 攝影機來源", "value": camera_id, "inline": True},
  ]
  if density_level:
    extra_fields.append(
        {"name": "🌫️ 濃度等級", "value": density_level, "inline": True}
    )
  if occlusion_flag:
    extra_fields.append({
        "name": "⚠️ 特別提醒",
        "value": "本次事件觸發時未偵測到煙囪主體，可能為廠區其他位置起火或視角遮蔽情境，建議加派人力現場確認。",
        "inline": False,
    })

  payload = {
      "username": "石化園區 AI 偵測系統",
      "avatar_url": "https://i.imgur.com/8nS47H3.png",
      "embeds": [{
          "title": (
              "🆘 系統安全核心告警：偵測到複合型高風險事故 🆘"
              if is_composite
              else "🚨 系統安全核心告警：偵測到煙囪異常事件 🚨"
          ),
          "description": (
              "系統偵測到廠區異常持續時間已達臨界值，已自動歸因並觸發主動防護通報機制。"
          ),
          "color": 10038562 if is_composite else 15158332,
          "fields": [
              {
                  "name": "📌 專案名稱",
                  "value": (
                      "AI在石化園區工業煙囪煙霧安全偵測與其在事故歸因分析之研究"
                  ),
                  "inline": False,
              },
              {"name": "⚠️ 異常事件", "value": event_display, "inline": True},
              {
                  "name": "⏱ 累積秒數",
                  "value": f"`{duration} 秒` (門檻: 3.0秒)",
                  "inline": True,
              },
              *extra_fields,
              {"name": "⏰ 觸發時間", "value": timestamp, "inline": False},
              {
                  "name": "🔴 安全狀態",
                  "value": "```diff\n- TRIGGERED\n```",
                  "inline": False,
              },
          ],
          "footer": {"text": "NCKU CSIE AI Safety Monitor System • 2026"},
      }],
  }

  try:
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    if response.status_code in [200, 204]:
      print(
          f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 [Discord成功]"
          f" ({camera_id}) 告警卡片已送達 Discord 頻道！"
      )
    else:
      print(
          f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord失敗] HTTP"
          f" 錯誤代碼: {response.status_code}"
      )
  except Exception as e:
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord錯誤] 連線失敗:"
        f" {e}"
    )


def send_discord_alert_async(
    timestamp, duration, event_type="smoke",
    camera_id="CAM-01", density_level=None, occlusion_flag=0,
):
  """外部接口：採用非阻塞執行緒發送，確保主推論迴圈維持極高 FPS"""
  thread = threading.Thread(
      target=_send_discord_worker,
      args=(timestamp, duration, event_type, camera_id, density_level, occlusion_flag),
      daemon=True,
  )
  thread.start()
