from datetime import datetime
import threading
import requests

# =====================================================================
# Discord Webhook 設定區 (完全解耦，方便隨時修改 Webhook 網址)[cite: 7]
# =====================================================================
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"  #[cite: 7]


def _send_discord_worker(
    timestamp, duration, event_type="煙霧外溢 (Smoky Event)"
):  #[cite: 7]
  """實際在背景執行緒中運行的 Discord 發送工作[cite: 7]"""
  if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":  #[cite: 7]
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [Discord] 尚未設定"
        " Webhook URL，跳過發送。"
    )  #[cite: 7]
    return

  # 事件中文名稱映射
  event_map = {
      "fire": "🔥 煙囪起火燃燒 (Chimney Flare/Fire)",
      "black_smoky": "💨 嚴重黑煙排放 (Black Smoke Emission)",
      "white_smoky": "🌫️ 白煙外溢異常 (White Smoke Detected)",
  }
  event_display = event_map.get(event_type, event_type)

  # 專為成大資工專題與海報視覺美化設計的卡片排版[cite: 7]
  payload = {
      "username": "石化園區 AI 偵測系統",  #[cite: 7]
      "avatar_url": "https://i.imgur.com/8nS47H3.png",  # 機器人預設頭像[cite: 7]
      "embeds": [{
          "title": "🚨 系統安全核心告警：偵測到煙囪異常事件 🚨",
          "description": (  #[cite: 7]
              "系統偵測到廠區煙囪異常持續時間已達臨界值，已自動歸因並觸發主動防護通報機制。"
          ),
          "color": 15158332,  # 鮮豔紅色邊條[cite: 7]
          "fields": [
              {
                  "name": "📌 專案名稱",
                  "value": (
                      "AI在石化園區工業煙囪煙霧安全偵測與其在事故歸因分析之研究"
                  ),
                  "inline": False,
              },  #[cite: 7]
              {"name": "⚠️ 異常事件", "value": event_display, "inline": True},
              {
                  "name": "⏱ 累積秒數",
                  "value": f"`{duration} 秒` (門檻: 3.0秒)",
                  "inline": True,
              },  #[cite: 7]
              {"name": "⏰ 觸發時間", "value": timestamp, "inline": False},  #[cite: 7]
              {
                  "name": "🔴 安全狀態",
                  "value": "```diff\n- TRIGGERED\n```",
                  "inline": False,
              },  #[cite: 7]
          ],
          "footer": {"text": "NCKU CSIE AI Safety Monitor System • 2026"},  #[cite: 7]
      }],
  }

  try:
    response = requests.post(
        DISCORD_WEBHOOK_URL, json=payload, timeout=5
    )  #[cite: 7]
    if response.status_code in [200, 204]:  #[cite: 7]
      print(
          f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 [Discord成功]"
          " 告警卡片已送達 Discord 頻道！"
      )  #[cite: 7]
    else:
      print(
          f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord失敗] HTTP"
          f" 錯誤代碼: {response.status_code}"
      )  #[cite: 7]
  except Exception as e:  #[cite: 7]
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord錯誤] 連線失敗:"
        f" {e}"
    )  #[cite: 7]


def send_discord_alert_async(
    timestamp, duration, event_type="smoke"
):  #[cite: 7]
  """外部接口：採用非阻塞執行緒發送，確保主推論迴圈維持極高 FPS[cite: 7]"""
  thread = threading.Thread(
      target=_send_discord_worker,
      args=(timestamp, duration, event_type),
      daemon=True,
  )  #[cite: 7]
  thread.start()  #[cite: 7]