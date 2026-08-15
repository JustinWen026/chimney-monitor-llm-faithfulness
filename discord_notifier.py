from datetime import datetime
import os
import threading
import requests

# =====================================================================
# Discord Webhook 設定區（v2：改為環境變數讀取，不再硬編碼在程式碼裡）
#
# 使用方式（Windows PowerShell，設定一次即可，之後每次開終端機都會自動生效）：
#   [System.Environment]::SetEnvironmentVariable(
#       "DISCORD_WEBHOOK_URL", "你的webhook網址", "User")
#
# 或者只在當前這個 PowerShell 視窗暫時生效（每次重開視窗要重設）：
#   $env:DISCORD_WEBHOOK_URL = "你的webhook網址"
#
# ⚠️ 提醒：舊版 discord_notifier.py 裡的 Webhook 網址已經外流在對話紀錄中，
# 請務必到 Discord 頻道設定 → 整合 → Webhook，刪除舊的並重新產生一組新的。
# =====================================================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def _send_discord_worker(
    timestamp, duration, event_type="煙霧外溢 (Smoky Event)"
):
  """實際在背景執行緒中運行的 Discord 發送工作"""
  if not DISCORD_WEBHOOK_URL:
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [Discord] 尚未設定"
        " DISCORD_WEBHOOK_URL 環境變數，跳過發送。"
    )
    return

  # 事件中文名稱映射（v2：新增複合事件）
  event_map = {
      "black_smoky+fire": "🆘 複合型高風險事故：疑似火災或爆燃 (Composite Critical Event)",
      "fire": "🔥 煙囪起火燃燒 (Chimney Flare/Fire)",
      "black_smoky": "💨 嚴重黑煙排放 (Black Smoke Emission)",
      "white_smoky": "🌫️ 白煙外溢異常 (White Smoke Detected)",
  }
  event_display = event_map.get(event_type, event_type)
  is_composite = "black_smoky" in event_type and "fire" in event_type

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
              "系統偵測到廠區煙囪異常持續時間已達臨界值，已自動歸因並觸發主動防護通報機制。"
          ),
          "color": 10038562 if is_composite else 15158332,  # 複合事件用更深的暗紅色區隔
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
          " 告警卡片已送達 Discord 頻道！"
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


def send_discord_alert_async(timestamp, duration, event_type="smoke"):
  """外部接口：採用非阻塞執行緒發送，確保主推論迴圈維持極高 FPS"""
  thread = threading.Thread(
      target=_send_discord_worker,
      args=(timestamp, duration, event_type),
      daemon=True,
  )
  thread.start()
