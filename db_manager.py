from datetime import datetime
import os
import sqlite3

DB_PATH = r"D:\NCKU_paper_Capstone_Project\safety_incidents.db"


def init_db():
  """初始化 SQLite 資料庫與事件表（含 risk_level 欄位遷移）"""
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                risk_level INTEGER DEFAULT 1,
                duration REAL NOT NULL,
                snapshot_path TEXT,
                status TEXT DEFAULT 'PENDING',
                ack_time TEXT,
                mttr_seconds REAL DEFAULT 0.0,
                operator_notes TEXT DEFAULT ''
            )
        """)
    conn.commit()

    # 若資料庫是舊版（沒有 risk_level 欄位），這裡做安全遷移
    try:
      cursor.execute(
          "ALTER TABLE incidents ADD COLUMN risk_level INTEGER DEFAULT 1"
      )
      conn.commit()
    except sqlite3.OperationalError:
      # 欄位已存在，忽略即可
      pass


def determine_severity_and_risk(event_type: str):
  """
  Layer 3 歸因邏輯（v2）：根據事件類型判斷嚴重度與風險等級。

  複合事件（black_smoky + fire 同時觸發）優先序最高，
  對應前一階段討論的 RiskLevel 5（複合型高風險：疑似火災或爆燃）。
  event_type 若為複合事件，字串格式為 "black_smoky+fire"。
  """
  if "fire" in event_type and "black_smoky" in event_type:
    return "CRITICAL", 5  # 複合風險：疑似火災或爆燃
  elif event_type == "fire":
    return "CRITICAL", 4  # 火炬異常燃燒 / 火災前兆
  elif event_type == "black_smoky":
    return "HIGH", 3       # 燃燒控制失效
  elif event_type == "white_smoky":
    return "WARNING", 2    # 疑似製程洩漏
  else:
    return "WARNING", 1


def log_incident(event_type, duration, snapshot_path):
  """記錄新觸發的工安異常事件（v2：含 risk_level）"""
  init_db()

  severity, risk_level = determine_severity_and_risk(event_type)
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO incidents 
            (timestamp, event_type, severity, risk_level, duration, snapshot_path, status)
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
        """,
        (timestamp, event_type, severity, risk_level, duration, snapshot_path),
    )
    conn.commit()
    return cursor.lastrowid


def acknowledge_incident(incident_id, notes="已現場排查處理完畢"):
  """工安閉環簽收：計算 MTTR 並更新狀態為 RESOLVED"""
  ack_now = datetime.now()
  ack_time_str = ack_now.strftime("%Y-%m-%d %H:%M:%S")

  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp FROM incidents WHERE id = ?", (incident_id,)
    )
    row = cursor.fetchone()
    if row:
      trigger_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
      mttr_seconds = round((ack_now - trigger_time).total_seconds(), 1)

      cursor.execute(
          """
                UPDATE incidents 
                SET status = 'RESOLVED', ack_time = ?, mttr_seconds = ?, operator_notes = ?
                WHERE id = ?
            """,
          (ack_time_str, mttr_seconds, notes, incident_id),
      )
      conn.commit()
      return True
  return False


def get_all_incidents():
  """取得所有事故紀錄（回傳 tuple 清單）"""
  init_db()
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents ORDER BY id DESC")
    return cursor.fetchall()


def get_open_critical_count(minutes_threshold=5):
  """取得逾時未處置的 CRITICAL 事件數量，供前端升級告警橫幅使用"""
  init_db()
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM incidents
        WHERE status = 'PENDING' 
          AND severity = 'CRITICAL'
          AND (julianday('now', 'localtime') - julianday(timestamp)) * 1440 > ?
        """,
        (minutes_threshold,),
    )
    return cursor.fetchone()[0]
