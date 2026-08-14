from datetime import datetime
import os
import sqlite3

DB_PATH = r"D:\NCKU_paper_Capstone_Project\safety_incidents.db"


def init_db():
  """初始化 SQLite 資料庫與事件表"""
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                duration REAL NOT NULL,
                snapshot_path TEXT,
                status TEXT DEFAULT 'PENDING',
                ack_time TEXT,
                mttr_seconds REAL DEFAULT 0.0,
                operator_notes TEXT DEFAULT ''
            )
        """)
    conn.commit()


def log_incident(event_type, duration, snapshot_path):
  """記錄新觸發的工安異常事件"""
  init_db()

  # 嚴重度等級標定
  if event_type == "fire":
    severity = "CRITICAL"
  elif event_type == "black_smoky":
    severity = "HIGH"
  else:
    severity = "WARNING"

  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO incidents (timestamp, event_type, severity, duration, snapshot_path, status)
            VALUES (?, ?, ?, ?, ?, 'PENDING')
        """,
        (timestamp, event_type, severity, duration, snapshot_path),
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