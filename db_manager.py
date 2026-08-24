from datetime import datetime
import os
import sqlite3

DB_PATH = r"D:\NCKU_paper_Capstone_Project\safety_incidents.db"


def init_db():
  """初始化 SQLite 資料庫與事件表（v3：新增 camera_id / density_level / occlusion_flag）"""
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT DEFAULT 'CAM-01',
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                risk_level INTEGER DEFAULT 1,
                density_level TEXT,
                occlusion_flag INTEGER DEFAULT 0,
                duration REAL NOT NULL,
                snapshot_path TEXT,
                status TEXT DEFAULT 'PENDING',
                ack_time TEXT,
                mttr_seconds REAL DEFAULT 0.0,
                operator_notes TEXT DEFAULT ''
            )
        """)
    conn.commit()

    # 舊資料庫遷移：逐一嘗試新增欄位，欄位已存在就忽略錯誤
    for column_def in [
        "camera_id TEXT DEFAULT 'CAM-01'",
        "risk_level INTEGER DEFAULT 1",
        "density_level TEXT",
        "occlusion_flag INTEGER DEFAULT 0",
    ]:
      try:
        cursor.execute(f"ALTER TABLE incidents ADD COLUMN {column_def}")
        conn.commit()
      except sqlite3.OperationalError:
        pass


def determine_severity_and_risk(event_type: str):
  """Layer 3 歸因邏輯：根據事件類型判斷嚴重度與風險等級"""
  if "fire" in event_type and "black_smoky" in event_type:
    return "CRITICAL", 5
  elif event_type == "fire":
    return "CRITICAL", 4
  elif event_type == "black_smoky":
    return "HIGH", 3
  elif event_type == "white_smoky":
    return "WARNING", 2
  else:
    return "WARNING", 1


def log_incident(
    event_type,
    duration,
    snapshot_path,
    camera_id="CAM-01",
    density_level=None,
    occlusion_flag=0,
):
  """記錄新觸發的工安異常事件（v3：含攝影機來源、濃淡等級、遮蔽情境旗標）"""
  init_db()

  severity, risk_level = determine_severity_and_risk(event_type)
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO incidents 
            (timestamp, camera_id, event_type, severity, risk_level, 
             density_level, occlusion_flag, duration, snapshot_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
        """,
        (
            timestamp, camera_id, event_type, severity, risk_level,
            density_level, occlusion_flag, duration, snapshot_path,
        ),
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


def get_camera_list():
  """取得資料庫中所有出現過的攝影機 ID 清單，供 Dashboard 篩選器使用"""
  init_db()
  with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT camera_id FROM incidents ORDER BY camera_id"
    )
    return [row[0] for row in cursor.fetchall()]
