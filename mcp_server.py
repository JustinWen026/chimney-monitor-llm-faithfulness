import json
import sqlite3
from mcp.server.fastmcp import FastMCP

from db_manager import DB_PATH

mcp = FastMCP("ChimneyMonitor")


@mcp.tool()
def get_incident_by_id(incident_id: int) -> str:
    """
    依照指定的 incident_id，精確查詢該筆工安異常事件的結構化資料。

    【v3 重要修正】舊版的 get_latest_incident_log 永遠回傳「資料庫裡最新一筆」，
    在多支攝影機幾乎同時觸發告警、且報告生成有互斥鎖排隊等待的情況下，
    會導致某個事件的報告內容實際上抓到的是「另一個更晚寫入的事件」的資料——
    也就是報告內容跟它自己標題上的事件編號對不起來。
    這支工具改成依 incident_id 精確查詢，從根本避開這種競態條件（race condition）。
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cursor.fetchone()

    if row is None:
        return json.dumps(
            {"error": f"找不到 incident_id={incident_id} 的事件紀錄"}, ensure_ascii=False
        )

    incident_data = {
        "incident_id": row["id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "event_type": row["event_type"],
        "severity": row["severity"],
        "risk_level": row["risk_level"],
        "duration_seconds": row["duration"],
        "status": row["status"],
    }
    return json.dumps(incident_data, ensure_ascii=False)


@mcp.tool()
def get_latest_incident_log() -> str:
    """
    【已不建議使用，僅保留向下相容】獲取資料庫中最新一筆工安事件資料。
    在有多支攝影機並發觸發的情境下，這支工具查到的「最新」可能不是
    你原本要問的那一筆事件，請優先使用 get_incident_by_id。
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()

    if row is None:
        return json.dumps(
            {"error": "目前資料庫尚無任何工安事件紀錄"}, ensure_ascii=False
        )

    incident_data = {
        "incident_id": row["id"],
        "timestamp": row["timestamp"],
        "camera_id": row["camera_id"],
        "event_type": row["event_type"],
        "severity": row["severity"],
        "risk_level": row["risk_level"],
        "duration_seconds": row["duration"],
        "status": row["status"],
    }
    return json.dumps(incident_data, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
