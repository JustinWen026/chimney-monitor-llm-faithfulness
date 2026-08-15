import json
import sqlite3
from mcp.server.fastmcp import FastMCP
from db_manager import DB_PATH

mcp = FastMCP("ChimneyMonitor")


@mcp.tool()
def get_latest_incident_log() -> str:
    """
    獲取石化園區煙囪最新一筆工安異常事件的結構化資料。
    直接從真實運作中的 SQLite 工安資料庫讀取最新一筆紀錄，
    不是模擬數據（舊版 mcp_server.py 用的是寫死的假資料，這版已修正）。
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
        "event_type": row["event_type"],
        "severity": row["severity"],
        "risk_level": row["risk_level"],
        "duration_seconds": row["duration"],
        "status": row["status"],
    }
    return json.dumps(incident_data, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
