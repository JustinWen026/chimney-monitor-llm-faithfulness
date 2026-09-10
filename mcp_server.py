import json
import sqlite3
from mcp.server.fastmcp import FastMCP

# 【v2 修正】原本 mcp_server.py 寫的是 `from db_manager_v2 import DB_PATH`，
# 但專案裡實際存在的是 db_manager_v3.py，db_manager_v2.py 根本不存在。
# 這會導致 MCP Server 啟動時直接 ModuleNotFoundError 崩潰，
# report_generator 呼叫時只會看到 stdio 連線失敗，很難第一時間看出真正原因。
from db_manager import DB_PATH

mcp = FastMCP("ChimneyMonitor")


@mcp.tool()
def get_latest_incident_log() -> str:
    """
    獲取石化園區煙囪最新一筆工安異常事件的結構化資料。
    直接從真實運作中的 SQLite 工安資料庫讀取最新一筆紀錄。

    【v2 修正】原版只回傳 incident_id / timestamp / event_type / severity /
    risk_level / duration_seconds / status，缺少 camera_id。
    這支專案已經是多攝影機架構，少了 camera_id 會導致下游的
    「工具呼叫忠實度驗證」機制沒有足夠欄位可以比對，因此這裡補上。
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
