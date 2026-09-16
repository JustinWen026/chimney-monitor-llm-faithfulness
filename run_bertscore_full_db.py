"""
run_bertscore_full_db.py
------------------------
批次驅動腳本：對 safety_incidents.db 裡「全部」事件（普查，不是抽樣）
各跑一次 report_generator_bertscore_calibrated.py 既有的完整流程
（工具呼叫 + 欄位驗證 + 迭代重試 + BERTScore 計分）。

不改動 report_generator_bertscore_calibrated.py 本身任何一行邏輯，
純粹迴圈呼叫它既有的 _generate_report_async()，讓 bertscore_log.jsonl
累積到全部事件的分數。

單筆事件失敗（例如逾時、工具連線問題）不會中斷整個批次，會記錄下來
繼續跑下一筆，跑完印出成功/失敗統計，呼應你 multi_v2 那邊
n_runs_each_actual 的做法。

執行方式：
    conda activate smoke
    cd D:\\NCKU_paper_Capstone_Project
    python run_bertscore_full_db.py
"""

import asyncio
import sqlite3
import time

from db_manager import DB_PATH
import report_generator as rg


def fetch_all_incidents():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id, event_type FROM incidents ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    incidents = fetch_all_incidents()
    print(f"共 {len(incidents)} 筆事件（DB全量普查），開始逐筆跑 BERTScore 校準流程...\n")

    succeeded = []
    failed = []
    start_all = time.time()

    for i, (incident_id, event_type) in enumerate(incidents, start=1):
        print(f"[{i}/{len(incidents)}] incident_id={incident_id}, event_type={event_type}")
        start = time.time()
        try:
            asyncio.run(rg._generate_report_async(incident_id, event_type))
            succeeded.append(incident_id)
        except Exception as e:
            print(f"⚠️ incident_id={incident_id} 執行失敗：{e}")
            failed.append({"incident_id": incident_id, "event_type": event_type, "error": str(e)})
        print(f"   耗時 {time.time() - start:.1f} 秒\n")

    total_elapsed = time.time() - start_all
    print("=" * 70)
    print(f"批次完成。成功 {len(succeeded)}/{len(incidents)}，失敗 {len(failed)} 筆。")
    print(f"總耗時 {total_elapsed / 60:.1f} 分鐘。")
    if failed:
        print("失敗清單：")
        for f in failed:
            print(f"  incident_id={f['incident_id']} ({f['event_type']}): {f['error']}")
    print(f"\n完整分數已累積在 report_generator_bertscore_calibrated.py 產生的")
    print(f"bertscore_log.jsonl 裡（同一個 reports 資料夾），可以用")
    print(f"pandas.read_json('bertscore_log.jsonl', lines=True) 讀出來整理表格。")
    print("=" * 70)


if __name__ == "__main__":
    main()
