from datetime import datetime
import os
import sqlite3
from db_manager import DB_PATH, acknowledge_incident, init_db
import pandas as pd
from PIL import Image
import streamlit as st

# 設定網頁標題與寬版版面
st.set_page_config(
    page_title="石化園區工業煙囪 AI 工安告警系統",
    page_icon="🏭",
    layout="wide",
)

init_db()

st.title("🏭 石化園區工業煙囪 AI 影像安全監控與閉環管理平台")
st.caption(
    "NCKU CSIE AI Safety Monitor System • Edge-Based Real-time Incident Audit"
    " Trail"
)
st.markdown("---")


# 讀取資料庫
def load_data():
  with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", conn)
  return df


df = load_data()

# ==============================================================================
# 📊 功能 6：工安 KPI 關鍵指標卡片
# ==============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

total_incidents = len(df)
pending_count = len(df[df["status"] == "PENDING"]) if not df.empty else 0
resolved_df = df[df["status"] == "RESOLVED"] if not df.empty else pd.DataFrame()
avg_mttr = (
    round(resolved_df["mttr_seconds"].mean(), 1)
    if not resolved_df.empty and resolved_df["mttr_seconds"].mean() > 0
    else 0.0
)
fire_count = len(df[df["event_type"] == "fire"]) if not df.empty else 0
smoke_count = (
    len(df[df["event_type"].isin(["black_smoky", "white_smoky"])])
    if not df.empty
    else 0
)

col1.metric("🚨 總告警事件數", f"{total_incidents} 次")
col2.metric("⏳ 待簽收處置 (Pending)", f"{pending_count} 件", delta_color="inverse")
col3.metric("⏱️ 平均處置時間 (Mean MTTR)", f"{avg_mttr} 秒")
col4.metric("🔥 火焰/火炬事件", f"{fire_count} 次")
col5.metric("💨 排煙外溢事件", f"{smoke_count} 次")

st.markdown("---")

# ==============================================================================
# 🔍 功能 7：側邊欄篩選器 (Filter Sidebar)
# ==============================================================================
st.sidebar.header("🔍 事故篩選與查詢")
status_filter = st.sidebar.selectbox("處理狀態", ["全部", "PENDING", "RESOLVED"])
severity_filter = st.sidebar.selectbox(
    "嚴重度等級", ["全部", "CRITICAL", "HIGH", "WARNING"]
)
event_filter = st.sidebar.selectbox(
    "事件類型", ["全部", "fire", "black_smoky", "white_smoky"]
)

filtered_df = df.copy()
if not filtered_df.empty:
  if status_filter != "全部":
    filtered_df = filtered_df[filtered_df["status"] == status_filter]
  if severity_filter != "全部":
    filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
  if event_filter != "全部":
    filtered_df = filtered_df[filtered_df["event_type"] == event_filter]

# ==============================================================================
# 📋 主畫面：工安日誌表格 + 功能 8 色彩標記 + 功能 4 匯出 CSV
# ==============================================================================
tab1, tab2 = st.tabs(["📋 事故審計日誌與閉環處置", "📸 現場存證截圖調閱"])

with tab1:
  st.subheader("📑 歷史工安告警總覽")

  if filtered_df.empty:
    st.info("目前無符合條件的工安事件紀錄。")
  else:
    # 格式化顯示表格
    display_df = filtered_df[[
        "id",
        "timestamp",
        "event_type",
        "severity",
        "duration",
        "status",
        "ack_time",
        "mttr_seconds",
        "operator_notes",
    ]].copy()
    st.dataframe(display_df, use_container_width=True, height=280)

    # 功能 4：一鍵匯出日誌報表 (CSV)
    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 一鍵匯出工安日誌報表 (CSV)",
        data=csv_data,
        file_name=f'safety_incident_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime="text/csv",
    )

  # ==============================================================================
  # 🛠️ 功能 5：閉環管理處置簽收區塊 (MTTR 更新)
  # ==============================================================================
  st.markdown("### ✍️ 工安主管閉環簽收處置 (Closed-Loop Action)")
  pending_items = (
      df[df["status"] == "PENDING"]["id"].tolist() if not df.empty else []
  )

  if pending_items:
    c1, c2, c3 = st.columns([2, 4, 2])
    with c1:
      selected_id = st.selectbox(
          "選擇待處理事件 ID", pending_items, key="ack_select"
      )
    with c2:
      notes_input = st.text_input(
          "處置說明 / 排查備註",
          value="現場管線閥門已檢修，排煙恢復正常",
          key="ack_notes",
      )
    with c3:
      st.write("")
      st.write("")
      if st.button("✅ 確認簽收並解除警報", use_container_width=True):
        if acknowledge_incident(selected_id, notes_input):
          st.success(
              f"事件 #{selected_id} 已成功閉環簽收！MTTR 已自動計算記錄。"
          )
          st.rerun()
  else:
    st.success("🎉 目前所有工安告警皆已處理完畢，無待簽收項目！")

with tab2:
  st.subheader("📸 告警觸發瞬間 - 影像截圖存證調閱")
  if not filtered_df.empty:
    img_select_id = st.selectbox(
        "選擇要檢視截圖的事件 ID", filtered_df["id"].tolist()
    )
    selected_row = filtered_df[filtered_df["id"] == img_select_id].iloc[0]
    img_path = selected_row["snapshot_path"]

    if img_path and os.path.exists(img_path):
      st.image(
          Image.open(img_path),
          caption=(
              f"事件 #{img_select_id} | 類型: {selected_row['event_type']} |"
              f" 時間: {selected_row['timestamp']}"
          ),
          use_container_width=True,
      )
    else:
      st.warning("⚠️ 該事件未找到對應的截圖檔案。")
  else:
    st.info("無事件可供預覽截圖。")