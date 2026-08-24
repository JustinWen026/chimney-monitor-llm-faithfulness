from datetime import datetime
import os
import sqlite3
from db_manager import DB_PATH, acknowledge_incident, init_db, get_open_critical_count, get_camera_list
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="石化園區工業煙囪 AI 工安告警系統",
    page_icon="🏭",
    layout="wide",
)

st_autorefresh(interval=5000, key="dashboard_autorefresh")

init_db()

st.title("🏭 石化園區工業煙囪 AI 影像安全監控與閉環管理平台")
st.caption(
    "NCKU CSIE AI Safety Monitor System • Multi-Camera Edge-Based"
    " Real-time Incident Audit Trail"
)

ESCALATION_MINUTES = 5
open_critical_count = get_open_critical_count(minutes_threshold=ESCALATION_MINUTES)
if open_critical_count > 0:
  st.markdown(
      f"""
      <div style="
          background: linear-gradient(90deg, #7a0000, #c40000, #7a0000);
          background-size: 200% 100%;
          animation: pulse-alert 1.5s ease-in-out infinite;
          border: 2px solid #ff4b4b;
          border-radius: 8px;
          padding: 18px 24px;
          margin-bottom: 12px;
          text-align: center;
      ">
          <span style="font-size: 22px; font-weight: 700; color: white;">
              🆘 緊急升級警示：{open_critical_count} 件 CRITICAL 事故已超過
              {ESCALATION_MINUTES} 分鐘尚未簽收處置，請立即確認！
          </span>
      </div>
      <style>
      @keyframes pulse-alert {{
          0% {{ background-position: 0% 50%; }}
          50% {{ background-position: 100% 50%; }}
          100% {{ background-position: 0% 50%; }}
      }}
      </style>
      """,
      unsafe_allow_html=True,
  )

st.markdown("---")


def load_data():
  with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", conn)
  return df


df = load_data()

# ==============================================================================
# 📊 工安 KPI 關鍵指標卡片（v5：遮蔽情境卡片加上 help 提示說明）
# ==============================================================================
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

total_incidents = len(df)
pending_count = len(df[df["status"] == "PENDING"]) if not df.empty else 0
resolved_df = df[df["status"] == "RESOLVED"] if not df.empty else pd.DataFrame()
avg_mttr = (
    round(resolved_df["mttr_seconds"].mean(), 1)
    if not resolved_df.empty and resolved_df["mttr_seconds"].mean() > 0
    else 0.0
)
fire_involved_count = (
    len(df[df["event_type"].str.contains("fire", na=False)]) if not df.empty else 0
)
composite_count = (
    len(df[df["event_type"] == "black_smoky+fire"]) if not df.empty else 0
)
occlusion_count = (
    len(df[df["occlusion_flag"] == 1]) if not df.empty and "occlusion_flag" in df.columns else 0
)
camera_count = df["camera_id"].nunique() if not df.empty and "camera_id" in df.columns else 0

col1.metric("🚨 總告警事件數", f"{total_incidents} 次")
col2.metric("⏳ 待簽收處置", f"{pending_count} 件", delta_color="inverse")
col3.metric("⏱️ 平均處置時間", f"{avg_mttr} 秒")
col4.metric("🔥 火焰相關事件", f"{fire_involved_count} 次")
col5.metric("🆘 複合型高風險", f"{composite_count} 次", delta_color="inverse")
col6.metric(
    "👁️ 需確認位置事件",
    f"{occlusion_count} 次",
    help=(
        "觸發告警當下，畫面中未偵測到煙囪主體的事件數。"
        "可能代表異常發生於煙囪以外的廠區位置，或攝影機視角被遮蔽物擋住，"
        "建議這類事件優先派人現場確認實際起火/排煙位置。"
    ),
)
col7.metric("📷 監控攝影機路數", f"{camera_count} 路")

st.markdown("---")

st.subheader("📈 事故趨勢總覽（近 14 日）")
if df.empty:
  st.info("目前尚無歷史事件資料，無法繪製趨勢圖。")
else:
  trend_df = df.copy()
  trend_df["date"] = pd.to_datetime(trend_df["timestamp"]).dt.date
  pivot_df = trend_df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
  pivot_df = pivot_df.tail(14)
  st.bar_chart(pivot_df)

st.markdown("---")

st.subheader("🖼️ 最新告警現場縮圖")
recent_df = df.head(3) if not df.empty else pd.DataFrame()
if recent_df.empty:
  st.info("目前尚無告警紀錄。")
else:
  thumb_cols = st.columns(len(recent_df))
  for idx, (_, row) in enumerate(recent_df.iterrows()):
    with thumb_cols[idx]:
      severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡"}.get(row["severity"], "⚪")
      if row["snapshot_path"] and os.path.exists(row["snapshot_path"]):
        st.image(Image.open(row["snapshot_path"]), use_container_width=True)
      else:
        st.warning("無截圖")
      occlusion_tag = " ⚠️需確認位置" if row.get("occlusion_flag") == 1 else ""
      st.caption(
          f"{severity_emoji} #{row['id']} | {row.get('camera_id', 'CAM-01')} |"
          f" {row['event_type']}{occlusion_tag} | {row['timestamp'][-8:]}"
      )

st.markdown("---")

st.sidebar.header("🔍 事故篩選與查詢")
camera_options = ["全部"] + get_camera_list()
camera_filter = st.sidebar.selectbox("📷 攝影機來源", camera_options)
status_filter = st.sidebar.selectbox("處理狀態", ["全部", "PENDING", "RESOLVED"])
severity_filter = st.sidebar.selectbox("嚴重度等級", ["全部", "CRITICAL", "HIGH", "WARNING"])
event_type_options = ["全部"] + (sorted(df["event_type"].unique().tolist()) if not df.empty else [])
event_filter = st.sidebar.selectbox("事件類型", event_type_options)

filtered_df = df.copy()
if not filtered_df.empty:
  if camera_filter != "全部":
    filtered_df = filtered_df[filtered_df["camera_id"] == camera_filter]
  if status_filter != "全部":
    filtered_df = filtered_df[filtered_df["status"] == status_filter]
  if severity_filter != "全部":
    filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
  if event_filter != "全部":
    filtered_df = filtered_df[filtered_df["event_type"] == event_filter]

tab1, tab2 = st.tabs(["📋 事故審計日誌與閉環處置", "📸 現場存證截圖調閱"])

with tab1:
  st.subheader("📑 歷史工安告警總覽")

  if filtered_df.empty:
    st.info("目前無符合條件的工安事件紀錄。")
  else:
    display_cols = [
        "id", "timestamp", "camera_id", "event_type", "severity", "risk_level",
        "density_level", "occlusion_flag", "duration", "status", "ack_time",
        "mttr_seconds", "operator_notes",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[display_cols].copy()
    if "occlusion_flag" in display_df.columns:
      display_df["occlusion_flag"] = display_df["occlusion_flag"].map(
          {1: "⚠️ 需確認位置", 0: "✅ 正常"}
      )

    def highlight_severity(row):
      color_map = {
          "CRITICAL": "background-color: #ff4b4b; color: white",
          "HIGH": "background-color: #ff9d45; color: black",
          "WARNING": "background-color: #ffe08a; color: black",
      }
      style = color_map.get(row["severity"], "")
      return [style if col == "severity" else "" for col in row.index]

    styled_df = display_df.style.apply(highlight_severity, axis=1)
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=280,
        column_config={
            "occlusion_flag": st.column_config.TextColumn(
                "位置確認",
                help=(
                    "「⚠️ 需確認位置」代表觸發當下畫面中沒看到煙囪主體，"
                    "可能是廠區其他位置起火，或攝影機視角被遮蔽物擋住，"
                    "建議優先派人現場確認實際位置。"
                ),
            ),
        } if "occlusion_flag" in display_cols else None,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 一鍵匯出工安日誌報表 (CSV)",
        data=csv_data,
        file_name=f'safety_incident_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime="text/csv",
    )

  st.markdown("### ✍️ 工安主管閉環簽收處置 (Closed-Loop Action)")
  pending_items = df[df["status"] == "PENDING"]["id"].tolist() if not df.empty else []

  if pending_items:
    c1, c2, c3 = st.columns([2, 4, 2])
    with c1:
      selected_id = st.selectbox("選擇待處理事件 ID", pending_items, key="ack_select")
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
          st.success(f"事件 #{selected_id} 已成功閉環簽收！MTTR 已自動計算記錄。")
          st.rerun()
  else:
    st.success("🎉 目前所有工安告警皆已處理完畢，無待簽收項目！")

with tab2:
  st.subheader("📸 告警觸發瞬間 - 影像截圖存證調閱")
  if not filtered_df.empty:
    img_select_id = st.selectbox("選擇要檢視截圖的事件 ID", filtered_df["id"].tolist())
    selected_row = filtered_df[filtered_df["id"] == img_select_id].iloc[0]
    img_path = selected_row["snapshot_path"]

    if img_path and os.path.exists(img_path):
      occlusion_note = (
          "⚠️ 需確認位置（未偵測到煙囪主體） | "
          if selected_row.get("occlusion_flag") == 1
          else ""
      )
      st.image(
          Image.open(img_path),
          caption=(
              f"事件 #{img_select_id} | {selected_row.get('camera_id', 'CAM-01')} |"
              f" 類型: {selected_row['event_type']} | {occlusion_note}"
              f"時間: {selected_row['timestamp']}"
          ),
          use_container_width=True,
      )
    else:
      st.warning("⚠️ 該事件未找到對應的截圖檔案。")
  else:
    st.info("無事件可供預覽截圖。")
