
import os
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

# ========= Config =========
load_dotenv()  # loads .env if present

SF_CFG = dict(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    database=os.getenv("SNOWFLAKE_DATABASE", "IOT"),
    schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
    staging_table=os.getenv("SNOWFLAKE_STAGING_TABLE", "RTLS_STAGING"),
    positions_view=os.getenv("SNOWFLAKE_POSITIONS_VIEW", "RTLS_POSITIONS_V"),
    kpis_table=os.getenv("SNOWFLAKE_KPIS_TABLE", "RTLS_KPIS_DAILY"),
)

st.set_page_config(page_title="RTLS Trajectory Dashboard", layout="wide")
st.title("RTLS Trajectory Dashboard")

# ========= Helpers =========
def connect_snowflake():
    import snowflake.connector
    return snowflake.connector.connect(
        account=SF_CFG["account"],
        user=SF_CFG["user"],
        password=SF_CFG["password"],
        role=SF_CFG["role"],
        warehouse=SF_CFG["warehouse"],
        database=SF_CFG["database"],
        schema=SF_CFG["schema"],
    )

def load_from_snowflake(limit_rows=None):
    con = connect_snowflake()
    q = f"""
    SELECT vehicle_id, ts, x, y, theta
    FROM {SF_CFG["positions_view"]}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY vehicle_id ORDER BY ts) >= 1
    """
    if limit_rows:
        q += f" LIMIT {int(limit_rows)}"
    df = pd.read_sql(q, con)
    con.close()
    return df

def parse_uploaded_csv(file):
    try:
        df_raw = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(io.BytesIO(file.read()), encoding_errors="ignore")
    # try both flattened and JSON-ish shapes
    columns = {c.lower(): c for c in df_raw.columns}
    def get(name_opts):
        for n in name_opts:
            if n.lower() in columns: 
                return columns[n.lower()]
        return None
    ts_col = get(["PAYLOAD_TIMESTAMP","ts","timestamp"])
    x_col  = get(["PAYLOAD_AGVPOSITION_X","x"])
    y_col  = get(["PAYLOAD_AGVPOSITION_Y","y"])
    th_col = get(["PAYLOAD_AGVPOSITION_THETA","theta"])
    id_col = get(["PAYLOAD_SERIALNUMBER","vehicle_id","vehicle"])
    if not all([ts_col, x_col, y_col, id_col]):
        raise ValueError("CSV missing required columns: timestamp, x, y, vehicle_id")
    df = pd.DataFrame({
        "ts": pd.to_datetime(df_raw[ts_col], errors="coerce"),
        "x": pd.to_numeric(df_raw[x_col], errors="coerce"),
        "y": pd.to_numeric(df_raw[y_col], errors="coerce"),
        "theta": pd.to_numeric(df_raw[th_col], errors="coerce") if th_col else np.nan,
        "vehicle_id": df_raw[id_col].astype(str),
    }).dropna(subset=["ts","x","y","vehicle_id"])
    return df.sort_values(["vehicle_id","ts"]).reset_index(drop=True)

def compute_metrics(df):
    def per_vehicle(g):
        g = g.sort_values("ts").copy()
        g["dt_s"]    = g["ts"].diff().dt.total_seconds()
        g["dx"]      = g["x"].diff()
        g["dy"]      = g["y"].diff()
        g["step_m"]  = np.sqrt(g["dx"]**2 + g["dy"]**2)
        g["speed_m_s"] = g["step_m"] / g["dt_s"]
        return g
    out = df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    # filter unrealistic teleports
    MAX_SPEED = 10.0
    out = out[(out["speed_m_s"].isna()) | (out["speed_m_s"] <= MAX_SPEED)].copy()
    # smoothing
    out["x_s"] = out.groupby("vehicle_id")["x"].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    out["y_s"] = out.groupby("vehicle_id")["y"].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    return out

def kpi_table(df):
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        sp = g["speed_m_s"].replace([np.inf,-np.inf], np.nan).dropna()
        avg = float(sp[sp>0].mean()) if not sp.empty else 0.0
        mx  = float(sp.max()) if not sp.empty else 0.0
        dist= float(g["step_m"].fillna(0).sum())
        span= (g["ts"].max() - g["ts"].min()).total_seconds() if len(g)>1 else 0.0
        dwell = float(g["dt_s"].fillna(0)[(g["speed_m_s"].fillna(0) < 0.2)].sum())
        rows.append(dict(vehicle_id=vid, points=len(g), total_distance_m=dist,
                         avg_speed_m_s=avg, max_speed_m_s=mx, dwell_time_s=dwell, time_span_s=span))
    return pd.DataFrame(rows)

def plot_xy(df):
    st.subheader("Trajectory (XY)")
    for vid, g in df.groupby("vehicle_id"):
        fig = plt.figure()
        plt.plot(g["x_s"], g["y_s"], linewidth=1)
        plt.title(f"Trajectory: {vid}")
        plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.axis("equal"); plt.tight_layout()
        st.pyplot(fig)

def plot_heatmap(df):
    st.subheader("Position Heatmap")
    fig = plt.figure()
    plt.hist2d(df["x_s"], df["y_s"], bins=[60,60])
    plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.axis("equal"); plt.colorbar(label="counts"); plt.tight_layout()
    st.pyplot(fig)

def plot_speed_hist(df):
    st.subheader("Speed Distribution (m/s)")
    sp = df["speed_m_s"].replace([np.inf,-np.inf], np.nan).dropna()
    if len(sp)==0:
        st.info("No speeds to plot.")
        return
    fig = plt.figure()
    plt.hist(sp, bins=40)
    plt.xlabel("m/s"); plt.ylabel("count"); plt.tight_layout()
    st.pyplot(fig)

# ========= Sidebar =========
source = st.sidebar.radio("Data source", ["Snowflake", "CSV upload"], index=0)
st.sidebar.write("—")

if source == "Snowflake":
    if st.sidebar.button("Test Snowflake connection"):
        try:
            con = connect_snowflake()
            cur = con.cursor()
            cur.execute("SELECT CURRENT_VERSION()")
            ver = cur.fetchone()[0]
            st.sidebar.success(f"Connected ✓ (Snowflake {ver})")
            cur.close(); con.close()
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

limit_rows = st.sidebar.number_input("Row limit (0 = all)", min_value=0, value=0, step=1000)
run_btn = st.sidebar.button("Run Analysis")

# ========= Main workflow =========
@st.cache_data(show_spinner=False)
def cached_load(source, limit_rows, uploaded_csv):
    if source == "Snowflake":
        lim = None if not limit_rows or int(limit_rows)==0 else int(limit_rows)
        return load_from_snowflake(limit_rows=lim)
    else:
        return parse_uploaded_csv(uploaded_csv)

uploaded_file = None
if source == "CSV upload":
    uploaded_file = st.file_uploader("Upload an RTLS CSV export", type=["csv"])

if run_btn or (source=="CSV upload" and uploaded_file is not None):
    with st.spinner("Loading & analyzing..."):
        df_in = cached_load(source, limit_rows, uploaded_file)
        if df_in.empty:
            st.warning("No data found.")
        else:
            df_proc = compute_metrics(df_in)
            kpis = kpi_table(df_proc)
            # Top KPIs
            st.subheader("KPIs (per vehicle)")
            st.dataframe(kpis, use_container_width=True)
            # Visuals
            col1, col2 = st.columns([2,1])
            with col1:
                plot_xy(df_proc)
            with col2:
                plot_speed_hist(df_proc)
            plot_heatmap(df_proc)

            # ===== Export Tab =====
            st.header("📋 Data Export")
            exp1 = df_proc[["vehicle_id","ts","x","y","x_s","y_s","speed_m_s","dt_s","step_m"]].to_csv(index=False).encode("utf-8")
            st.download_button("Download processed positions (CSV)", exp1, file_name="rtls_positions_processed.csv")
            exp2 = kpis.to_csv(index=False).encode("utf-8")
            st.download_button("Download KPIs (CSV)", exp2, file_name="rtls_kpis.csv")

else:
    st.info("Select a source, optionally test the Snowflake connection, then click **Run Analysis**.")

st.caption("Tip: configure Snowflake credentials in a .env file (see .env.example).")
