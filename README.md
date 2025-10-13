# RTLS Trajectory Dashboard & Snowflake Daily KPIs (Ready-to-Run)

This package gives you:
1. **Streamlit dashboard** for trajectories, speeds, heatmaps, KPIs.
2. **Snowflake DDL + daily KPI task** that aggregates previous day data into `RTLS_KPIS_DAILY`.

## 1) Local setup

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill Snowflake credentials
streamlit run streamlit_app.py
```

## 2) Snowflake setup

Open **snowflake_setup.sql** in the Snowflake UI (or `snowsql`) and run it.
- It creates `IOT.PUBLIC` schema, staging table `RTLS_STAGING`,
  positions view `RTLS_POSITIONS_V`, KPIs table `RTLS_KPIS_DAILY`,
  procedure `SP_RTLS_MAKE_DAILY_KPIS(date)` and task `TASK_RTLS_KPIS_DAILY`.
- Start the task:
```sql
ALTER TASK TASK_RTLS_KPIS_DAILY RESUME;
```
- Run once manually (optional):
```sql
CALL SP_RTLS_MAKE_DAILY_KPIS('2025-06-23');
```

## 3) Loading data into Snowflake

If you already have CSVs in the **flattened** format (columns like the dashboard expects), insert them:
```sql
INSERT INTO RTLS_STAGING (
  PAYLOAD_TIMESTAMP, PAYLOAD_SERIALNUMBER, PAYLOAD_AGVPOSITION_X,
  PAYLOAD_AGVPOSITION_Y, PAYLOAD_AGVPOSITION_THETA,
  PAYLOAD_AGVPOSITION_MAPID, PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE,
  MQTT_TIMESTAMP_ISO_8601, MQTT_TOPIC
)
SELECT
  TO_TIMESTAMP_TZ($1), $2, $3, $4, $5, $6, $7, TO_TIMESTAMP_TZ($8), $9
FROM VALUES
  -- add your rows here or use COPY INTO from staged files
  ;
```

Or use Snowflake stages + COPY:
```sql
CREATE OR REPLACE STAGE IOT_CSV_STAGE FILE_FORMAT=(TYPE=CSV FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1);
-- Put files to stage (from workstation: snowflake-s3/azure toolchain or Snowsight upload)
COPY INTO RTLS_STAGING FROM @IOT_CSV_STAGE ON_ERROR='CONTINUE';
```

## 4) Using the dashboard

Choose **Snowflake** source to query `RTLS_POSITIONS_V` or **CSV upload** to analyze a local export.
The sidebar includes:
- **Connection Test** button.
- **Row limit**.
- **Run Analysis**.

Outputs:
- KPIs table per vehicle.
- Trajectory plot per vehicle.
- Speed histogram.
- Position heatmap.
- **📋 Data Export**: processed positions & KPIs as CSV downloads.

## 5) Security notes
- Prefer Snowflake key-pair auth for production (see `snowflake-connector-python` docs).
- Store `.env` securely; never commit credentials.

## 6) Customization
- Teleport filter: 10 m/s
- Dwell threshold: 0.2 m/s
- Heatmap bins: 60 x 60

Change thresholds in `streamlit_app.py` (search for MAX_SPEED and 0.2).
