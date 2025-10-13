
-- ===========================
-- RTLS Snowflake Setup (DDL)
-- ===========================
-- Adjust identifiers/roles/warehouses to your environment.

-- 0) Context
USE ROLE SYSADMIN;
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH WITH WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
CREATE DATABASE IF NOT EXISTS IOT;
CREATE SCHEMA IF NOT EXISTS IOT.PUBLIC;
USE DATABASE IOT;
USE SCHEMA PUBLIC;

-- 1) Staging table (flattened columns like your CSV)
CREATE OR REPLACE TABLE RTLS_STAGING (
  PAYLOAD_TIMESTAMP           TIMESTAMP_TZ,
  PAYLOAD_SERIALNUMBER        STRING,
  PAYLOAD_AGVPOSITION_X       DOUBLE,
  PAYLOAD_AGVPOSITION_Y       DOUBLE,
  PAYLOAD_AGVPOSITION_THETA   DOUBLE,
  PAYLOAD_AGVPOSITION_MAPID   STRING,
  PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE DOUBLE,
  MQTT_TIMESTAMP_ISO_8601     TIMESTAMP_TZ,
  MQTT_TOPIC                  STRING
);

-- 2) Canonical positions view
CREATE OR REPLACE VIEW RTLS_POSITIONS_V AS
SELECT
  PAYLOAD_SERIALNUMBER          AS vehicle_id,
  PAYLOAD_TIMESTAMP             AS ts,
  PAYLOAD_AGVPOSITION_X         AS x,
  PAYLOAD_AGVPOSITION_Y         AS y,
  PAYLOAD_AGVPOSITION_THETA     AS theta,
  MQTT_TOPIC                    AS topic,
  MQTT_TIMESTAMP_ISO_8601       AS received_at
FROM RTLS_STAGING
WHERE vehicle_id IS NOT NULL AND ts IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL;

-- 3) KPIs sink table (daily aggregation)
CREATE OR REPLACE TABLE RTLS_KPIS_DAILY (
  biz_date            DATE,
  vehicle_id          STRING,
  points              NUMBER,
  total_distance_m    DOUBLE,
  avg_speed_m_s       DOUBLE,
  max_speed_m_s       DOUBLE,
  dwell_time_s        DOUBLE,
  time_span_s         DOUBLE,
  created_at          TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP()
);

-- 4) Procedure to aggregate yesterday's data into KPIs
CREATE OR REPLACE PROCEDURE SP_RTLS_MAKE_DAILY_KPIS(target_date DATE)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
  -- Aggregate on a specific date (UTC). Use DATE(ts) filter.
  INSERT INTO RTLS_KPIS_DAILY (biz_date, vehicle_id, points, total_distance_m, avg_speed_m_s, max_speed_m_s, dwell_time_s, time_span_s)
  WITH base AS (
    SELECT vehicle_id, ts, x, y,
           LAG(ts) OVER (PARTITION BY vehicle_id ORDER BY ts)  AS prev_ts,
           LAG(x)  OVER (PARTITION BY vehicle_id ORDER BY ts)  AS prev_x,
           LAG(y)  OVER (PARTITION BY vehicle_id ORDER BY ts)  AS prev_y
    FROM RTLS_POSITIONS_V
    WHERE DATE(ts) = target_date
  ),
  steps AS (
    SELECT
      vehicle_id,
      ts,
      prev_ts,
      DATEDIFF('second', prev_ts, ts) AS dt_s,
      SQRT( POW(x - prev_x, 2) + POW(y - prev_y, 2) ) AS step_m,
      CASE WHEN DATEDIFF('second', prev_ts, ts) = 0 THEN NULL
           ELSE (SQRT( POW(x - prev_x, 2) + POW(y - prev_y, 2) ) / NULLIF(DATEDIFF('second', prev_ts, ts),0)) END AS speed_m_s
    FROM base
  ),
  filtered AS (
    SELECT * FROM steps WHERE speed_m_s IS NULL OR speed_m_s <= 10.0  -- teleport filter
  ),
  agg AS (
    SELECT
      vehicle_id,
      COUNT(*) AS points,
      SUM(COALESCE(step_m,0)) AS total_distance_m,
      AVG(NULLIF(speed_m_s,0)) AS avg_speed_m_s,
      MAX(speed_m_s) AS max_speed_m_s,
      SUM(CASE WHEN COALESCE(speed_m_s,0) < 0.2 THEN COALESCE(dt_s,0) ELSE 0 END) AS dwell_time_s,
      DATEDIFF('second', MIN(ts), MAX(ts)) AS time_span_s
    FROM filtered
    GROUP BY vehicle_id
  )
  SELECT target_date, vehicle_id, points, total_distance_m, avg_speed_m_s, max_speed_m_s, dwell_time_s, time_span_s
  FROM agg;
  SELECT 'OK';
$$;

-- 5) Task to run every day at 01:00 UTC for yesterday
-- (Requires ACCOUNTADMIN to enable task scheduling in some orgs)
USE ROLE ACCOUNTADMIN;
GRANT USAGE, OPERATE ON WAREHOUSE COMPUTE_WH TO ROLE SYSADMIN;

USE ROLE SYSADMIN;
CREATE OR REPLACE TASK TASK_RTLS_KPIS_DAILY
  WAREHOUSE = COMPUTE_WH
  SCHEDULE = 'USING CRON 0 1 * * * UTC'
AS
  CALL SP_RTLS_MAKE_DAILY_KPIS(DATEADD('day', -1, CURRENT_DATE()));

-- To start the task:
-- ALTER TASK TASK_RTLS_KPIS_DAILY RESUME;

-- To run ad-hoc for a specific date:
-- CALL SP_RTLS_MAKE_DAILY_KPIS('2025-06-23');
