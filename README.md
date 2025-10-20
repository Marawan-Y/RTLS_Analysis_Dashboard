# RTLS Trajectory Analytics Dashboard

An interactive Streamlit dashboard for exploring RTLS (Real-Time Location System) trajectories, computing KPIs, and visualizing paths and heatmaps. You can connect directly to **Snowflake** from inside the app (no coding required), or load a **CSV** sample.

---

## What you can do

- Connect to your **Snowflake** database by typing credentials in the sidebar  
- Choose which **columns** to load from your Snowflake table/view  
- Filter by **date** and **row limit** before loading  
- Upload **CSV** data instead of Snowflake (optional)  
- View **trajectories**, **heatmaps**, **speed analysis**, and **KPI tables**  
- Export processed data and KPIs as **CSV**  
- (Optional) Overlay an **SVG** plant layout behind the trajectories

---

## 1) Quick start (the shortest path)

> Works on Windows, macOS, and Linux.

1. **Install Python 3.10+**  
   - Windows: [python.org/downloads](https://www.python.org/downloads/) – check “Add python.exe to PATH” during install  
   - macOS (Homebrew):  
     ```bash
     brew install python
     ```
   - Linux (Debian/Ubuntu):  
     ```bash
     sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
     ```

2. **Download the project** (ZIP) or clone:
   ```bash
   git clone https://github.com/Marawan-Y/RTLS_Analysis_Dashboard.git
   cd RTLS_Analysis_Dashboard
   ```

3. **Create a virtual environment** (recommended):
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate
     ```
   - macOS/Linux (bash/zsh):
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the app**:
   ```bash
   streamlit run streamlit_app.py --server.port=5000 --server.address=127.0.0.1
   ```
   Streamlit will open in your browser (usually at http://localhost:5000).

---

## 2) Using the app

### A) Connect to Snowflake (no coding)

1. In the **left sidebar**, pick **“Snowflake”** as data source.  
2. Fill in your Snowflake details:
   - **Account**: e.g. `ab12345.eu-central-1`
   - **User** / **Password**
   - **Role**: e.g. `SYSADMIN`
   - **Warehouse**: e.g. `COMPUTE_WH`
   - **Database**: e.g. `IOT`
   - **Schema**: e.g. `PUBLIC`
   - **Table or View**: e.g. `RTLS_STAGING` or `RTLS_POSITIONS_V`
3. Click **🔌 Connect & Load Columns**.  
   - You should see a success message and a list of available columns.
4. Choose which columns you want to load.  
   - The app will **automatically ensure** the required ones are included:
     - `PAYLOAD_SERIALNUMBER`, `PAYLOAD_TIMESTAMP`, `PAYLOAD_AGVPOSITION_X`, `PAYLOAD_AGVPOSITION_Y`
5. (Optional) Add **filters**:
   - **Row limit** (0 = all)
   - **Date filter** (exact date)
6. Click **Run Analysis**.

> 🔐 **Security tip:** The app does **not** save your password. If you share screenshots or recordings, redact credentials.

### B) Load a CSV instead

1. In the sidebar, choose **“CSV Upload”**.  
2. Upload a file with at least these columns:  
   - `PAYLOAD_TIMESTAMP`, `PAYLOAD_SERIALNUMBER`, `PAYLOAD_AGVPOSITION_X`, `PAYLOAD_AGVPOSITION_Y`  
   Optional columns (auto-used if present):  
   - `PAYLOAD_AGVPOSITION_THETA`, `PAYLOAD_AGVPOSITION_MAPID`, `PAYLOAD_AGVPOSITION_POSITIONINITIALIZED`, `PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE`, `PAYLOAD_VELOCITY_VX`, `PAYLOAD_VELOCITY_VY`, `PAYLOAD_VELOCITY_OMEGA`, `MQTT_TIMESTAMP_ISO_8601`, `MQTT_TOPIC`
3. Click **Run Analysis**.

---

## 3) Optional: overlay an SVG plant layout

1. In the sidebar, enable **Use plant layout**.  
2. Upload an **.svg** file of your site layout.  
3. The app parses and scales the layout to your data bounds and draws it semi-transparent behind the trajectories/heatmaps.

---

## 4) Optional: environment variables (legacy / fallback)

The app is meant for non-developers and doesn’t require a `.env`.  
If you prefer defaults to pre-fill the sidebar, create a file named `.env` in the project root:

```env
SNOWFLAKE_ACCOUNT=ab12345.eu-central-1
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=SYSADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=IOT
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_TABLE=RTLS_STAGING
```

> These values will appear as **defaults** in the UI. You can still edit them live in the app.

---

## 5) Data model (what the app expects)

Minimum columns:
- `PAYLOAD_TIMESTAMP` (timestamp)
- `PAYLOAD_SERIALNUMBER` (vehicle id)
- `PAYLOAD_AGVPOSITION_X`, `PAYLOAD_AGVPOSITION_Y` (numeric positions)

Useful optional columns:
- `PAYLOAD_AGVPOSITION_THETA`, `PAYLOAD_AGVPOSITION_MAPID`,
  `PAYLOAD_AGVPOSITION_POSITIONINITIALIZED`, `PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE`,
  `PAYLOAD_VELOCITY_VX`, `PAYLOAD_VELOCITY_VY`, `PAYLOAD_VELOCITY_OMEGA`,
  `MQTT_TIMESTAMP_ISO_8601`, `MQTT_TOPIC`

The app automatically **renames** key Snowflake fields into internal names:
- `PAYLOAD_SERIALNUMBER → vehicle_id`
- `PAYLOAD_TIMESTAMP → ts`
- `PAYLOAD_AGVPOSITION_X → x`, `PAYLOAD_AGVPOSITION_Y → y`
- …and so on for the optional columns

---

## 6) Troubleshooting

- **“st.session_state has no attribute 'sf_cfg'”**  
  You’re running an older file. Pull the latest `streamlit_app.py` where session state is initialized **before** rendering the sidebar.

- **Cannot connect to Snowflake**  
  - Recheck Account format (e.g., `ab12345.eu-central-1`)  
  - Confirm **Role**, **Warehouse**, **Database**, **Schema** exist and your **User** has access  
  - Your network may require a VPN or allowlisting Snowflake IPs  
  - Try a minimal test:
    ```sql
    SELECT CURRENT_VERSION();
    ```

- **Columns not found**  
  - Make sure your **Table or View** name is correct (Snowflake names are often **UPPERCASE**).  
  - Use the **🔌 Connect & Load Columns** button again to refresh the column list.  
  - Ensure the required fields exist (`PAYLOAD_*` columns listed above).

- **Matplotlib/plot errors**  
  Run:
  ```bash
  pip install --upgrade matplotlib numpy pandas
  ```

- **Streamlit doesn’t open**  
  - Check the terminal output for the local URL (e.g., http://localhost:8501)  
  - Try a different port:
    ```bash
    streamlit run streamlit_app.py --server.port 8502
    ```

---

## 7) Developer notes (optional)

- Python ≥ 3.10 recommended  
- Main entry point: `streamlit_app.py`  
- All Snowflake queries are built dynamically from UI selections  
- No database **setup** SQL is run by the app (it assumes your Snowflake objects already exist)

---

## 8) License & Credits

This dashboard uses:
- **Streamlit** for UI
- **Pandas/Numpy** for data processing
- **Matplotlib** for plots
- An internal **SVG** parser/renderer for layout overlays

---

### Need help?

If you run into any issues, feel free to open an issue with:
- OS (Windows/macOS/Linux)  
- Python version (e.g., 3.11)  
- A screenshot or the exact error text  
- Whether you’re using **Snowflake** or **CSV** mode
