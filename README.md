# Enhanced RTLS Trajectory Analytics Dashboard

An advanced Streamlit dashboard for Real-Time Location System (RTLS) trajectory analysis with multi-tab interface, SVG plant layout support, and comprehensive KPI tracking.

## Key Enhancements

### 1. **Multi-Tab Interface**
- **Overview Tab**: System-wide metrics and quick summary
- **KPIs Tab**: Detailed performance indicators with utilization charts
- **Trajectories Tab**: Vehicle path visualization with optional plant layout overlay
- **Heatmap Tab**: Position density visualization
- **Speed Analysis Tab**: Speed distribution and temporal analysis
- **Export Tab**: Data download functionality with preview

### 2. **SVG Plant Layout Integration**
- Upload SVG files of your facility layout
- Automatic scaling and alignment with trajectory data
- Transparent overlay for better context
- Support for common SVG elements (rectangles, circles, paths, etc.)

### 3. **Enhanced Snowflake Integration**
- Direct connection to existing Snowflake tables
- Proper handling of the exact table structure from your MQTT/RTLS data
- Date filtering capabilities
- Row limit controls for performance

## Prerequisites

- Python 3.8+
- Snowflake account with RTLS data tables
- SVG file of plant layout (optional)

## Installation

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Snowflake connection**:
```bash
cp .env.example .env
# Edit .env with your Snowflake credentials
```

4. **Prepare your SVG layout** (optional):
   - Export your plant layout as SVG from CAD software
   - Ensure coordinates align with your RTLS coordinate system

## Configuration

### Environment Variables (.env file)
```env
SNOWFLAKE_ACCOUNT=your_account.region
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=SYSADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=IOT
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_STAGING_TABLE=RTLS_STAGING
```

### Expected Table Structure

The dashboard expects data with these columns:
- `PAYLOAD_TIMESTAMP` - Position timestamp
- `PAYLOAD_SERIALNUMBER` - Vehicle ID
- `PAYLOAD_AGVPOSITION_X` - X coordinate
- `PAYLOAD_AGVPOSITION_Y` - Y coordinate
- `PAYLOAD_AGVPOSITION_THETA` - Orientation
- `PAYLOAD_AGVPOSITION_MAPID` - Map identifier
- `PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE` - Localization quality
- `PAYLOAD_VELOCITY_VX`, `PAYLOAD_VELOCITY_VY`, `PAYLOAD_VELOCITY_OMEGA` - Velocity components

## Running the Dashboard

### Basic Usage
```bash
streamlit run streamlit_app.py
```

## Features Guide

### 1. Data Source Selection
- **Snowflake**: Connect directly to your database
- **CSV Upload**: Analyze exported data files

### 2. Connection Testing
- Click "Test Connection" to verify Snowflake connectivity
- View available tables in your schema

### 3. Query Filters
- **Row Limit**: Control data volume (0 = unlimited)
- **Date Filter**: Select specific days for analysis

### 4. Layout Integration
- Check "Use plant layout"
- Upload your SVG file
- The layout will automatically scale to match trajectory coordinates

### 5. Analysis Settings
- **Max Speed Filter**: Remove unrealistic movements (default: 10 m/s)
- **Dwell Threshold**: Define stationary threshold (default: 0.2 m/s)
- **Smoothing Window**: Trajectory smoothing parameter

## KPI Calculations

The dashboard calculates:
- **Total Distance**: Sum of all movements
- **Average Speed**: Mean velocity when moving
- **Max/Min Speed**: Speed extremes
- **Dwell Time**: Time spent below threshold
- **Utilization %**: Percentage of time in motion
- **Time Span**: Total observation period

## Visualization Features

### Trajectory Plot
- Color-coded by vehicle
- Start (green) and end (red) markers
- Optional SVG layout background
- Smoothed paths for clarity

### Heatmap
- Adjustable resolution (20-100 bins)
- Density-based coloring
- Layout overlay support

### Speed Analysis
- Distribution histogram
- Time-series plot with smoothing
- Statistical summaries

## Data Export

Export options include:
- **Processed Positions**: Cleaned trajectory data with calculated metrics
- **KPI Summary**: Aggregated performance metrics per vehicle

## Troubleshooting

### Common Issues

1. **Snowflake Connection Failed**
   - Verify credentials in .env file
   - Check network/firewall settings
   - Ensure warehouse is running

2. **SVG Layout Not Displaying**
   - Check SVG file format (standard SVG 1.1)
   - Verify coordinate systems match
   - Try simplifying complex SVG elements

3. **Performance Issues**
   - Use row limits for large datasets
   - Apply date filters to reduce data volume
   - Consider upgrading Snowflake warehouse size

## Advanced Customization

### Modifying SVG Handler
Edit `svg_layout_handler.py` to:
- Add support for additional SVG elements
- Customize rendering styles
- Implement coordinate transformations

### Adding New KPIs
In `streamlit_app.py`, modify the `kpi_table()` function:
```python
def kpi_table(df):
    # Add your custom KPI calculations
    custom_kpi = calculate_custom_metric(df)
    # Include in results
```

### Creating Custom Tabs
Add new tabs in the main interface:
```python
tab1, tab2, ..., custom_tab = st.tabs([..., "Custom Analysis"])
with custom_tab:
    # Your custom analysis code
```

## File Structure

```
project/
├── streamlit_app.py           # Main dashboard application
├── svg_layout_handler.py       # SVG parsing and rendering module
├── .env.example               # Environment variable template
├── .env                       # Your configuration (git-ignored)
├── requirements.txt           # Python dependencies
├── snowflake_setup.sql       # Database setup script
└── README.md                  # This file
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Snowflake connection logs
3. Verify data format matches expectations
4. Test with sample data first

---

**Note**: Ensure your Snowflake tables are properly populated and the daily KPI task is running for complete functionality.