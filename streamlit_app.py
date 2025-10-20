import os
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import streamlit as st
from dotenv import load_dotenv
import base64
from pathlib import Path

# ======= (SVG handler required imports) =======
import xml.etree.ElementTree as ET
from matplotlib.path import Path as MplPath
import re
# ==============================================

# ========= Config =========
# Load environment variables if present.  These provide sensible defaults for the
# Snowflake connection fields but are no longer required because users can
# override them in the UI.  Using load_dotenv allows the app to continue
# functioning in development environments without breaking when .env is missing.
load_dotenv()

# Set default Snowflake configuration values from the environment.  These are
# used to pre-populate the credential fields in the sidebar but are not
# required.  If no environment variables are set the fields will be blank.
DEFAULT_SF_CFG = dict(
    account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
    user=os.getenv("SNOWFLAKE_USER", ""),
    password=os.getenv("SNOWFLAKE_PASSWORD", ""),
    role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    database=os.getenv("SNOWFLAKE_DATABASE", "IOT"),
    schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
)

# Configure the Streamlit page.  We use a wide layout and expand the sidebar
# because the application has a number of configuration options.
st.set_page_config(
    page_title="RTLS Trajectory Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling of tabs and metric cards.  This CSS is
# unchanged from the original application and provides a more polished look.
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title and divider
st.title("RTLS Trajectory Analytics Dashboard")
st.markdown("---")

# ========= Session State Management =========
# Initialise session state variables on first load.  These variables track
# whether data has been loaded, hold the processed data and KPIs, store the
# uploaded SVG layout and parsed handler, and hold Snowflake connection
# information entered by the user.
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_proc = None
    st.session_state.kpis = None
    st.session_state.layout_svg = None
    st.session_state.svg_handler = None
    # Snowflake session configuration fields
    st.session_state.sf_cfg = DEFAULT_SF_CFG.copy()
    st.session_state.table_name = "RTLS_STAGING"
    st.session_state.available_columns = None
    st.session_state.selected_columns = None

# ========= SVG Layout Handler =========
class SVGLayoutHandler:
    """
    Handler for SVG plant layouts to be used as backgrounds in trajectory plots.
    The implementation here mirrors the standalone module but is defined inline
    to avoid external imports.  It parses SVG elements and can render them on
    matplotlib axes.
    """

    def __init__(self, svg_content=None):
        """
        Initialise the SVG handler.

        Args:
            svg_content (str): Raw SVG content as a string.
        """
        self.svg_content = svg_content
        self.elements = []
        # Default bounds used when viewBox or width/height are not specified.
        self.bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}

        if svg_content:
            self.parse_svg()

    def parse_svg(self):
        """Parse the SVG content and extract drawable elements."""
        if not self.svg_content:
            return

        try:
            root = ET.fromstring(self.svg_content)

            # Determine the coordinate bounds using the viewBox or width/height attributes
            viewbox = root.get('viewBox')
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    self.bounds = {
                        'min_x': float(parts[0]),
                        'min_y': float(parts[1]),
                        'max_x': float(parts[0]) + float(parts[2]),
                        'max_y': float(parts[1]) + float(parts[3])
                    }
            else:
                width = root.get('width', '100')
                height = root.get('height', '100')
                self.bounds = {
                    'min_x': 0,
                    'min_y': 0,
                    'max_x': float(re.sub(r'[^\d.]', '', width)),
                    'max_y': float(re.sub(r'[^\d.]', '', height))
                }

            # Recursively parse elements in the SVG
            self._parse_element(root)

        except Exception as e:
            print(f"Error parsing SVG: {e}")

    def _parse_element(self, element, transform=None):
        """Recursively parse SVG elements into primitive definitions."""
        # Handle group elements which may have a transform
        if element.tag.endswith('g'):
            group_transform = element.get('transform')
            for child in element:
                self._parse_element(child, group_transform)

        # Handle individual element types
        elif element.tag.endswith('rect'):
            self._parse_rect(element, transform)
        elif element.tag.endswith('circle'):
            self._parse_circle(element, transform)
        elif element.tag.endswith('ellipse'):
            self._parse_ellipse(element, transform)
        elif element.tag.endswith('line'):
            self._parse_line(element, transform)
        elif element.tag.endswith('polyline'):
            self._parse_polyline(element, transform)
        elif element.tag.endswith('polygon'):
            self._parse_polygon(element, transform)
        elif element.tag.endswith('path'):
            self._parse_path(element, transform)

        # Parse child elements recursively
        for child in element:
            self._parse_element(child, transform)

    def _parse_rect(self, element, transform):
        """Parse a rectangle element into a dictionary representation."""
        try:
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))
            style = self._parse_style(element)
            self.elements.append({
                'type': 'rect',
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'style': style,
                'transform': transform
            })
        except Exception:
            pass

    def _parse_circle(self, element, transform):
        """Parse a circle element."""
        try:
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            r = float(element.get('r', 0))
            style = self._parse_style(element)
            self.elements.append({
                'type': 'circle',
                'cx': cx,
                'cy': cy,
                'r': r,
                'style': style,
                'transform': transform
            })
        except Exception:
            pass

    def _parse_ellipse(self, element, transform):
        """Parse an ellipse element."""
        try:
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            rx = float(element.get('rx', 0))
            ry = float(element.get('ry', 0))
            style = self._parse_style(element)
            self.elements.append({
                'type': 'ellipse',
                'cx': cx,
                'cy': cy,
                'rx': rx,
                'ry': ry,
                'style': style,
                'transform': transform
            })
        except Exception:
            pass

    def _parse_line(self, element, transform):
        """Parse a line element."""
        try:
            x1 = float(element.get('x1', 0))
            y1 = float(element.get('y1', 0))
            x2 = float(element.get('x2', 0))
            y2 = float(element.get('y2', 0))
            style = self._parse_style(element)
            self.elements.append({
                'type': 'line',
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'style': style,
                'transform': transform
            })
        except Exception:
            pass

    def _parse_polyline(self, element, transform):
        """Parse a polyline element."""
        try:
            points_str = element.get('points', '')
            points = self._parse_points(points_str)
            if points:
                style = self._parse_style(element)
                self.elements.append({
                    'type': 'polyline',
                    'points': points,
                    'style': style,
                    'transform': transform
                })
        except Exception:
            pass

    def _parse_polygon(self, element, transform):
        """Parse a polygon element."""
        try:
            points_str = element.get('points', '')
            points = self._parse_points(points_str)
            if points:
                style = self._parse_style(element)
                self.elements.append({
                    'type': 'polygon',
                    'points': points,
                    'style': style,
                    'transform': transform
                })
        except Exception:
            pass

    def _parse_path(self, element, transform):
        """Parse a path element.  This simplified parser stores the raw path
        definition string rather than generating a complete shape because
        accurately parsing arbitrary SVG paths is outside the scope of this
        application."""
        try:
            d = element.get('d', '')
            style = self._parse_style(element)
            self.elements.append({
                'type': 'path',
                'd': d,
                'style': style,
                'transform': transform
            })
        except Exception:
            pass

    def _parse_points(self, points_str):
        """Convert a coordinate string into a list of (x, y) tuples."""
        points = []
        parts = points_str.replace(',', ' ').split()
        for i in range(0, len(parts) - 1, 2):
            try:
                x = float(parts[i])
                y = float(parts[i + 1])
                points.append((x, y))
            except Exception:
                pass
        return points

    def _parse_style(self, element):
        """Parse style attributes from an SVG element."""
        style = {
            'fill': element.get('fill', 'none'),
            'stroke': element.get('stroke', 'black'),
            'stroke_width': float(element.get('stroke-width', 1)),
            'opacity': float(element.get('opacity', 1.0)),
            'fill_opacity': float(element.get('fill-opacity', 1.0)),
            'stroke_opacity': float(element.get('stroke-opacity', 1.0))
        }
        style_attr = element.get('style', '')
        if style_attr:
            for item in style_attr.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'fill':
                        style['fill'] = value
                    elif key == 'stroke':
                        style['stroke'] = value
                    elif key == 'stroke-width':
                        try:
                            style['stroke_width'] = float(value.replace('px', ''))
                        except Exception:
                            pass
                    elif key == 'opacity':
                        try:
                            style['opacity'] = float(value)
                        except Exception:
                            pass
        return style

    def render_on_axes(self, ax, scale_to_data=None, alpha=0.3):
        """
        Render all parsed SVG elements on a Matplotlib axis.  Elements are
        scaled to match data coordinates if scale_to_data is provided.

        Args:
            ax: Matplotlib axes on which to draw.
            scale_to_data: ((x_min, x_max), (y_min, y_max)) to map SVG to
                data coordinates.
            alpha: Transparency factor for the entire layout.
        """
        if not self.elements:
            return

        scale_x = 1.0
        scale_y = 1.0
        offset_x = 0
        offset_y = 0
        if scale_to_data:
            x_range, y_range = scale_to_data
            svg_width = self.bounds['max_x'] - self.bounds['min_x']
            svg_height = self.bounds['max_y'] - self.bounds['min_y']
            if svg_width > 0 and svg_height > 0:
                data_width = x_range[1] - x_range[0]
                data_height = y_range[1] - y_range[0]
                scale_x = data_width / svg_width
                scale_y = data_height / svg_height
                # Maintain aspect ratio by using the smaller scale for both axes
                scale = min(scale_x, scale_y)
                scale_x = scale_y = scale
                offset_x = x_range[0]
                offset_y = y_range[0]

        for elem in self.elements:
            self._render_element(ax, elem, scale_x, scale_y, offset_x, offset_y, alpha)

    def _render_element(self, ax, elem, scale_x, scale_y, offset_x, offset_y, alpha):
        """Render a single SVG element onto an axis."""
        elem_type = elem['type']
        style = elem['style']
        fill_color = style['fill'] if style['fill'] != 'none' else None
        edge_color = style['stroke'] if style['stroke'] != 'none' else None
        linewidth = style['stroke_width'] * min(scale_x, scale_y)
        elem_alpha = style['opacity'] * alpha
        if elem_type == 'rect':
            rect = patches.Rectangle(
                (elem['x'] * scale_x + offset_x, elem['y'] * scale_y + offset_y),
                elem['width'] * scale_x,
                elem['height'] * scale_y,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=elem_alpha
            )
            ax.add_patch(rect)
        elif elem_type == 'circle':
            circle = patches.Circle(
                (elem['cx'] * scale_x + offset_x, elem['cy'] * scale_y + offset_y),
                elem['r'] * min(scale_x, scale_y),
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=elem_alpha
            )
            ax.add_patch(circle)
        elif elem_type == 'ellipse':
            ellipse = patches.Ellipse(
                (elem['cx'] * scale_x + offset_x, elem['cy'] * scale_y + offset_y),
                elem['rx'] * 2 * scale_x,
                elem['ry'] * 2 * scale_y,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=elem_alpha
            )
            ax.add_patch(ellipse)
        elif elem_type == 'line':
            ax.plot(
                [elem['x1'] * scale_x + offset_x, elem['x2'] * scale_x + offset_x],
                [elem['y1'] * scale_y + offset_y, elem['y2'] * scale_y + offset_y],
                color=edge_color if edge_color else 'black',
                linewidth=linewidth,
                alpha=elem_alpha
            )
        elif elem_type == 'polyline':
            points = elem.get('points')
            if points:
                x_coords = [p[0] * scale_x + offset_x for p in points]
                y_coords = [p[1] * scale_y + offset_y for p in points]
                ax.plot(x_coords, y_coords, color=edge_color if edge_color else 'black', linewidth=linewidth, alpha=elem_alpha)
        elif elem_type == 'polygon':
            points = elem.get('points')
            if points:
                scaled_points = [(p[0] * scale_x + offset_x, p[1] * scale_y + offset_y) for p in points]
                polygon = patches.Polygon(scaled_points, facecolor=fill_color, edgecolor=edge_color, linewidth=linewidth, alpha=elem_alpha)
                ax.add_patch(polygon)

# Helper to integrate the layout with the trajectory plot
def integrate_layout_with_plot(ax, df, svg_handler):
    """
    Draw the SVG layout on the provided axes scaled to fit the trajectory data.
    If smoothed coordinates are available they are used for calculating bounds.
    """
    if svg_handler and svg_handler.elements:
        x_col = "x_s" if "x_s" in df.columns else "x"
        y_col = "y_s" if "y_s" in df.columns else "y"
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        x_padding = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 1.0
        y_padding = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1.0
        x_range = (x_min - x_padding, x_max + x_padding)
        y_range = (y_min - y_padding, y_max + y_padding)
        svg_handler.render_on_axes(ax, scale_to_data=(x_range, y_range), alpha=0.3)

# ========= Snowflake Integration =========
def connect_snowflake(cfg):
    """Connect to Snowflake using the provided configuration dictionary."""
    import snowflake.connector
    return snowflake.connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        role=cfg["role"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
    )

def load_from_snowflake(cfg, table_name, selected_cols, limit_rows=None, date_filter=None):
    """
    Load data from Snowflake based on the columns selected by the user.  This
    function builds a SELECT statement dynamically.  It always includes
    PAYLOAD_SERIALNUMBER, PAYLOAD_TIMESTAMP, PAYLOAD_AGVPOSITION_X and
    PAYLOAD_AGVPOSITION_Y in the query even if the user does not select them,
    because they are required for trajectory analysis.

    Args:
        cfg: Dict with Snowflake connection parameters.
        table_name: Name of the table to query.
        selected_cols: List of column names selected by the user.
        limit_rows: Optional integer to limit number of rows returned.
        date_filter: Optional date object to filter PAYLOAD_TIMESTAMP.
    Returns:
        pandas.DataFrame
    """
    # Ensure required columns are present
    required = ['PAYLOAD_SERIALNUMBER', 'PAYLOAD_TIMESTAMP', 'PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y']
    cols_set = set(selected_cols) if selected_cols else set()
    for col in required:
        cols_set.add(col)
    column_list = list(cols_set)
    col_select_sql = ", ".join(column_list)
    # Build base query
    full_table = f"{cfg['database']}.{cfg['schema']}.{table_name}"
    query = f"SELECT {col_select_sql} FROM {full_table}"
    conditions = ["PAYLOAD_SERIALNUMBER IS NOT NULL", "PAYLOAD_TIMESTAMP IS NOT NULL",
                  "PAYLOAD_AGVPOSITION_X IS NOT NULL", "PAYLOAD_AGVPOSITION_Y IS NOT NULL"]
    if date_filter:
        # Convert date to ISO string (YYYY-MM-DD)
        conditions.append(f"DATE(PAYLOAD_TIMESTAMP) = '{date_filter}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY PAYLOAD_SERIALNUMBER, PAYLOAD_TIMESTAMP"
    if limit_rows and limit_rows > 0:
        query += f" LIMIT {int(limit_rows)}"
    con = connect_snowflake(cfg)
    df = pd.read_sql(query, con)
    con.close()
    # Rename columns to the names expected by downstream processing
    column_mapping = {
        'PAYLOAD_TIMESTAMP': 'ts',
        'PAYLOAD_SERIALNUMBER': 'vehicle_id',
        'PAYLOAD_AGVPOSITION_X': 'x',
        'PAYLOAD_AGVPOSITION_Y': 'y',
        'PAYLOAD_AGVPOSITION_THETA': 'theta',
        'PAYLOAD_AGVPOSITION_MAPID': 'map_id',
        'PAYLOAD_AGVPOSITION_POSITIONINITIALIZED': 'position_initialized',
        'PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE': 'localization_score',
        'PAYLOAD_VELOCITY_VX': 'velocity_x',
        'PAYLOAD_VELOCITY_VY': 'velocity_y',
        'PAYLOAD_VELOCITY_OMEGA': 'velocity_omega',
    }
    rename_cols = {col: new for col, new in column_mapping.items() if col in df.columns}
    df = df.rename(columns=rename_cols)
    # Convert data types where appropriate
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], errors="coerce")
    for numeric_col in ['x', 'y', 'theta', 'localization_score', 'velocity_x', 'velocity_y', 'velocity_omega']:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
    if 'position_initialized' in df.columns:
        df['position_initialized'] = df['position_initialized'].astype(bool, errors='ignore')
    # Drop rows missing critical fields
    drop_cols = [c for c in ['ts', 'x', 'y', 'vehicle_id'] if c in df.columns]
    df = df.dropna(subset=drop_cols)
    return df.sort_values(['vehicle_id', 'ts']).reset_index(drop=True)

# ========= CSV Parsing (unchanged) =========
def parse_uploaded_csv(file):
    """Parse an uploaded CSV file using the expected RTLS data schema."""
    try:
        df_raw = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(io.BytesIO(file.read()), encoding_errors="ignore")
    column_mapping = {
        'PAYLOAD_TIMESTAMP': 'ts',
        'PAYLOAD_SERIALNUMBER': 'vehicle_id',
        'PAYLOAD_AGVPOSITION_X': 'x',
        'PAYLOAD_AGVPOSITION_Y': 'y',
        'PAYLOAD_AGVPOSITION_THETA': 'theta',
        'PAYLOAD_AGVPOSITION_MAPID': 'map_id',
        'PAYLOAD_AGVPOSITION_POSITIONINITIALIZED': 'position_initialized',
        'PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE': 'localization_score',
        'PAYLOAD_VELOCITY_VX': 'velocity_x',
        'PAYLOAD_VELOCITY_VY': 'velocity_y',
        'PAYLOAD_VELOCITY_OMEGA': 'velocity_omega'
    }
    required_cols = ['PAYLOAD_TIMESTAMP', 'PAYLOAD_SERIALNUMBER', 'PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y']
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    df_clean = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df_raw.columns:
            if old_col == 'PAYLOAD_TIMESTAMP':
                df_clean[new_col] = pd.to_datetime(df_raw[old_col], errors="coerce")
            elif old_col in ['PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y', 'PAYLOAD_AGVPOSITION_THETA', 'PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE', 'PAYLOAD_VELOCITY_VX', 'PAYLOAD_VELOCITY_VY', 'PAYLOAD_VELOCITY_OMEGA']:
                df_clean[new_col] = pd.to_numeric(df_raw[old_col], errors="coerce")
            elif old_col == 'PAYLOAD_AGVPOSITION_POSITIONINITIALIZED':
                df_clean[new_col] = df_raw[old_col].astype(bool)
            else:
                df_clean[new_col] = df_raw[old_col].astype(str)
    df_clean = df_clean.dropna(subset=['ts', 'x', 'y', 'vehicle_id'])
    return df_clean.sort_values(['vehicle_id', 'ts']).reset_index(drop=True)

# ========= Metrics and KPI Calculation (unchanged) =========
def compute_metrics(df):
    """
    Compute per-vehicle metrics such as step distance, speed, acceleration and
    smoothed position.  This function is unchanged from the original but is
    reproduced here for completeness.
    """
    def per_vehicle(g):
        g = g.sort_values("ts").copy()
        g["dt_s"] = g["ts"].diff().dt.total_seconds()
        g["dx"] = g["x"].diff()
        g["dy"] = g["y"].diff()
        g["step_m"] = np.sqrt(g["dx"]**2 + g["dy"]**2)
        g["speed_m_s"] = g["step_m"] / g["dt_s"]
        g["acceleration_m_s2"] = g["speed_m_s"].diff() / g["dt_s"]
        return g
    out = df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    MAX_SPEED = 10.0
    out = out[(out["speed_m_s"].isna()) | (out["speed_m_s"] <= MAX_SPEED)].copy()
    out["x_s"] = out.groupby("vehicle_id")["x"].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    out["y_s"] = out.groupby("vehicle_id")["y"].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    return out

def kpi_table(df):
    """Generate a KPI table summarising metrics per vehicle."""
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        sp = g["speed_m_s"].replace([np.inf, -np.inf], np.nan).dropna()
        kpi_dict = {
            'vehicle_id': vid,
            'points': len(g),
            'total_distance_m': float(g["step_m"].fillna(0).sum()),
            'avg_speed_m_s': float(sp[sp > 0].mean()) if not sp.empty else 0.0,
            'max_speed_m_s': float(sp.max()) if not sp.empty else 0.0,
            'min_speed_m_s': float(sp[sp > 0].min()) if not sp.empty else 0.0,
            'dwell_time_s': float(g["dt_s"].fillna(0)[(g["speed_m_s"].fillna(0) < 0.2)].sum()),
            'moving_time_s': float(g["dt_s"].fillna(0)[(g["speed_m_s"].fillna(0) >= 0.2)].sum()),
            'time_span_s': (g["ts"].max() - g["ts"].min()).total_seconds() if len(g) > 1 else 0.0,
            'start_time': g["ts"].min(),
            'end_time': g["ts"].max(),
            'utilization_%': 0.0
        }
        if kpi_dict['time_span_s'] > 0:
            kpi_dict['utilization_%'] = (kpi_dict['moving_time_s'] / kpi_dict['time_span_s']) * 100
        rows.append(kpi_dict)
    return pd.DataFrame(rows)

# ========= Plotting Functions (unchanged) =========
def plot_trajectory_with_layout(df, layout_svg=None, svg_handler=None):
    """
    Plot trajectories for each vehicle with an optional SVG layout as a
    background.  The smoothed coordinates are used when available.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    if svg_handler and svg_handler.elements:
        integrate_layout_with_plot(ax, df, svg_handler)
    elif layout_svg:
        # If a raw SVG string is provided but not parsed, do nothing.  Users
        # should rely on SVGLayoutHandler for proper scaling and rendering.
        pass
    colors = plt.cm.tab10(np.linspace(0, 1, df['vehicle_id'].nunique()))
    for idx, (vid, g) in enumerate(df.groupby("vehicle_id")):
        ax.plot(g["x_s"], g["y_s"], linewidth=2, alpha=0.7, label=f"{vid}", color=colors[idx])
        ax.scatter(g["x_s"].iloc[0], g["y_s"].iloc[0], marker='o', s=100, c='green', edgecolors='darkgreen', linewidth=2, zorder=5)
        ax.scatter(g["x_s"].iloc[-1], g["y_s"].iloc[-1], marker='s', s=100, c='red', edgecolors='darkred', linewidth=2, zorder=5)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title("Vehicle Trajectories", fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()
    return fig

def plot_heatmap_with_layout(df, layout_svg=None, svg_handler=None):
    """
    Create a 2D histogram (heatmap) of the positions, optionally overlaid with
    the plant layout.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if svg_handler and svg_handler.elements:
        integrate_layout_with_plot(ax, df, svg_handler)
    elif layout_svg:
        pass
    h = ax.hist2d(df["x_s"], df["y_s"], bins=[60, 60], cmap='YlOrRd', alpha=0.7)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title("Position Density Heatmap", fontsize=14, fontweight='bold')
    ax.axis("equal")
    plt.colorbar(h[3], ax=ax, label="Frequency")
    plt.tight_layout()
    return fig

def plot_speed_distribution(df):
    """
    Display a histogram of speeds and a time series plot of smoothed speeds for
    each vehicle.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sp = df["speed_m_s"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sp) > 0:
        ax1.hist(sp, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(sp.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sp.mean():.2f} m/s')
        ax1.set_xlabel("Speed (m/s)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Speed Distribution", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    for vid, g in df.groupby("vehicle_id"):
        ax2.plot(g["ts"], g["speed_m_s"].rolling(5, center=True, min_periods=1).mean(), label=vid, alpha=0.7)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Speed (m/s)", fontsize=12)
    ax2.set_title("Speed Over Time (Smoothed)", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_utilization_chart(kpis):
    """Plot utilization percentage and total distance for each vehicle."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.barh(kpis['vehicle_id'], kpis['utilization_%'], color='teal', alpha=0.7)
    ax1.set_xlabel("Utilization (%)", fontsize=12)
    ax1.set_ylabel("Vehicle ID", fontsize=12)
    ax1.set_title("Vehicle Utilization", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax2.bar(range(len(kpis)), kpis['total_distance_m'], color='coral', alpha=0.7)
    ax2.set_xticks(range(len(kpis)))
    ax2.set_xticklabels(kpis['vehicle_id'], rotation=45, ha='right')
    ax2.set_ylabel("Total Distance (m)", fontsize=12)
    ax2.set_title("Distance Traveled by Vehicle", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

# ========= Sidebar Configuration =========
with st.sidebar:
    st.header("⚙️ Configuration")
    # Data Source selection
    st.subheader("Data Source")
    source = st.radio("Select data source:", ["Snowflake", "CSV Upload"], index=0)
    st.markdown("---")
    if source == "Snowflake":
        st.subheader("Snowflake Credentials")
        # Input fields for credentials.  Use session state to persist values
        st.session_state.sf_cfg['account'] = st.text_input("Account", value=st.session_state.sf_cfg.get('account', ''))
        st.session_state.sf_cfg['user'] = st.text_input("Username", value=st.session_state.sf_cfg.get('user', ''))
        st.session_state.sf_cfg['password'] = st.text_input("Password", type="password", value=st.session_state.sf_cfg.get('password', ''))
        st.session_state.sf_cfg['role'] = st.text_input("Role", value=st.session_state.sf_cfg.get('role', 'SYSADMIN'))
        st.session_state.sf_cfg['warehouse'] = st.text_input("Warehouse", value=st.session_state.sf_cfg.get('warehouse', 'COMPUTE_WH'))
        st.session_state.sf_cfg['database'] = st.text_input("Database", value=st.session_state.sf_cfg.get('database', 'IOT'))
        st.session_state.sf_cfg['schema'] = st.text_input("Schema", value=st.session_state.sf_cfg.get('schema', 'PUBLIC'))
        st.session_state.table_name = st.text_input("Table Name", value=st.session_state.table_name)
        # Button to test the connection and fetch columns
        if st.button("🔌 Connect & Load Columns"):
            try:
                con = connect_snowflake(st.session_state.sf_cfg)
                # Retrieve column names from information_schema.  Use upper case table and schema names
                db = st.session_state.sf_cfg['database']
                sch = st.session_state.sf_cfg['schema']
                tbl = st.session_state.table_name
                query_cols = f"SELECT COLUMN_NAME FROM {db}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{sch.upper()}' AND TABLE_NAME = '{tbl.upper()}' ORDER BY ORDINAL_POSITION"
                cols_df = pd.read_sql(query_cols, con)
                con.close()
                if not cols_df.empty:
                    st.session_state.available_columns = cols_df['COLUMN_NAME'].tolist()
                    st.success(f"Loaded {len(st.session_state.available_columns)} columns from {tbl}")
                else:
                    st.session_state.available_columns = []
                    st.warning("No columns found for the specified table.")
            except Exception as e:
                st.session_state.available_columns = None
                st.error(f"Failed to connect or load columns: {str(e)}")
        # Column selection once columns are available
        if st.session_state.available_columns:
            st.subheader("Select Columns to Load")
            # Preselect the required columns; disable removal by always adding them in load_from_snowflake
            default_required = ['PAYLOAD_SERIALNUMBER', 'PAYLOAD_TIMESTAMP', 'PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y']
            preselect = [col for col in default_required if col in st.session_state.available_columns]
            st.session_state.selected_columns = st.multiselect("Columns", options=st.session_state.available_columns, default=preselect)
        # Query filters
        st.subheader("Query Filters")
        limit_rows = st.number_input(
            "Row limit (0 = all)", min_value=0, value=10000, step=1000, help="Limit the number of rows fetched from Snowflake"
        )
        use_date_filter = st.checkbox("Filter by date")
        date_filter = None
        if use_date_filter:
            date_filter = st.date_input("Select date")
    else:
        # CSV Upload configuration
        st.subheader("CSV Upload")
        st.info("Upload a CSV file with RTLS data in the expected format")
        limit_rows = 0
        date_filter = None
    st.markdown("---")
    # Layout settings
    st.subheader("🗺️ Layout Settings")
    use_layout = st.checkbox("Use plant layout", value=False)
    if use_layout:
        layout_file = st.file_uploader("Upload SVG layout file", type=['svg'], help="Upload an SVG file of your plant layout")
        if layout_file:
            st.session_state.layout_svg = layout_file.read().decode()
            try:
                st.session_state.svg_handler = SVGLayoutHandler(st.session_state.layout_svg)
                st.success("✅ Layout loaded and parsed")
            except Exception as e:
                st.session_state.svg_handler = None
                st.error(f"❌ Failed to parse SVG: {e}")
    st.markdown("---")
    # Analysis settings
    st.subheader("Analysis Settings")
    max_speed = st.slider("Max speed filter (m/s)", 5.0, 20.0, 10.0, 0.5)
    dwell_threshold = st.slider("Dwell threshold (m/s)", 0.1, 0.5, 0.2, 0.05)
    smoothing_window = st.slider("Smoothing window", 1, 10, 3, 1)
    st.markdown("---")
    # Run analysis button
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

# ========= Main Content Area =========
# Determine whether a CSV file was uploaded (for CSV mode)
if source == "CSV Upload":
    uploaded_file = st.file_uploader(
        "Upload RTLS CSV file", type=['csv'], help="Upload a CSV file with RTLS trajectory data"
    )
else:
    uploaded_file = None

if run_analysis or (source == "CSV Upload" and uploaded_file is not None):
    with st.spinner("Loading and processing data..."):
        try:
            # Load data according to the selected source
            if source == "Snowflake":
                if not st.session_state.selected_columns:
                    st.warning("Please connect to Snowflake and select columns before running the analysis.")
                    raise ValueError("No columns selected")
                # Build query using date filter if provided
                date_str = None
                if date_filter:
                    try:
                        date_str = date_filter.strftime("%Y-%m-%d")
                    except Exception:
                        date_str = str(date_filter)
                df_raw = load_from_snowflake(
                    st.session_state.sf_cfg,
                    st.session_state.table_name,
                    st.session_state.selected_columns,
                    limit_rows=limit_rows if limit_rows > 0 else None,
                    date_filter=date_str
                )
            else:
                df_raw = parse_uploaded_csv(uploaded_file)
            if df_raw.empty:
                st.warning("⚠️ No data found with the specified filters")
            else:
                # Process data
                df_proc = compute_metrics(df_raw)
                kpis = kpi_table(df_proc)
                st.session_state.data_loaded = True
                st.session_state.df_proc = df_proc
                st.session_state.kpis = kpis
                # Create tabs for views
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "KPIs", "Trajectories", "Heatmap", "Speed Analysis", "Export"])
                # Tab 1: Overview
                with tab1:
                    st.header("System Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Vehicles", len(kpis))
                    with col2:
                        st.metric("Total Points", f"{len(df_proc):,}")
                    with col3:
                        st.metric("Time Range", f"{(df_proc['ts'].max() - df_proc['ts'].min()).total_seconds() / 3600:.1f} hrs")
                    with col4:
                        st.metric("Avg Distance", f"{kpis['total_distance_m'].mean():.1f} m")
                    st.markdown("---")
                    st.subheader("Vehicle Summary")
                    summary_df = kpis[['vehicle_id', 'points', 'total_distance_m', 'avg_speed_m_s', 'utilization_%']].round(2)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                # Tab 2: Detailed KPIs
                with tab2:
                    st.header("Detailed KPIs Analysis")
                    st.subheader("Complete KPI Table")
                    st.dataframe(
                        kpis.round(2),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "utilization_%": st.column_config.ProgressColumn(
                                "Utilization %",
                                help="Percentage of time vehicle was moving",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        }
                    )
                    st.subheader("Utilization Analysis")
                    fig_util = plot_utilization_chart(kpis)
                    st.pyplot(fig_util)
                # Tab 3: Trajectory Visualisation
                with tab3:
                    st.header("Trajectory Visualisation")
                    selected_vehicles = st.multiselect(
                        "Select vehicles to display:",
                        options=df_proc['vehicle_id'].unique(),
                        default=df_proc['vehicle_id'].unique()
                    )
                    if selected_vehicles:
                        filtered_df = df_proc[df_proc['vehicle_id'].isin(selected_vehicles)]
                        fig_traj = plot_trajectory_with_layout(
                            filtered_df,
                            st.session_state.layout_svg if use_layout else None,
                            svg_handler=(st.session_state.svg_handler if use_layout else None)
                        )
                        st.pyplot(fig_traj)
                    else:
                        st.info("Select at least one vehicle to display trajectories")
                # Tab 4: Heatmap
                with tab4:
                    st.header("Position Heatmap")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        bins = st.slider("Heatmap resolution", 20, 100, 60, 10)
                    fig_heat = plot_heatmap_with_layout(
                        df_proc,
                        st.session_state.layout_svg if use_layout else None,
                        svg_handler=(st.session_state.svg_handler if use_layout else None)
                    )
                    st.pyplot(fig_heat)
                # Tab 5: Speed Analysis
                with tab5:
                    st.header("Speed Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Speed", f"{df_proc['speed_m_s'].mean():.2f} m/s")
                    with col2:
                        st.metric("Max Speed", f"{df_proc['speed_m_s'].max():.2f} m/s")
                    with col3:
                        st.metric("Vehicles Below Threshold", f"{(df_proc['speed_m_s'] < dwell_threshold).sum()}")
                    fig_speed = plot_speed_distribution(df_proc)
                    st.pyplot(fig_speed)
                # Tab 6: Data Export
                with tab6:
                    st.header("Data Export")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Processed Positions")
                        export_cols = ['vehicle_id', 'ts', 'x', 'y', 'x_s', 'y_s', 'speed_m_s', 'dt_s', 'step_m']
                        positions_csv = df_proc[export_cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Positions CSV",
                            data=positions_csv,
                            file_name=f"rtls_positions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    with col2:
                        st.subheader("KPI Summary")
                        kpis_csv = kpis.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download KPIs CSV",
                            data=kpis_csv,
                            file_name=f"rtls_kpis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    st.subheader("Export Preview")
                    preview_tab1, preview_tab2 = st.tabs(["Positions Preview", "KPIs Preview"])
                    with preview_tab1:
                        st.dataframe(df_proc[export_cols].head(100), use_container_width=True)
                    with preview_tab2:
                        st.dataframe(kpis, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.exception(e)
else:
    # Welcome screen when analysis has not yet been run
    st.info("Configure settings in the sidebar and click **Run Analysis** to start")
    with st.expander("How to use this dashboard"):
        st.markdown("""
        ### Quick Start Guide

        1. **Select Data Source**: Choose between Snowflake or CSV upload
        2. **Configure Settings**: Set filters, credentials and analysis parameters
        3. **Run Analysis**: Click the Run Analysis button
        4. **Explore Results**: Navigate through the different tabs

        ### Data Requirements

        Your data should contain the following columns:
        - `PAYLOAD_TIMESTAMP`: Timestamp of the position
        - `PAYLOAD_SERIALNUMBER`: Vehicle identifier
        - `PAYLOAD_AGVPOSITION_X`: X coordinate
        - `PAYLOAD_AGVPOSITION_Y`: Y coordinate
        - `PAYLOAD_AGVPOSITION_THETA`: Orientation angle

        ### Features

        - **Multi-tab Interface**: Organised views for different analyses
        - **SVG Layout Support**: Upload plant layout as background
        - **Real-time Metrics**: Live KPI calculations
        - **Export Functionality**: Download processed data and reports
        """
        )