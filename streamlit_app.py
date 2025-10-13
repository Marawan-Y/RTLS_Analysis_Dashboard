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

st.set_page_config(
    page_title="RTLS Trajectory Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

st.title("RTLS Trajectory Analytics Dashboard")
st.markdown("---")

# ========= Session State Management =========
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_proc = None
    st.session_state.kpis = None
    st.session_state.layout_svg = None
    # (SVG handler)
    st.session_state.svg_handler = None

# ========= SVG Layout Handler (merged) =========
class SVGLayoutHandler:
    """
    Handler for SVG plant layouts to be used as backgrounds in trajectory plots
    """
    
    def __init__(self, svg_content=None):
        """
        Initialize SVG handler
        
        Args:
            svg_content (str): SVG content as string
        """
        self.svg_content = svg_content
        self.elements = []
        self.bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
        
        if svg_content:
            self.parse_svg()
    
    def parse_svg(self):
        """
        Parse SVG content and extract drawable elements
        """
        if not self.svg_content:
            return
        
        try:
            root = ET.fromstring(self.svg_content)
            
            # Get SVG viewBox or width/height
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
            
            # Parse elements recursively
            self._parse_element(root)
            
        except Exception as e:
            print(f"Error parsing SVG: {e}")
    
    def _parse_element(self, element, transform=None):
        """
        Recursively parse SVG elements
        """
        # Handle groups
        if element.tag.endswith('g'):
            group_transform = element.get('transform')
            for child in element:
                self._parse_element(child, group_transform)
        
        # Handle rectangles
        elif element.tag.endswith('rect'):
            self._parse_rect(element, transform)
        
        # Handle circles
        elif element.tag.endswith('circle'):
            self._parse_circle(element, transform)
        
        # Handle ellipses
        elif element.tag.endswith('ellipse'):
            self._parse_ellipse(element, transform)
        
        # Handle lines
        elif element.tag.endswith('line'):
            self._parse_line(element, transform)
        
        # Handle polylines
        elif element.tag.endswith('polyline'):
            self._parse_polyline(element, transform)
        
        # Handle polygons
        elif element.tag.endswith('polygon'):
            self._parse_polygon(element, transform)
        
        # Handle paths
        elif element.tag.endswith('path'):
            self._parse_path(element, transform)
        
        # Recursively parse children
        for child in element:
            self._parse_element(child, transform)
    
    def _parse_rect(self, element, transform):
        """Parse rectangle element"""
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
        except:
            pass
    
    def _parse_circle(self, element, transform):
        """Parse circle element"""
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
        except:
            pass
    
    def _parse_ellipse(self, element, transform):
        """Parse ellipse element"""
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
        except:
            pass
    
    def _parse_line(self, element, transform):
        """Parse line element"""
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
        except:
            pass
    
    def _parse_polyline(self, element, transform):
        """Parse polyline element"""
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
        except:
            pass
    
    def _parse_polygon(self, element, transform):
        """Parse polygon element"""
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
        except:
            pass
    
    def _parse_path(self, element, transform):
        """Parse path element (simplified)"""
        try:
            d = element.get('d', '')
            style = self._parse_style(element)
            
            # For complex paths, we'll create a simplified representation
            # In production, you'd use a proper SVG path parser
            self.elements.append({
                'type': 'path',
                'd': d,
                'style': style,
                'transform': transform
            })
        except:
            pass
    
    def _parse_points(self, points_str):
        """Parse points string into list of coordinates"""
        points = []
        parts = points_str.replace(',', ' ').split()
        for i in range(0, len(parts)-1, 2):
            try:
                x = float(parts[i])
                y = float(parts[i+1])
                points.append((x, y))
            except:
                pass
        return points
    
    def _parse_style(self, element):
        """Parse style attributes"""
        style = {
            'fill': element.get('fill', 'none'),
            'stroke': element.get('stroke', 'black'),
            'stroke_width': float(element.get('stroke-width', 1)),
            'opacity': float(element.get('opacity', 1.0)),
            'fill_opacity': float(element.get('fill-opacity', 1.0)),
            'stroke_opacity': float(element.get('stroke-opacity', 1.0))
        }
        
        # Parse style attribute if present
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
                        except:
                            pass
                    elif key == 'opacity':
                        try:
                            style['opacity'] = float(value)
                        except:
                            pass
        
        return style
    
    def render_on_axes(self, ax, scale_to_data=None, alpha=0.3):
        """
        Render SVG elements on matplotlib axes
        
        Args:
            ax: Matplotlib axes object
            scale_to_data: Tuple of (x_range, y_range) to scale SVG to data coordinates
            alpha: Overall transparency for the layout
        """
        if not self.elements:
            return
        
        # Calculate scaling factors if needed
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
                
                # Use uniform scaling to maintain aspect ratio
                scale = min(scale_x, scale_y)
                scale_x = scale_y = scale
                
                offset_x = x_range[0]
                offset_y = y_range[0]
        
        # Render each element
        for elem in self.elements:
            self._render_element(ax, elem, scale_x, scale_y, offset_x, offset_y, alpha)
    
    def _render_element(self, ax, elem, scale_x, scale_y, offset_x, offset_y, alpha):
        """Render individual SVG element on axes"""
        elem_type = elem['type']
        style = elem['style']
        
        # Prepare colors
        fill_color = style['fill'] if style['fill'] != 'none' else None
        edge_color = style['stroke'] if style['stroke'] != 'none' else None
        linewidth = style['stroke_width'] * min(scale_x, scale_y)
        
        # Apply overall alpha
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
            points = elem['points']
            if points:
                x_coords = [p[0] * scale_x + offset_x for p in points]
                y_coords = [p[1] * scale_y + offset_y for p in points]
                ax.plot(
                    x_coords, y_coords,
                    color=edge_color if edge_color else 'black',
                    linewidth=linewidth,
                    alpha=elem_alpha
                )
        
        elif elem_type == 'polygon':
            points = elem['points']
            if points:
                scaled_points = [(p[0] * scale_x + offset_x, p[1] * scale_y + offset_y) for p in points]
                polygon = patches.Polygon(
                    scaled_points,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=linewidth,
                    alpha=elem_alpha
                )
                ax.add_patch(polygon)

def integrate_layout_with_plot(ax, df, svg_handler):
    """
    Helper function to integrate SVG layout with trajectory plot
    
    Args:
        ax: Matplotlib axes
        df: DataFrame with trajectory data
        svg_handler: SVGLayoutHandler instance
    """
    if svg_handler and svg_handler.elements:
        # Prefer smoothed columns if present
        x_col = "x_s" if "x_s" in df.columns else "x"
        y_col = "y_s" if "y_s" in df.columns else "y"

        # Get data ranges
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 1.0
        y_padding = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1.0
        
        x_range = (x_min - x_padding, x_max + x_padding)
        y_range = (y_min - y_padding, y_max + y_padding)
        
        # Render SVG on axes
        svg_handler.render_on_axes(ax, scale_to_data=(x_range, y_range), alpha=0.3)

# ========= Helper Functions =========
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

def load_from_snowflake(limit_rows=None, date_filter=None):
    """
    Load data from Snowflake with the exact table structure as provided
    """
    con = connect_snowflake()
    
    # Build query based on the actual table structure
    base_query = f"""
    SELECT 
        PAYLOAD_SERIALNUMBER as vehicle_id,
        PAYLOAD_TIMESTAMP as ts,
        PAYLOAD_AGVPOSITION_X as x,
        PAYLOAD_AGVPOSITION_Y as y,
        PAYLOAD_AGVPOSITION_THETA as theta,
        PAYLOAD_AGVPOSITION_MAPID as map_id,
        PAYLOAD_AGVPOSITION_POSITIONINITIALIZED as position_initialized,
        PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE as localization_score,
        PAYLOAD_VELOCITY_VX as velocity_x,
        PAYLOAD_VELOCITY_VY as velocity_y,
        PAYLOAD_VELOCITY_OMEGA as velocity_omega,
        MQTT_TIMESTAMP_ISO_8601 as mqtt_timestamp,
        MQTT_TOPIC as topic
    FROM {SF_CFG['staging_table']}
    WHERE PAYLOAD_SERIALNUMBER IS NOT NULL 
        AND PAYLOAD_TIMESTAMP IS NOT NULL 
        AND PAYLOAD_AGVPOSITION_X IS NOT NULL 
        AND PAYLOAD_AGVPOSITION_Y IS NOT NULL
    """
    
    if date_filter:
        base_query += f" AND DATE(PAYLOAD_TIMESTAMP) = '{date_filter}'"
    
    base_query += " ORDER BY PAYLOAD_SERIALNUMBER, PAYLOAD_TIMESTAMP"
    
    if limit_rows and limit_rows > 0:
        base_query += f" LIMIT {int(limit_rows)}"
    
    df = pd.read_sql(base_query, con)
    con.close()
    return df

def parse_uploaded_csv(file):
    """
    Parse uploaded CSV with the exact structure as provided
    """
    try:
        df_raw = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(io.BytesIO(file.read()), encoding_errors="ignore")
    
    # Map columns based on the provided CSV structure
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
    
    # Check if required columns exist
    required_cols = ['PAYLOAD_TIMESTAMP', 'PAYLOAD_SERIALNUMBER', 
                     'PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y']
    
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Create clean dataframe with renamed columns
    df_clean = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df_raw.columns:
            if old_col == 'PAYLOAD_TIMESTAMP':
                df_clean[new_col] = pd.to_datetime(df_raw[old_col], errors="coerce")
            elif old_col in ['PAYLOAD_AGVPOSITION_X', 'PAYLOAD_AGVPOSITION_Y', 
                            'PAYLOAD_AGVPOSITION_THETA', 'PAYLOAD_AGVPOSITION_LOCALIZATIONSCORE',
                            'PAYLOAD_VELOCITY_VX', 'PAYLOAD_VELOCITY_VY', 'PAYLOAD_VELOCITY_OMEGA']:
                df_clean[new_col] = pd.to_numeric(df_raw[old_col], errors="coerce")
            elif old_col == 'PAYLOAD_AGVPOSITION_POSITIONINITIALIZED':
                df_clean[new_col] = df_raw[old_col].astype(bool)
            else:
                df_clean[new_col] = df_raw[old_col].astype(str)
    
    # Drop rows with missing essential data
    df_clean = df_clean.dropna(subset=["ts", "x", "y", "vehicle_id"])
    
    return df_clean.sort_values(["vehicle_id", "ts"]).reset_index(drop=True)

def compute_metrics(df):
    """Enhanced metrics computation"""
    def per_vehicle(g):
        g = g.sort_values("ts").copy()
        g["dt_s"] = g["ts"].diff().dt.total_seconds()
        g["dx"] = g["x"].diff()
        g["dy"] = g["y"].diff()
        g["step_m"] = np.sqrt(g["dx"]**2 + g["dy"]**2)
        g["speed_m_s"] = g["step_m"] / g["dt_s"]
        
        # Add acceleration
        g["acceleration_m_s2"] = g["speed_m_s"].diff() / g["dt_s"]
        
        return g
    
    out = df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    
    # Filter unrealistic teleports
    MAX_SPEED = 10.0
    out = out[(out["speed_m_s"].isna()) | (out["speed_m_s"] <= MAX_SPEED)].copy()
    
    # Smoothing
    out["x_s"] = out.groupby("vehicle_id")["x"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).median()
    )
    out["y_s"] = out.groupby("vehicle_id")["y"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).median()
    )
    
    return out

def kpi_table(df):
    """Generate comprehensive KPI table"""
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        sp = g["speed_m_s"].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate various KPIs
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
            'utilization_%': 0.0  # Will calculate below
        }
        
        # Calculate utilization
        if kpi_dict['time_span_s'] > 0:
            kpi_dict['utilization_%'] = (kpi_dict['moving_time_s'] / kpi_dict['time_span_s']) * 100
        
        rows.append(kpi_dict)
    
    return pd.DataFrame(rows)

# ======== UPDATED: use real SVG handler instead of placeholder ========
def plot_trajectory_with_layout(df, layout_svg=None, svg_handler=None):
    """Plot trajectory with optional SVG layout background"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Render real SVG if provided via handler
    if svg_handler and svg_handler.elements:
        integrate_layout_with_plot(ax, df, svg_handler)
    elif layout_svg:
        # (kept compatibility: if only raw string is present, no-op to avoid placeholder)
        pass
    
    # Plot trajectories for each vehicle
    colors = plt.cm.tab10(np.linspace(0, 1, df['vehicle_id'].nunique()))
    
    for idx, (vid, g) in enumerate(df.groupby("vehicle_id")):
        ax.plot(g["x_s"], g["y_s"], linewidth=2, alpha=0.7, 
                label=f"{vid}", color=colors[idx])
        
        # Mark start and end points
        ax.scatter(g["x_s"].iloc[0], g["y_s"].iloc[0], 
                  marker='o', s=100, c='green', edgecolors='darkgreen', 
                  linewidth=2, zorder=5)
        ax.scatter(g["x_s"].iloc[-1], g["y_s"].iloc[-1], 
                  marker='s', s=100, c='red', edgecolors='darkred', 
                  linewidth=2, zorder=5)
    
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title("Vehicle Trajectories", fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    
    plt.tight_layout()
    return fig

def plot_heatmap_with_layout(df, layout_svg=None, svg_handler=None):
    """Plot position heatmap with optional SVG layout background"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Render real SVG if provided via handler
    if svg_handler and svg_handler.elements:
        integrate_layout_with_plot(ax, df, svg_handler)
    elif layout_svg:
        # (kept compatibility: if only raw string is present, no-op to avoid placeholder)
        pass
    
    # Create heatmap
    h = ax.hist2d(df["x_s"], df["y_s"], bins=[60, 60], cmap='YlOrRd', alpha=0.7)
    
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title("Position Density Heatmap", fontsize=14, fontweight='bold')
    ax.axis("equal")
    
    cbar = plt.colorbar(h[3], ax=ax, label="Frequency")
    
    plt.tight_layout()
    return fig
# =====================================================================

def plot_speed_distribution(df):
    """Enhanced speed distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Speed histogram
    sp = df["speed_m_s"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sp) > 0:
        ax1.hist(sp, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(sp.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sp.mean():.2f} m/s')
        ax1.set_xlabel("Speed (m/s)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Speed Distribution", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Speed over time
    for vid, g in df.groupby("vehicle_id"):
        ax2.plot(g["ts"], g["speed_m_s"].rolling(5, center=True, min_periods=1).mean(), 
                label=vid, alpha=0.7)
    
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Speed (m/s)", fontsize=12)
    ax2.set_title("Speed Over Time (Smoothed)", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_utilization_chart(kpis):
    """Plot vehicle utilization chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Utilization bar chart
    ax1.barh(kpis['vehicle_id'], kpis['utilization_%'], color='teal', alpha=0.7)
    ax1.set_xlabel("Utilization (%)", fontsize=12)
    ax1.set_ylabel("Vehicle ID", fontsize=12)
    ax1.set_title("Vehicle Utilization", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Distance comparison
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
    
    # Data Source Selection
    st.subheader("Data Source")
    source = st.radio("Select data source:", ["Snowflake", "CSV Upload"], index=0)
    
    st.markdown("---")
    
    # Snowflake Configuration
    if source == "Snowflake":
        st.subheader("Snowflake Settings")
        
        # Connection test
        if st.button("🔌 Test Connection", type="primary"):
            try:
                con = connect_snowflake()
                cur = con.cursor()
                cur.execute("SELECT CURRENT_VERSION()")
                ver = cur.fetchone()[0]
                st.success(f"✅ Connected! Version: {ver}")
                
                # Show available tables
                cur.execute(f"SHOW TABLES IN {SF_CFG['database']}.{SF_CFG['schema']}")
                tables = cur.fetchall()
                if tables:
                    st.info(f"Found {len(tables)} tables in schema")
                
                cur.close()
                con.close()
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        
        # Query filters
        st.subheader("Query Filters")
        limit_rows = st.number_input(
            "Row limit (0 = all)", 
            min_value=0, 
            value=10000, 
            step=1000,
            help="Limit the number of rows fetched from Snowflake"
        )
        
        use_date_filter = st.checkbox("Filter by date")
        date_filter = None
        if use_date_filter:
            date_filter = st.date_input("Select date")
    
    # CSV Upload Configuration
    else:
        st.subheader("CSV Upload")
        st.info("Upload a CSV file with RTLS data in the expected format")
        limit_rows = 0
    
    st.markdown("---")
    
    # Layout Configuration
    st.subheader("🗺️ Layout Settings")
    use_layout = st.checkbox("Use plant layout", value=False)
    if use_layout:
        layout_file = st.file_uploader(
            "Upload SVG layout file", 
            type=['svg'],
            help="Upload an SVG file of your plant layout"
        )
        if layout_file:
            st.session_state.layout_svg = layout_file.read().decode()
            # Build/parse handler immediately
            try:
                st.session_state.svg_handler = SVGLayoutHandler(st.session_state.layout_svg)
                st.success("✅ Layout loaded and parsed")
            except Exception as e:
                st.session_state.svg_handler = None
                st.error(f"❌ Failed to parse SVG: {e}")
    
    st.markdown("---")
    
    # Analysis Settings
    st.subheader("Analysis Settings")
    max_speed = st.slider("Max speed filter (m/s)", 5.0, 20.0, 10.0, 0.5)
    dwell_threshold = st.slider("Dwell threshold (m/s)", 0.1, 0.5, 0.2, 0.05)
    smoothing_window = st.slider("Smoothing window", 1, 10, 3, 1)
    
    st.markdown("---")
    
    # Run Analysis Button
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

# ========= Main Content Area =========
if source == "CSV Upload":
    uploaded_file = st.file_uploader(
        "Upload RTLS CSV file", 
        type=['csv'],
        help="Upload a CSV file with RTLS trajectory data"
    )
else:
    uploaded_file = None

# Main analysis workflow
if run_analysis or (source == "CSV Upload" and uploaded_file is not None):
    with st.spinner("Loading and processing data..."):
        try:
            # Load data based on source
            if source == "Snowflake":
                df_raw = load_from_snowflake(
                    limit_rows=limit_rows if limit_rows > 0 else None,
                    date_filter=date_filter if use_date_filter else None
                )
            else:
                df_raw = parse_uploaded_csv(uploaded_file)
            
            if df_raw.empty:
                st.warning("⚠️ No data found with the specified filters")
            else:
                # Process data
                df_proc = compute_metrics(df_raw)
                kpis = kpi_table(df_proc)
                
                # Store in session state
                st.session_state.data_loaded = True
                st.session_state.df_proc = df_proc
                st.session_state.kpis = kpis
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Overview", 
                    "KPIs", 
                    "Trajectories", 
                    "Heatmap", 
                    "Speed Analysis",
                    "Export"
                ])
                
                # Tab 1: Overview
                with tab1:
                    st.header("System Overview")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Vehicles", len(kpis))
                    with col2:
                        st.metric("Total Points", f"{len(df_proc):,}")
                    with col3:
                        st.metric("Time Range", 
                                 f"{(df_proc['ts'].max() - df_proc['ts'].min()).total_seconds()/3600:.1f} hrs")
                    with col4:
                        st.metric("Avg Distance", f"{kpis['total_distance_m'].mean():.1f} m")
                    
                    st.markdown("---")
                    
                    # Quick stats per vehicle
                    st.subheader("Vehicle Summary")
                    summary_df = kpis[['vehicle_id', 'points', 'total_distance_m', 
                                      'avg_speed_m_s', 'utilization_%']].round(2)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Tab 2: Detailed KPIs
                with tab2:
                    st.header("Detailed KPIs Analysis")
                    
                    # Full KPI table
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
                    
                    # Utilization charts
                    st.subheader("Utilization Analysis")
                    fig_util = plot_utilization_chart(kpis)
                    st.pyplot(fig_util)
                
                # Tab 3: Trajectory Visualization
                with tab3:
                    st.header("Trajectory Visualization")
                    
                    # Vehicle selector
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
                    
                    # Heatmap settings
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
                    
                    # Speed statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Speed", 
                                 f"{df_proc['speed_m_s'].mean():.2f} m/s")
                    with col2:
                        st.metric("Max Speed", 
                                 f"{df_proc['speed_m_s'].max():.2f} m/s")
                    with col3:
                        st.metric("Vehicles Below Threshold", 
                                 f"{(df_proc['speed_m_s'] < dwell_threshold).sum()}")
                    
                    # Speed plots
                    fig_speed = plot_speed_distribution(df_proc)
                    st.pyplot(fig_speed)
                
                # Tab 6: Data Export
                with tab6:
                    st.header("Data Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Processed Positions")
                        export_cols = ['vehicle_id', 'ts', 'x', 'y', 'x_s', 'y_s', 
                                      'speed_m_s', 'dt_s', 'step_m']
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
                    
                    # Preview of export data
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
    # Welcome screen
    st.info("Configure settings in the sidebar and click **Run Analysis** to start")
    
    # Display instructions
    with st.expander("How to use this dashboard"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Select Data Source**: Choose between Snowflake or CSV upload
        2. **Configure Settings**: Set filters and analysis parameters
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
        
        - **Multi-tab Interface**: Organized views for different analyses
        - **SVG Layout Support**: Upload plant layout as background
        - **Real-time Metrics**: Live KPI calculations
        - **Export Functionality**: Download processed data and reports
        """)
