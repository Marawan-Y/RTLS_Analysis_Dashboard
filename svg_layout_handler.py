"""
SVG Layout Handler Module for RTLS Dashboard
Handles parsing and rendering of SVG plant layouts as matplotlib backgrounds
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import re

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
        # Get data ranges
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_range = (x_min - x_padding, x_max + x_padding)
        y_range = (y_min - y_padding, y_max + y_padding)
        
        # Render SVG on axes
        svg_handler.render_on_axes(ax, scale_to_data=(x_range, y_range), alpha=0.3)