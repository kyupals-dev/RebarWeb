"""
Phase 6: Dimension Calculation
Calculates physical dimensions from polygon data
"""

import numpy as np

class Phase06DimensionCalculation:
    """Phase 6: Calculate dimensions"""
    
    def __init__(self):
        self.pixel_to_cm_factor = 0.1  # Calibration factor
        print("📏 Phase 6: Dimension Calculation initialized")
    
    def calculate_dimensions(self, polygon_result):
        """Calculate physical dimensions from polygons"""
        try:
            print("📏 Phase 6: Calculating dimensions...")
            
            if not polygon_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Invalid polygon result'
                }
            
            polygons = polygon_result.get('polygons', [])
            
            if len(polygons) == 0:
                # Default dimensions if no polygons
                dimensions = self._get_default_dimensions()
            else:
                # Calculate from largest polygon
                largest_polygon = max(polygons, key=lambda p: p.get('area', 0))
                dimensions = self._calculate_from_polygon(largest_polygon)
            
            print(f"   ✅ Phase 6: Calculated dimensions: {dimensions['display']}")
            
            return {
                'success': True,
                'dimensions': dimensions,
                'calculation_method': 'polygon_based' if len(polygons) > 0 else 'default'
            }
            
        except Exception as e:
            print(f"   ❌ Phase 6 error: {str(e)}")
            return {
                'success': False,
                'error': f'Dimension calculation failed: {str(e)}'
            }
    
    def _calculate_from_polygon(self, polygon):
        """Calculate dimensions from a polygon"""
        try:
            vertices = polygon.get('vertices', [])
            
            if len(vertices) < 4:
                return self._get_default_dimensions()
            
            # Calculate bounding box
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]
            
            width_px = max(x_coords) - min(x_coords)
            height_px = max(y_coords) - min(y_coords)
            
            # Convert to centimeters
            width_cm = max(width_px * self.pixel_to_cm_factor, 10)
            height_cm = max(height_px * self.pixel_to_cm_factor, 10)
            
            # Assume square cross-section for now
            length_cm = max(width_cm, height_cm)
            
            # Standard rebar height
            depth_cm = 200.0
            
            volume_cm3 = length_cm * width_cm * depth_cm
            
            return {
                'length': round(length_cm, 1),
                'width': round(width_cm, 1),
                'height': round(depth_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': f"{length_cm:.0f}cm x {width_cm:.0f}cm x {depth_cm:.0f}cm = {volume_cm3:.0f}cm³",
                'method': 'polygon_analysis'
            }
            
        except Exception as e:
            print(f"   ⚠️ Polygon calculation error: {e}")
            return self._get_default_dimensions()
    
    def _get_default_dimensions(self):
        """Get default dimensions"""
        return {
            'length': 25.4,
            'width': 25.4,
            'height': 200.0,
            'unit': 'cm',
            'volume': 101600,
            'display': '25cm x 25cm x 200cm = 101600cm³',
            'method': 'default_fallback'
        }
