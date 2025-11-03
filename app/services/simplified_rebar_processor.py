"""
Simplified Rebar Detection Processor
app/services/simplified_rebar_processor.py

Focuses on detecting 2 front_vertical + 11 front_horizontal rebars
with intersection verification and 4.5cm offset calculation
"""

import numpy as np
import cv2

class SimplifiedRebarProcessor:
    """
    Simplified processor for rebar detection and measurement
    
    Targets:
    - 2 front_vertical rebars
    - 11 front_horizontal rebars  
    - Valid intersections between them
    - 4.5cm offset for square column calculation
    """
    
    def __init__(self):
        self.offset_cm = 4.5  # 4.5cm offset for each side
        self.target_vertical = 2
        self.target_horizontal = 11
        
        # Validation thresholds
        self.min_intersection_area = 20  # Minimum pixels for intersection
        self.strong_intersection_area = 50  # Strong intersection threshold
        self.min_intersections_required = 5  # Minimum intersections to validate structure
        
        # Pixel to cm conversion (calibrated for 160-200cm distance)
        self.pixel_to_cm_factor = 0.12
        
        print("🎯 Simplified Rebar Processor initialized")
        print(f"   Target: {self.target_vertical} vertical + {self.target_horizontal} horizontal")
        print(f"   Offset: {self.offset_cm}cm each side")
        
    def process_detections(self, instances, image_shape):
        """
        Process Detectron2 outputs into simplified rebar measurements
        
        Args:
            instances: Detectron2 instances output
            image_shape: (height, width, channels)
            
        Returns:
            dict: Processed results with dimensions and cement mixture
        """
        try:
            print("🔄 Processing detections with simplified logic...")
            
            # Extract detection data
            if len(instances) == 0:
                return {'success': False, 'error': 'No detections found'}
            
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            print(f"   📊 Raw detections: {len(instances)} total")
            
            # Filter for target classes
            detections = self._filter_detections(boxes, scores, classes, masks)
            
            if not detections['success']:
                return detections
            
            # Verify intersections
            intersections = self._find_intersections(detections['vertical'], detections['horizontal'])
            
            if len(intersections) < self.min_intersections_required:
                return {
                    'success': False,
                    'error': f'Only {len(intersections)} intersections found, expected at least {self.min_intersections_required}'
                }
            
            # Calculate dimensions
            dimensions = self._calculate_dimensions(
                detections['vertical'], 
                detections['horizontal'], 
                intersections, 
                image_shape
            )
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            # Create summary
            result = {
                'success': True,
                'vertical_count': len(detections['vertical']),
                'horizontal_count': len(detections['horizontal']),
                'intersection_count': len(intersections),
                'valid_intersections': len([i for i in intersections if i['is_valid']]),
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'detections_summary': {
                    'vertical_rebars': detections['vertical'],
                    'horizontal_rebars': detections['horizontal'],
                    'intersections': intersections
                },
                'target_achievement': {
                    'vertical_target': self.target_vertical,
                    'vertical_actual': len(detections['vertical']),
                    'horizontal_target': self.target_horizontal,
                    'horizontal_actual': len(detections['horizontal']),
                    'target_met': (
                        len(detections['vertical']) == self.target_vertical and 
                        len(detections['horizontal']) == self.target_horizontal
                    )
                }
            }
            
            print(f"✅ Processing complete:")
            print(f"   🟢 {result['vertical_count']} vertical (target: {self.target_vertical})")
            print(f"   🔴 {result['horizontal_count']} horizontal (target: {self.target_horizontal})")
            print(f"   🟡 {result['valid_intersections']} valid intersections")
            print(f"   🎯 Target achieved: {result['target_achievement']['target_met']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            return {'success': False, 'error': f'Processing failed: {str(e)}'}
    
    def _filter_detections(self, boxes, scores, classes, masks):
        """Filter detections for front_vertical and front_horizontal only"""
        class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        
        vertical_rebars = []
        horizontal_rebars = []
        
        print("   🔍 Filtering detections by class...")
        
        for i, (box, score, cls, mask) in enumerate(zip(boxes, scores, classes, masks)):
            class_name = class_names[cls]
            confidence = float(score)
            
            detection = {
                'id': i,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': box.tolist(),
                'mask': mask,
                'centroid': self._get_centroid(mask),
                'mask_area': float(np.sum(mask))
            }
            
            if class_name == 'front_vertical':
                vertical_rebars.append(detection)
                print(f"     ✅ V{len(vertical_rebars)}: confidence {confidence:.3f}")
            elif class_name == 'front_horizontal':
                horizontal_rebars.append(detection)
                print(f"     ✅ H{len(horizontal_rebars)}: confidence {confidence:.3f}")
            else:
                print(f"     ⚪ Ignored {class_name}: confidence {confidence:.3f}")
        
        # Validate counts with some flexibility
        vertical_ok = 1 <= len(vertical_rebars) <= 3  # Allow 1-3 vertical
        horizontal_ok = 8 <= len(horizontal_rebars) <= 15  # Allow 8-15 horizontal
        
        if not vertical_ok:
            return {
                'success': False,
                'error': f'Expected ~{self.target_vertical} vertical rebars, found {len(vertical_rebars)} (range: 1-3 acceptable)'
            }
        
        if not horizontal_ok:
            return {
                'success': False,
                'error': f'Expected ~{self.target_horizontal} horizontal rebars, found {len(horizontal_rebars)} (range: 8-15 acceptable)'
            }
        
        print(f"   ✅ Validation passed: {len(vertical_rebars)} vertical, {len(horizontal_rebars)} horizontal")
        
        return {
            'success': True,
            'vertical': vertical_rebars,
            'horizontal': horizontal_rebars
        }
    
    def _get_centroid(self, mask):
        """Get centroid of binary mask using OpenCV moments"""
        try:
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return (0, 0)
        except Exception as e:
            print(f"   ⚠️ Centroid calculation error: {e}")
            return (0, 0)
    
    def _find_intersections(self, vertical_rebars, horizontal_rebars):
        """Find intersections between vertical and horizontal rebars"""
        print("   🔗 Finding intersections...")
        
        intersections = []
        
        for v_idx, vertical in enumerate(vertical_rebars):
            for h_idx, horizontal in enumerate(horizontal_rebars):
                # Calculate mask intersection
                intersection_mask = np.logical_and(vertical['mask'], horizontal['mask'])
                intersection_area = np.sum(intersection_mask)
                
                if intersection_area > self.min_intersection_area:
                    intersection = {
                        'vertical_id': v_idx,
                        'horizontal_id': h_idx,
                        'vertical_confidence': vertical['confidence'],
                        'horizontal_confidence': horizontal['confidence'],
                        'area': float(intersection_area),
                        'centroid': self._get_centroid(intersection_mask),
                        'is_valid': intersection_area > self.strong_intersection_area,
                        'strength': 'strong' if intersection_area > self.strong_intersection_area else 'weak'
                    }
                    intersections.append(intersection)
        
        # Sort by area (largest intersections first)
        intersections.sort(key=lambda x: x['area'], reverse=True)
        
        valid_count = len([i for i in intersections if i['is_valid']])
        print(f"   🔗 Found {len(intersections)} total intersections ({valid_count} strong)")
        
        # Log top intersections
        for i, intersection in enumerate(intersections[:5]):  # Show top 5
            strength = intersection['strength']
            area = intersection['area']
            v_id = intersection['vertical_id']
            h_id = intersection['horizontal_id']
            print(f"     {i+1}. V{v_id+1} × H{h_id+1}: {area:.0f}px ({strength})")
        
        return intersections
    
    def _calculate_dimensions(self, vertical_rebars, horizontal_rebars, intersections, image_shape):
        """Calculate rebar column dimensions with 4.5cm offset"""
        try:
            print("📏 Calculating dimensions with offset...")
            
            height, width, _ = image_shape
            
            # Calculate width from vertical rebar spacing
            if len(vertical_rebars) >= 2:
                # Sort verticals by x-position (left to right)
                verticals_sorted = sorted(vertical_rebars, key=lambda v: v['centroid'][0])
                left_vertical = verticals_sorted[0]
                right_vertical = verticals_sorted[-1]
                
                # Distance between outer edges of vertical rebars
                left_edge = left_vertical['bbox'][0]   # x1 of leftmost rebar
                right_edge = right_vertical['bbox'][2]  # x2 of rightmost rebar
                
                rebar_span_px = right_edge - left_edge
                print(f"   📐 Width calculation: V1 to V{len(verticals_sorted)} = {rebar_span_px:.1f}px")
            else:
                # Single vertical - estimate from bbox width
                if vertical_rebars:
                    bbox = vertical_rebars[0]['bbox']
                    rebar_span_px = bbox[2] - bbox[0]  # x2 - x1
                    print(f"   📐 Width calculation: Single vertical bbox = {rebar_span_px:.1f}px")
                else:
                    rebar_span_px = width * 0.3  # 30% of image width
                    print(f"   📐 Width calculation: Fallback = {rebar_span_px:.1f}px")
            
            # Calculate height from horizontal rebar spacing
            if len(horizontal_rebars) >= 2:
                # Sort horizontals by y-position (top to bottom)
                horizontals_sorted = sorted(horizontal_rebars, key=lambda h: h['centroid'][1])
                top_horizontal = horizontals_sorted[0]
                bottom_horizontal = horizontals_sorted[-1]
                
                # Distance between outer edges of horizontal rebars
                top_edge = top_horizontal['bbox'][1]    # y1 of topmost rebar
                bottom_edge = bottom_horizontal['bbox'][3]  # y2 of bottommost rebar
                
                rebar_height_px = bottom_edge - top_edge
                print(f"   📏 Height calculation: H1 to H{len(horizontals_sorted)} = {rebar_height_px:.1f}px")
            else:
                # Estimate from image height
                rebar_height_px = height * 0.8  # 80% of image height
                print(f"   📏 Height calculation: Fallback = {rebar_height_px:.1f}px")
            
            # Convert pixels to centimeters
            internal_width_cm = rebar_span_px * self.pixel_to_cm_factor
            internal_height_cm = rebar_height_px * self.pixel_to_cm_factor
            
            print(f"   🔄 Pixel to cm conversion (factor: {self.pixel_to_cm_factor}):")
            print(f"      Internal width: {internal_width_cm:.1f}cm")
            print(f"      Internal height: {internal_height_cm:.1f}cm")
            
            # Add offset for square column dimensions
            column_width_cm = internal_width_cm + (2 * self.offset_cm)
            column_length_cm = column_width_cm  # Square column assumption
            column_height_cm = internal_height_cm  # Height doesn't get offset
            
            # Ensure minimum realistic dimensions
            column_width_cm = max(column_width_cm, 15.0)
            column_length_cm = max(column_length_cm, 15.0)
            column_height_cm = max(column_height_cm, 50.0)
            
            # Calculate volume
            volume_cm3 = column_length_cm * column_width_cm * column_height_cm
            volume_m3 = volume_cm3 / 1_000_000
            
            # Create display string
            display_string =  (f"{column_length_cm:.1f}cm x {column_width_cm:.1f}cm x "
                            f"{column_height_cm:.1f}cm = {volume_cm3:.0f}cm³ = {volume_m3:.6f}m³")
            
            result = {
                'length': round(column_length_cm, 1),
                'width': round(column_width_cm, 1),
                'height': round(column_height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'volume_m3': round(volume_m3, 6),
                'display': display_string,
                'method': 'intersection_based_with_offset',
                'offset_applied': self.offset_cm,
                'internal_dimensions': {
                    'width_cm': round(internal_width_cm, 1),
                    'height_cm': round(internal_height_cm, 1)
                },
                'pixel_measurements': {
                    'width_px': round(rebar_span_px, 1),
                    'height_px': round(rebar_height_px, 1),
                    'conversion_factor': self.pixel_to_cm_factor
                },
                'calculation_details': {
                    'vertical_rebars_used': len(vertical_rebars),
                    'horizontal_rebars_used': len(horizontal_rebars),
                    'intersections_considered': len(intersections)
                }
            }
            
            print(f"   ✅ Final dimensions calculated:")
            print(f"      Internal: {internal_width_cm:.1f} x {internal_height_cm:.1f} cm")
            print(f"      With offset: {display_string}")
            print(f"      Offset applied: +{self.offset_cm}cm each side")
            
            return result
            
        except Exception as e:
            print(f"❌ Dimension calculation error: {str(e)}")
            # Return safe default with offset
            default_internal = 25.0
            default_side = default_internal + (2 * self.offset_cm)
            default_volume = default_side * default_side * 200
            default_volume_m3 = default_volume / 1_000_000
            
            return {
                'length': default_side,
                'width': default_side,
                'height': 200.0,
                'unit': 'cm',
                'volume': default_volume,
                'volume_m3': round(default_volume_m3, 6),
                'display': f'{default_side}cm x {default_side}cm x 200cm = {default_volume:.0f}cm³ = {default_volume_m3:.6f}m³',
                'method': 'fallback_with_offset',
                'offset_applied': self.offset_cm,
                'error': str(e)
            }
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture based on column volume using Philippine standards"""
        try:
            print("🧮 Calculating cement mixture...")
            
            volume_cm3 = dimensions.get('volume', 0)
            volume_m3 = volume_cm3 / 1000000  # Convert cm³ to m³
            
            print(f"   📊 Column volume: {volume_cm3:.0f} cm³ = {volume_m3:.6f} m³")
            
            # Standard Philippine concrete mix ratios
            cement_ratio = 1
            sand_ratio = 2
            aggregate_ratio = 3
            
            # Calculate concrete volume needed
            # Account for formwork, compaction, and wastage
            concrete_volume_factor = 1.4  # 40% more concrete than calculated volume
            total_concrete_volume_m3 = volume_m3 * concrete_volume_factor
            
            print(f"   🏗️ Total concrete needed: {total_concrete_volume_m3:.6f} m³ (factor: {concrete_volume_factor})")
            
            # Calculate material quantities based on ratio
            total_parts = cement_ratio + sand_ratio + aggregate_ratio  # 6 total parts
            volume_per_part = total_concrete_volume_m3 / total_parts
            
            # Individual material volumes
            cement_volume_m3 = volume_per_part * cement_ratio
            sand_volume_m3 = volume_per_part * sand_ratio
            aggregate_volume_m3 = volume_per_part * aggregate_ratio
            
            # Convert cement volume to bags (1 bag ≈ 0.035 m³)
            cement_bag_volume = 0.035  # m³ per bag
            cement_bags = cement_volume_m3 / cement_bag_volume
            
            result = {
                'cement_ratio': cement_ratio,
                'sand_ratio': sand_ratio,
                'aggregate_ratio': aggregate_ratio,
                'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
                'cement_bags': round(cement_bags, 2),
                'sand_volume_m3': round(sand_volume_m3, 4),
                'aggregate_volume_m3': round(aggregate_volume_m3, 4),
                'total_concrete_volume_m3': round(total_concrete_volume_m3, 4),
                'column_volume_m3': round(volume_m3, 4),
                'wastage_factor': concrete_volume_factor,
                'calculation_method': 'philippine_standard_mix',
                'cement_bag_volume_m3': cement_bag_volume
            }
            
            print(f"   ✅ Cement mixture calculated:")
            print(f"      📦 Cement: {result['cement_bags']} bags")
            print(f"      🏖️ Sand: {result['sand_volume_m3']} m³")
            print(f"      🪨 Aggregate: {result['aggregate_volume_m3']} m³")
            print(f"      🏗️ Total concrete: {result['total_concrete_volume_m3']} m³")
            
            return result
            
        except Exception as e:
            print(f"❌ Cement calculation error: {str(e)}")
            # Return safe default
            return {
                'cement_ratio': 1,
                'sand_ratio': 2,
                'aggregate_ratio': 3,
                'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
                'cement_bags': 3.0,
                'sand_volume_m3': 0.0002,
                'aggregate_volume_m3': 0.0003,
                'total_concrete_volume_m3': 0.0007,
                'error': str(e)
            }
    
    def create_analysis_visualization(self, image, vertical_rebars, horizontal_rebars, 
                                    intersections, dimensions):
        """Create comprehensive visualization showing detected rebars and measurements"""
        try:
            print("🎨 Creating detailed analysis visualization...")
            
            result_image = image.copy()
            
            # Draw vertical rebars in green with enhanced visualization
            print(f"   🟢 Drawing {len(vertical_rebars)} vertical rebars...")
            for i, vertical in enumerate(vertical_rebars):
                mask = vertical['mask']
                bbox = vertical['bbox']
                confidence = vertical['confidence']
                centroid = vertical['centroid']
                
                # Colored mask overlay with transparency
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask] = [0, 255, 0]  # Green
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
                
                # Bounding box
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Centroid marker
                cv2.circle(result_image, centroid, 8, (0, 255, 0), -1)
                cv2.circle(result_image, centroid, 10, (255, 255, 255), 2)
                
                # Label with ID and confidence
                label = f"V{i+1} ({confidence:.2f})"
                cv2.putText(result_image, label, (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_image, label, (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Draw horizontal rebars in red with selective labeling
            print(f"   🔴 Drawing {len(horizontal_rebars)} horizontal rebars...")
            for i, horizontal in enumerate(horizontal_rebars):
                mask = horizontal['mask']
                bbox = horizontal['bbox']
                confidence = horizontal['confidence']
                centroid = horizontal['centroid']
                
                # Colored mask overlay with transparency
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask] = [0, 0, 255]  # Red
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
                
                # Bounding box
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Centroid marker
                cv2.circle(result_image, centroid, 6, (0, 0, 255), -1)
                cv2.circle(result_image, centroid, 8, (255, 255, 255), 1)
                
                # Label only for first few and last few to avoid clutter
                if i < 3 or i >= len(horizontal_rebars) - 2:
                    label = f"H{i+1} ({confidence:.2f})"
                    cv2.putText(result_image, label, (x2+5, y1+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(result_image, label, (x2+5, y1+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw intersection points with different styles for strength
            valid_intersections = [i for i in intersections if i['is_valid']]
            weak_intersections = [i for i in intersections if not i['is_valid']]
            
            print(f"   🟡 Drawing {len(valid_intersections)} strong + {len(weak_intersections)} weak intersections...")
            
            # Strong intersections in bright yellow
            for i, intersection in enumerate(valid_intersections[:15]):  # Limit to top 15
                cx, cy = intersection['centroid']
                area = intersection['area']
                
                # Large yellow circle for strong intersections
                cv2.circle(result_image, (cx, cy), 8, (0, 255, 255), -1)  # Yellow fill
                cv2.circle(result_image, (cx, cy), 10, (255, 255, 255), 2)  # White border
                
                # Add intersection ID for first few
                if i < 5:
                    cv2.putText(result_image, f"{i+1}", (cx-5, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Weak intersections in orange
            for intersection in weak_intersections[:10]:  # Limit to 10
                cx, cy = intersection['centroid']
                cv2.circle(result_image, (cx, cy), 4, (0, 165, 255), -1)  # Orange
                cv2.circle(result_image, (cx, cy), 6, (255, 255, 255), 1)  # White border
            
            # Add comprehensive information overlay
            self._add_comprehensive_info_overlay(result_image, dimensions, 
                                               len(vertical_rebars), len(horizontal_rebars), 
                                               len(valid_intersections), len(weak_intersections))
            
            print(f"   ✅ Visualization complete:")
            print(f"      🟢 {len(vertical_rebars)} vertical rebars")
            print(f"      🔴 {len(horizontal_rebars)} horizontal rebars")
            print(f"      🟡 {len(valid_intersections)} strong intersections")
            print(f"      🟠 {len(weak_intersections)} weak intersections")
            
            return result_image
            
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return image
    
    def _add_comprehensive_info_overlay(self, image, dimensions, v_count, h_count, 
                                      strong_int_count, weak_int_count):
        """Add comprehensive information overlay to the image"""
        try:
            height, width = image.shape[:2]
            
            # Main dimensions box (top)
            box_height = 120
            cv2.rectangle(image, (10, 10), (width-10, box_height), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (width-10, box_height), (255, 255, 255), 2)
            
            # Title
            cv2.putText(image, "SIMPLIFIED REBAR ANALYSIS RESULTS", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Dimensions
            dimensions_text = dimensions['display']
            cv2.putText(image, dimensions_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Offset info
            offset_text = f"Offset Applied: +{dimensions['offset_applied']}cm each side"
            cv2.putText(image, offset_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Method and internal dimensions
            internal = dimensions.get('internal_dimensions', {})
            internal_text = f"Internal: {internal.get('width_cm', 0):.1f} x {internal.get('height_cm', 0):.1f} cm"
            cv2.putText(image, internal_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Detection summary box (bottom)
            summary_height = 100
            summary_top = height - summary_height - 10
            cv2.rectangle(image, (10, summary_top), (width-10, height-10), (0, 0, 0), -1)
            cv2.rectangle(image, (10, summary_top), (width-10, height-10), (255, 255, 255), 2)
            
            # Detection counts with color coding
            y_pos = summary_top + 25
            
            # Vertical count
            vertical_text = f"Vertical Rebars: {v_count}/2"
            vertical_color = (0, 255, 0) if v_count == 2 else (0, 255, 255)
            cv2.putText(image, vertical_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vertical_color, 2)
            
            # Horizontal count  
            horizontal_text = f"Horizontal Rebars: {h_count}/11"
            horizontal_color = (0, 255, 0) if h_count == 11 else (0, 255, 255)
            cv2.putText(image, horizontal_text, (250, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, horizontal_color, 2)
            
            # Intersection counts
            y_pos += 25
            intersections_text = f"Intersections: {strong_int_count} strong + {weak_int_count} weak"
            cv2.putText(image, intersections_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Target achievement status
            y_pos += 25
            target_met = (v_count == 2 and h_count == 11)
            target_text = f"Target Achievement: {'ACHIEVED' if target_met else 'PARTIAL'}"
            target_color = (0, 255, 0) if target_met else (0, 255, 255)
            cv2.putText(image, target_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
            
            # Processing method
            method_text = f"Method: {dimensions.get('method', 'unknown')}"
            cv2.putText(image, method_text, (300, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
        except Exception as e:
            print(f"⚠️ Error adding comprehensive info overlay: {e}")
    
    def validate_rebar_structure(self, vertical_rebars, horizontal_rebars, intersections):
        """
        Validate that detected rebars form a proper rebar structure
        
        Returns:
            dict: Validation results with success status and details
        """
        try:
            print("✅ Validating rebar structure...")
            
            validation_results = {
                'success': True,
                'issues': [],
                'warnings': [],
                'score': 0,
                'max_score': 100
            }
            
            # Check vertical rebar count (weight: 30 points)
            if len(vertical_rebars) == self.target_vertical:
                validation_results['score'] += 30
            elif abs(len(vertical_rebars) - self.target_vertical) == 1:
                validation_results['score'] += 20
                validation_results['warnings'].append(f"Expected {self.target_vertical} vertical rebars, found {len(vertical_rebars)}")
            else:
                validation_results['issues'].append(f"Vertical rebar count too far from target: {len(vertical_rebars)} vs {self.target_vertical}")
            
            # Check horizontal rebar count (weight: 30 points)
            if len(horizontal_rebars) == self.target_horizontal:
                validation_results['score'] += 30
            elif abs(len(horizontal_rebars) - self.target_horizontal) <= 2:
                validation_results['score'] += 20
                validation_results['warnings'].append(f"Expected {self.target_horizontal} horizontal rebars, found {len(horizontal_rebars)}")
            else:
                validation_results['issues'].append(f"Horizontal rebar count too far from target: {len(horizontal_rebars)} vs {self.target_horizontal}")
            
            # Check intersection count (weight: 25 points)
            expected_intersections = len(vertical_rebars) * len(horizontal_rebars)
            strong_intersections = len([i for i in intersections if i['is_valid']])
            
            if strong_intersections >= expected_intersections * 0.8:  # 80% of expected
                validation_results['score'] += 25
            elif strong_intersections >= expected_intersections * 0.6:  # 60% of expected
                validation_results['score'] += 15
                validation_results['warnings'].append(f"Lower intersection count: {strong_intersections} vs expected ~{expected_intersections}")
            else:
                validation_results['issues'].append(f"Too few intersections: {strong_intersections} vs expected ~{expected_intersections}")
            
            # Check rebar arrangement (weight: 15 points)
            if len(vertical_rebars) >= 2:
                # Check if verticals are reasonably spaced
                verticals_sorted = sorted(vertical_rebars, key=lambda v: v['centroid'][0])
                spacing = verticals_sorted[-1]['centroid'][0] - verticals_sorted[0]['centroid'][0]
                if spacing > 100:  # Reasonable spacing in pixels
                    validation_results['score'] += 15
                else:
                    validation_results['warnings'].append("Vertical rebars may be too close together")
                    validation_results['score'] += 5
            
            # Determine overall success
            if validation_results['score'] >= 80:
                validation_results['success'] = True
                validation_results['quality'] = 'excellent'
            elif validation_results['score'] >= 60:
                validation_results['success'] = True
                validation_results['quality'] = 'good'
            elif validation_results['score'] >= 40:
                validation_results['success'] = True
                validation_results['quality'] = 'acceptable'
            else:
                validation_results['success'] = False
                validation_results['quality'] = 'poor'
            
            print(f"   📊 Validation score: {validation_results['score']}/100 ({validation_results['quality']})")
            if validation_results['issues']:
                print(f"   ❌ Issues: {len(validation_results['issues'])}")
                for issue in validation_results['issues']:
                    print(f"      - {issue}")
            if validation_results['warnings']:
                print(f"   ⚠️ Warnings: {len(validation_results['warnings'])}")
                for warning in validation_results['warnings']:
                    print(f"      - {warning}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'score': 0,
                'quality': 'failed'
            }


# Utility functions for external use
def create_test_processor():
    """Create a test instance of SimplifiedRebarProcessor"""
    return SimplifiedRebarProcessor()

def validate_processor_setup():
    """Validate that the processor is set up correctly"""
    try:
        processor = SimplifiedRebarProcessor()
        print("✅ SimplifiedRebarProcessor validation:")
        print(f"   Target vertical: {processor.target_vertical}")
        print(f"   Target horizontal: {processor.target_horizontal}")
        print(f"   Offset: {processor.offset_cm}cm")
        print(f"   Pixel conversion: {processor.pixel_to_cm_factor}")
        print(f"   Min intersections: {processor.min_intersections_required}")
        return True
    except Exception as e:
        print(f"❌ Processor validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test the processor setup
    validate_processor_setup()
