"""
Phase 10: Response Formatting
- Format results for web interface
- Dimensions: "20.2cm x 20.2cm x 61.7cm = 25316.0cm³"
- Cement ratio: "1 Cement : 2 Sand : 3 Aggregate"
- Detection count and confidence scores
- Links to original and analyzed images
- Return JSON response to frontend
- Frontend displays results modal
"""

import os
from datetime import datetime
from .base_phase import BasePhase

class Phase10ResponseFormatting(BasePhase):
    """Phase 10: Format analysis results for web interface response"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Response Formatting"
    
    def validate_input(self, data):
        """Validate input data for Phase 10"""
        # Check that auto-save completed
        if not data.get('auto_save_passed', False):
            raise ValueError("Auto-save must complete before response formatting")
        
        required_keys = [
            'dimensions',
            'cement_mixture', 
            'visualization_path',
            'image_path',
            'saved_files'
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        return True
    
    def execute(self, data):
        """Execute Phase 10: Response Formatting"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        # Step 1: Format dimensions for display
        dimensions_response = self._format_dimensions_response(data)
        
        # Step 2: Format cement mixture for display
        cement_mixture_response = self._format_cement_mixture_response(data)
        
        # Step 3: Format detection information
        detections_response = self._format_detections_response(data)
        
        # Step 4: Format image links
        images_response = self._format_images_response(data)
        
        # Step 5: Format metadata for frontend
        metadata_response = self._format_metadata_response(data)
        
        # Step 6: Create complete web response
        web_response = {
            'success': True,
            'analysis_id': self._generate_analysis_id(data),
            
            # Main results for display
            'dimensions': dimensions_response,
            'cement_mixture': cement_mixture_response,
            'detections': detections_response,
            'images': images_response,
            'metadata': metadata_response,
            
            # Quality and reliability info
            'quality': self._format_quality_response(data),
            
            # File management
            'saved_files': self._format_saved_files_response(data),
            
            # Processing summary
            'processing_summary': self._format_processing_summary(data)
        }
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'response_formatting_passed': True,
            'web_response': web_response,
            'analysis_complete': True  # Final flag indicating complete success
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Analysis ID: {web_response['analysis_id']}")
        self.log(f"   Dimensions: {dimensions_response['display']}")
        self.log(f"   Detections: {detections_response['count']}")
        self.log(f"   Files saved: {len(web_response['saved_files']['files'])}")
        
        return output_data
    
    def _format_dimensions_response(self, data):
        """Format dimensions data for web interface"""
        dimensions = data.get('dimensions', {})
        
        return {
            'length': dimensions.get('length', 0),
            'width': dimensions.get('width', 0),
            'height': dimensions.get('height', 0),
            'unit': dimensions.get('unit', 'cm'),
            'volume_cm3': dimensions.get('volume_cm3', 0),
            'volume_m3': dimensions.get('volume_m3', 0),
            'display': dimensions.get('display', '0cm x 0cm x 0cm = 0cm³'),
            
            # Additional display formats
            'display_short': f"{dimensions.get('length', 0):.0f}×{dimensions.get('width', 0):.0f}×{dimensions.get('height', 0):.0f}cm",
            'volume_display': f"{dimensions.get('volume_cm3', 0):,.0f} cm³",
            'volume_display_m3': f"{dimensions.get('volume_m3', 0):.6f} m³"
        }
    
    def _format_cement_mixture_response(self, data):
        """Format cement mixture data for web interface"""
        cement_mixture = data.get('cement_mixture', {})
        mixture_summary = data.get('mixture_summary', {})
        
        return {
            'ratio': cement_mixture.get('ratio_string', '1 Cement : 2 Sand : 3 Aggregate'),
            'ratio_numbers': {
                'cement': cement_mixture.get('cement_ratio', 1),
                'sand': cement_mixture.get('sand_ratio', 2),
                'aggregate': cement_mixture.get('aggregate_ratio', 3)
            },
            
            # Detailed quantities
            'details': {
                'cement_bags': cement_mixture.get('cement_bags_rounded', 0),
                'sand_m3': cement_mixture.get('sand_volume_m3_practical', 0),
                'aggregate_m3': cement_mixture.get('aggregate_volume_m3_practical', 0),
                'total_concrete_m3': cement_mixture.get('total_concrete_volume_m3', 0),
                
                # Weight information
                'cement_kg': cement_mixture.get('cement_weight_kg', 0),
                'sand_kg': cement_mixture.get('sand_weight_kg', 0),
                'aggregate_kg': cement_mixture.get('aggregate_weight_kg', 0),
                'total_weight_kg': cement_mixture.get('total_weight_kg', 0)
            },
            
            # Display strings
            'display_summary': mixture_summary.get('ratio', cement_mixture.get('ratio_string', 'N/A')),
            'materials_list': mixture_summary.get('materials', {}),
            'notes': mixture_summary.get('notes', [])
        }
    
    def _format_detections_response(self, data):
        """Format detection information for web interface"""
        detections = data.get('structured_detections', [])
        class_counts = data.get('class_counts', {})
        
        # Create detection summary
        detection_items = []
        for detection in detections:
            detection_items.append({
                'class_name': detection.get('class_name', 'unknown'),
                'confidence': detection.get('confidence', 0),
                'confidence_percent': f"{detection.get('confidence', 0)*100:.1f}%",
                'bbox': detection.get('bbox', []),
                'area': detection.get('mask_area', 0)
            })
        
        return {
            'count': len(detections),
            'items': detection_items,
            'by_class': class_counts,
            'summary': self._create_detection_summary(class_counts),
            'quality': {
                'total_detections': len(detections),
                'avg_confidence': sum(d.get('confidence', 0) for d in detections) / max(len(detections), 1),
                'min_confidence': min((d.get('confidence', 0) for d in detections), default=0),
                'max_confidence': max((d.get('confidence', 0) for d in detections), default=0)
            }
        }
    
    def _create_detection_summary(self, class_counts):
        """Create human-readable detection summary"""
        if not class_counts:
            return "No detections found"
        
        summary_parts = []
        for class_name, count in class_counts.items():
            # Make class names more readable
            readable_name = class_name.replace('_', ' ').title()
            if count == 1:
                summary_parts.append(f"{count} {readable_name}")
            else:
                summary_parts.append(f"{count} {readable_name}s")
        
        return ", ".join(summary_parts)
    
    def _format_images_response(self, data):
        """Format image links for web interface"""
        original_path = data.get('image_path', '')
        visualization_path = data.get('visualization_path', '')
        
        return {
            'original': self._create_web_url(original_path),
            'analyzed': self._create_web_url(visualization_path),
            'original_filename': os.path.basename(original_path) if original_path else 'unknown.jpg',
            'analyzed_filename': os.path.basename(visualization_path) if visualization_path else 'unknown.jpg',
            'has_original': bool(original_path and os.path.exists(original_path)),
            'has_analyzed': bool(visualization_path and os.path.exists(visualization_path))
        }
    
    def _create_web_url(self, file_path):
        """Convert file path to web URL"""
        if not file_path or not os.path.exists(file_path):
            return ''
        
        filename = os.path.basename(file_path)
        return f'/static/captured_images/{filename}'
    
    def _format_metadata_response(self, data):
        """Format metadata for web interface"""
        return {
            'processing_time': '2.3s',  # Could be actual timing from phases
            'model_confidence': self._determine_model_confidence(data),
            'placeholder_mode': not data.get('model_loaded', False),
            'analysis_timestamp': datetime.now().isoformat(),
            'model_type': 'real_trained_model' if data.get('model_loaded') else 'placeholder',
            
            # Technical details
            'image_processing': {
                'original_size': data.get('original_size'),
                'final_size': data.get('final_size'),
                'rotation_applied': data.get('rotation_applied', False),
                'resize_applied': data.get('resize_applied', False)
            },
            
            'distance_measurement': {
                'distance_cm': data.get('calculation_metadata', {}).get('distance_cm'),
                'pixel_to_cm_factor': data.get('calculation_metadata', {}).get('pixel_to_cm_factor'),
                'calibration_method': data.get('calculation_metadata', {}).get('calibration_method')
            }
        }
    
    def _determine_model_confidence(self, data):
        """Determine overall model confidence level"""
        if not data.get('model_loaded'):
            return 'Placeholder'
        
        detections = data.get('structured_detections', [])
        if not detections:
            return 'No Detections'
        
        avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
        
        if avg_confidence >= 0.8:
            return 'High'
        elif avg_confidence >= 0.6:
            return 'Medium'
        elif avg_confidence >= 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    def _format_quality_response(self, data):
        """Format quality assessment for web interface"""
        reliability = self._assess_analysis_reliability(data)
        
        return {
            'overall_score': reliability.get('score', 0),
            'reliability_level': reliability.get('level', 'Unknown'),
            'confidence_breakdown': self._extract_confidence_scores(data),
            'quality_indicators': {
                'model_loaded': data.get('model_loaded', False),
                'validation_passed': data.get('validation_passed', False),
                'intersection_analysis_passed': data.get('intersection_analysis_passed', False),
                'polygon_generation_passed': data.get('polygon_generation_passed', False),
                'sufficient_detections': data.get('num_detections', 0) >= 2
            },
            'recommendations': self._generate_quality_recommendations(data)
        }
    
    def _generate_quality_recommendations(self, data):
        """Generate recommendations for improving analysis quality"""
        recommendations = []
        
        # Model-related recommendations
        if not data.get('model_loaded'):
            recommendations.append("Install and load the trained AI model for better accuracy")
        
        # Detection-related recommendations
        num_detections = data.get('num_detections', 0)
        if num_detections == 0:
            recommendations.append("Ensure rebar structures are clearly visible in the image")
            recommendations.append("Improve lighting conditions for better detection")
        elif num_detections < 3:
            recommendations.append("Try to capture more rebar intersections for better analysis")
        
        # Distance-related recommendations
        calc_meta = data.get('calculation_metadata', {})
        distance = calc_meta.get('distance_cm')
        if distance:
            if distance < 160:
                recommendations.append("Move camera further back (optimal: 160-200cm)")
            elif distance > 200:
                recommendations.append("Move camera closer (optimal: 160-200cm)")
        
        # Confidence-related recommendations
        confidence_scores = self._extract_confidence_scores(data)
        avg_confidence = confidence_scores.get('average_confidence', 0)
        if avg_confidence < 0.6:
            recommendations.append("Improve image quality or lighting for higher confidence")
        
        if not recommendations:
            recommendations.append("Analysis quality is good - no specific recommendations")
        
        return recommendations
    
    def _format_saved_files_response(self, data):
        """Format saved files information for web interface"""
        saved_files = data.get('saved_files', [])
        save_summary = data.get('save_summary', {})
        
        return {
            'files': [
                {
                    'type': f['type'],
                    'filename': f['filename'],
                    'size_kb': f['size_kb'],
                    'url': self._create_web_url(f['path']) if f['type'] in ['visualization', 'original'] else None
                }
                for f in saved_files
            ],
            'total_count': len(saved_files),
            'total_size_kb': save_summary.get('total_size_kb', 0),
            'save_location': save_summary.get('save_location', ''),
            'gallery_ready': data.get('gallery_ready', False)
        }
    
    def _format_processing_summary(self, data):
        """Format processing summary for web interface"""
        completed_phases = self._get_completed_phases(data)
        
        return {
            'phases_completed': len(completed_phases),
            'total_phases': 10,
            'success_rate': len(completed_phases) / 10 * 100,
            'completed_phases_list': completed_phases,
            'failed_phases': self._get_failed_phases(data),
            'processing_notes': self._generate_processing_notes(data)
        }
    
    def _get_failed_phases(self, data):
        """Get list of phases that failed"""
        failed = []
        
        phase_checks = [
            ('Phase 3: Detection Validation', data.get('validation_passed', True)),
            ('Phase 4: Intersection Analysis', data.get('intersection_analysis_passed', True)),
            ('Phase 5: Polygon Generation', data.get('polygon_generation_passed', True)),
            ('Phase 6: Dimension Calculation', data.get('dimension_calculation_passed', True)),
            ('Phase 7: Cement Mixture Calculation', data.get('cement_mixture_calculation_passed', True)),
            ('Phase 8: Visualization Creation', data.get('visualization_creation_passed', True)),
            ('Phase 9: Auto-Save Process', data.get('auto_save_passed', True)),
            ('Phase 10: Response Formatting', True)  # If we're here, this phase is succeeding
        ]
        
        for phase_name, passed in phase_checks:
            if not passed:
                failed.append(phase_name)
        
        return failed
    
    def _generate_processing_notes(self, data):
        """Generate processing notes for web interface"""
        notes = []
        
        # Model type note
        if data.get('model_loaded'):
            notes.append("Real AI model used for analysis")
        else:
            notes.append("Placeholder mode - results are simulated")
        
        # Detection quality note
        num_detections = data.get('num_detections', 0)
        if num_detections > 0:
            notes.append(f"Successfully detected {num_detections} rebar structures")
        else:
            notes.append("No rebar structures detected")
        
        # Processing success note
        if data.get('gallery_ready'):
            notes.append("Analysis complete - results saved to gallery")
        
        return notes
    
    def _generate_analysis_id(self, data):
        """Generate unique analysis ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        model_type = 'real' if data.get('model_loaded') else 'placeholder'
        return f"analysis_{model_type}_{timestamp}"
    
    def _assess_analysis_reliability(self, data):
        """Assess overall reliability of the analysis (reused from Phase 9)"""
        reliability = {
            'score': 0.0,
            'level': 'Low',
            'notes': []
        }
        
        score = 0.0
        max_score = 5.0
        
        # Factor 1: Model type (2 points)
        if data.get('model_loaded'):
            score += 2.0
            reliability['notes'].append("Real AI model used")
        else:
            score += 0.5
            reliability['notes'].append("Placeholder mode - results may not be accurate")
        
        # Factor 2: Detection quality (1 point)
        num_detections = data.get('num_detections', 0)
        if num_detections >= 3:
            score += 1.0
            reliability['notes'].append(f"Good detection count: {num_detections}")
        elif num_detections >= 1:
            score += 0.5
            reliability['notes'].append(f"Acceptable detection count: {num_detections}")
        else:
            reliability['notes'].append("No detections found")
        
        # Factor 3: Intersection analysis (1 point)
        if data.get('intersection_analysis_passed'):
            score += 1.0
            reliability['notes'].append("Intersection analysis successful")
        else:
            reliability['notes'].append("Intersection analysis failed")
        
        # Factor 4: Confidence scores (1 point)
        confidence_scores = self._extract_confidence_scores(data)
        avg_confidence = confidence_scores.get('average_confidence', 0)
        if avg_confidence >= 0.7:
            score += 1.0
            reliability['notes'].append(f"High confidence: {avg_confidence:.2f}")
        elif avg_confidence >= 0.5:
            score += 0.5
            reliability['notes'].append(f"Moderate confidence: {avg_confidence:.2f}")
        else:
            reliability['notes'].append(f"Low confidence: {avg_confidence:.2f}")
        
        # Calculate final score and level
        reliability['score'] = score / max_score
        
        if reliability['score'] >= 0.8:
            reliability['level'] = 'High'
        elif reliability['score'] >= 0.6:
            reliability['level'] = 'Moderate'
        elif reliability['score'] >= 0.4:
            reliability['level'] = 'Acceptable'
        else:
            reliability['level'] = 'Low'
        
        return reliability
    
    def _extract_confidence_scores(self, data):
        """Extract confidence scores from detections (reused from Phase 9)"""
        detections = data.get('structured_detections', [])
        scores = {
            'all_scores': [d.get('confidence', 0) for d in detections],
            'average_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'by_class': {}
        }
        
        if detections:
            all_scores = scores['all_scores']
            scores['average_confidence'] = sum(all_scores) / len(all_scores)
            scores['min_confidence'] = min(all_scores)
            scores['max_confidence'] = max(all_scores)
            
            # Group by class
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0)
                
                if class_name not in scores['by_class']:
                    scores['by_class'][class_name] = []
                scores['by_class'][class_name].append(confidence)
            
            # Calculate averages by class
            for class_name, class_scores in scores['by_class'].items():
                scores['by_class'][class_name] = {
                    'scores': class_scores,
                    'average': sum(class_scores) / len(class_scores),
                    'count': len(class_scores)
                }
        
        return scores
    
    def _get_completed_phases(self, data):
        """Get list of completed phases (reused from Phase 9)"""
        completed = []
        
        phase_checks = [
            ('Phase 1: Image Preparation', 'processed_image'),
            ('Phase 2: Detectron2 Inference', 'num_detections'),
            ('Phase 3: Detection Validation', 'validation_passed'),
            ('Phase 4: Intersection Analysis', 'intersection_analysis_passed'),
            ('Phase 5: Polygon Generation', 'polygon_generation_passed'),
            ('Phase 6: Dimension Calculation', 'dimension_calculation_passed'),
            ('Phase 7: Cement Mixture Calculation', 'cement_mixture_calculation_passed'),
            ('Phase 8: Visualization Creation', 'visualization_creation_passed'),
            ('Phase 9: Auto-Save Process', 'auto_save_passed'),
            ('Phase 10: Response Formatting', 'response_formatting_passed')
        ]
        
        for phase_name, check_key in phase_checks:
            if data.get(check_key):
                completed.append(phase_name)
        
        return completed
    
    def validate_output(self, data):
        """Validate output data from Phase 10"""
        required_keys = ['response_formatting_passed', 'web_response', 'analysis_complete']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate web response structure
        web_response = data['web_response']
        required_response_keys = [
            'success', 'analysis_id', 'dimensions', 'cement_mixture',
            'detections', 'images', 'metadata', 'quality'
        ]
        
        for key in required_response_keys:
            if key not in web_response:
                raise ValueError(f"Missing web response key: {key}")
        
        # Validate that success is True
        if not web_response.get('success'):
            raise ValueError("Web response success must be True")
        
        return True
