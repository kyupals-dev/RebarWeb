"""
Phase 9: Auto-Save Process
- Save enhanced visualization as JPG
- Create metadata JSON file with: Dimensions, Cement mixture ratios, Detection details, Analysis timestamp, Model confidence scores
- Both files saved to gallery automatically
"""

import json
import os
from datetime import datetime
from .base_phase import BasePhase

class Phase09AutoSaveProcess(BasePhase):
    """Phase 9: Automatically save analysis results and metadata"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Auto-Save Process"
        self.save_metadata_json = True
        self.save_summary_txt = True
    
    def validate_input(self, data):
        """Validate input data for Phase 9"""
        required_keys = [
            'visualization_path',
            'dimensions', 
            'cement_mixture',
            'image_path'
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Check visualization file exists
        viz_path = data['visualization_path']
        if not os.path.exists(viz_path):
            raise ValueError(f"Visualization file not found: {viz_path}")
        
        return True
    
    def execute(self, data):
        """Execute Phase 9: Auto-Save Process"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        saved_files = []
        
        # Step 1: Visualization is already saved from Phase 8
        viz_path = data['visualization_path']
        saved_files.append({
            'type': 'visualization',
            'path': viz_path,
            'filename': os.path.basename(viz_path),
            'size_kb': os.path.getsize(viz_path) / 1024
        })
        
        self.log(f"✅ Visualization already saved: {os.path.basename(viz_path)}")
        
        # Step 2: Create and save metadata JSON file
        if self.save_metadata_json:
            metadata_path = self._save_metadata_json(data)
            if metadata_path:
                saved_files.append({
                    'type': 'metadata',
                    'path': metadata_path,
                    'filename': os.path.basename(metadata_path),
                    'size_kb': os.path.getsize(metadata_path) / 1024
                })
        
        # Step 3: Create and save summary text file
        if self.save_summary_txt:
            summary_path = self._save_summary_text(data)
            if summary_path:
                saved_files.append({
                    'type': 'summary', 
                    'path': summary_path,
                    'filename': os.path.basename(summary_path),
                    'size_kb': os.path.getsize(summary_path) / 1024
                })
        
        # Step 4: Create save summary
        save_summary = {
            'total_files_saved': len(saved_files),
            'files': saved_files,
            'save_location': os.path.dirname(viz_path),
            'save_timestamp': datetime.now().isoformat(),
            'total_size_kb': sum(f['size_kb'] for f in saved_files)
        }
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'auto_save_passed': True,
            'saved_files': saved_files,
            'save_summary': save_summary,
            'gallery_ready': True  # Signal that files are ready for gallery
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Files saved: {len(saved_files)}")
        self.log(f"   Total size: {save_summary['total_size_kb']:.1f} KB")
        for file_info in saved_files:
            self.log(f"   - {file_info['type']}: {file_info['filename']} ({file_info['size_kb']:.1f} KB)")
        
        return output_data
    
    def _save_metadata_json(self, data):
        """Create and save comprehensive metadata JSON file"""
        try:
            from app.utils.config import config
            
            # Generate metadata filename based on visualization filename
            viz_path = data['visualization_path']
            base_name = os.path.splitext(os.path.basename(viz_path))[0]
            metadata_filename = f'{base_name}_metadata.json'
            metadata_path = os.path.join(config.UPLOAD_FOLDER, metadata_filename)
            
            # Create comprehensive metadata
            metadata = {
                'analysis_info': {
                    'analysis_id': base_name,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_total': self._calculate_total_processing_time(data),
                    'model_type': 'real_model' if data.get('model_loaded') else 'placeholder',
                    'version': '1.0',
                    'phases_completed': self._get_completed_phases(data)
                },
                
                'input_data': {
                    'original_image_path': data.get('image_path'),
                    'original_image_size': data.get('original_size'),
                    'final_image_size': data.get('final_size'),
                    'rotation_applied': data.get('rotation_applied', False),
                    'resize_applied': data.get('resize_applied', False)
                },
                
                'detection_results': {
                    'num_detections': data.get('num_detections', 0),
                    'detection_threshold': data.get('detection_threshold', 0.2),
                    'class_counts': data.get('class_counts', {}),
                    'structured_detections': self._serialize_detections(data.get('structured_detections', [])),
                    'validation_passed': data.get('validation_passed', False)
                },
                
                'intersection_analysis': {
                    'analysis_passed': data.get('intersection_analysis_passed', False),
                    'intersection_area': data.get('intersection_area', 0),
                    'num_components': data.get('num_components', 0),
                    'num_centroid_rows': len(data.get('centroid_rows', [])),
                    'analysis_parameters': data.get('analysis_parameters', {})
                },
                
                'polygon_generation': {
                    'generation_passed': data.get('polygon_generation_passed', False),
                    'polygon_count': data.get('polygon_count', 0),
                    'total_polygon_area': data.get('total_polygon_area', 0),
                    'bounding_box': data.get('bounding_box', {}),
                    'generation_parameters': data.get('generation_parameters', {})
                },
                
                'dimensions': data.get('dimensions', {}),
                'dimension_calculation': data.get('calculation_metadata', {}),
                
                'cement_mixture': data.get('cement_mixture', {}),
                'mixture_summary': data.get('mixture_summary', {}),
                
                'visualization': {
                    'visualization_path': data.get('visualization_path'),
                    'has_detectron2_overlay': data.get('visualization_metadata', {}).get('has_detectron2_overlay', False),
                    'has_polygons': data.get('visualization_metadata', {}).get('has_polygons', False),
                    'has_bounding_box': data.get('visualization_metadata', {}).get('has_bounding_box', False)
                },
                
                'quality_metrics': {
                    'confidence_scores': self._extract_confidence_scores(data),
                    'detection_quality': data.get('detection_quality_summary', {}),
                    'analysis_reliability': self._assess_analysis_reliability(data)
                }
            }
            
            # Save JSON with pretty formatting
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            self.log(f"✅ Metadata saved: {metadata_filename}")
            return metadata_path
            
        except Exception as e:
            self.log(f"❌ Error saving metadata JSON: {e}")
            return None
    
    def _save_summary_text(self, data):
        """Create and save human-readable summary text file"""
        try:
            from app.utils.config import config
            
            # Generate summary filename
            viz_path = data['visualization_path']
            base_name = os.path.splitext(os.path.basename(viz_path))[0]
            summary_filename = f'{base_name}_summary.txt'
            summary_path = os.path.join(config.UPLOAD_FOLDER, summary_filename)
            
            # Create human-readable summary
            summary_lines = []
            summary_lines.append("=" * 60)
            summary_lines.append("REBAR VISTA - AI ANALYSIS SUMMARY")
            summary_lines.append("=" * 60)
            summary_lines.append("")
            
            # Analysis info
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            model_type = "Real AI Model" if data.get('model_loaded') else "Placeholder Mode"
            summary_lines.append(f"Analysis Date: {timestamp}")
            summary_lines.append(f"Model Type: {model_type}")
            summary_lines.append(f"Image: {os.path.basename(data.get('image_path', 'unknown'))}")
            summary_lines.append("")
            
            # Detection results
            summary_lines.append("DETECTION RESULTS:")
            summary_lines.append("-" * 20)
            summary_lines.append(f"Total Detections: {data.get('num_detections', 0)}")
            
            class_counts = data.get('class_counts', {})
            for class_name, count in class_counts.items():
                summary_lines.append(f"  - {class_name}: {count}")
            summary_lines.append("")
            
            # Dimensions
            dimensions = data.get('dimensions', {})
            summary_lines.append("CALCULATED DIMENSIONS:")
            summary_lines.append("-" * 22)
            summary_lines.append(f"Result: {dimensions.get('display', 'N/A')}")
            summary_lines.append(f"Length: {dimensions.get('length', 0)} cm")
            summary_lines.append(f"Width: {dimensions.get('width', 0)} cm") 
            summary_lines.append(f"Height: {dimensions.get('height', 0)} cm")
            summary_lines.append(f"Volume: {dimensions.get('volume_cm3', 0):,.0f} cm³")
            summary_lines.append("")
            
            # Distance and calibration
            calc_meta = data.get('calculation_metadata', {})
            summary_lines.append("MEASUREMENT DETAILS:")
            summary_lines.append("-" * 20)
            summary_lines.append(f"Camera Distance: {calc_meta.get('distance_cm', 0)} cm")
            summary_lines.append(f"Pixel-to-CM Factor: {calc_meta.get('pixel_to_cm_factor', 0):.4f}")
            summary_lines.append(f"Calibration Method: {calc_meta.get('calibration_method', 'unknown')}")
            summary_lines.append("")
            
            # Cement mixture
            cement_mixture = data.get('cement_mixture', {})
            summary_lines.append("CEMENT MIXTURE CALCULATION:")
            summary_lines.append("-" * 28)
            summary_lines.append(f"Mix Ratio: {cement_mixture.get('ratio_string', 'N/A')}")
            summary_lines.append(f"Cement Bags: {cement_mixture.get('cement_bags_rounded', 0)} bags")
            summary_lines.append(f"Sand Volume: {cement_mixture.get('sand_volume_m3_practical', 0)} m³")
            summary_lines.append(f"Aggregate Volume: {cement_mixture.get('aggregate_volume_m3_practical', 0)} m³")
            summary_lines.append(f"Total Concrete: {cement_mixture.get('total_concrete_volume_m3', 0)} m³")
            summary_lines.append("")
            
            # Analysis quality
            summary_lines.append("ANALYSIS QUALITY:")
            summary_lines.append("-" * 17)
            reliability = self._assess_analysis_reliability(data)
            summary_lines.append(f"Overall Reliability: {reliability.get('level', 'Unknown')}")
            summary_lines.append(f"Confidence Score: {reliability.get('score', 0):.2f}")
            
            quality_notes = reliability.get('notes', [])
            if quality_notes:
                summary_lines.append("Quality Notes:")
                for note in quality_notes:
                    summary_lines.append(f"  - {note}")
            summary_lines.append("")
            
            # Files created
            summary_lines.append("FILES CREATED:")
            summary_lines.append("-" * 14)
            summary_lines.append(f"Visualization: {os.path.basename(data.get('visualization_path', 'N/A'))}")
            summary_lines.append(f"Metadata: {os.path.basename(summary_path).replace('_summary.txt', '_metadata.json')}")
            summary_lines.append(f"Summary: {os.path.basename(summary_path)}")
            summary_lines.append("")
            
            summary_lines.append("=" * 60)
            summary_lines.append("End of Analysis Summary")
            summary_lines.append("=" * 60)
            
            # Write summary file
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            self.log(f"✅ Summary saved: {summary_filename}")
            return summary_path
            
        except Exception as e:
            self.log(f"❌ Error saving summary text: {e}")
            return None
    
    def _serialize_detections(self, detections):
        """Convert detections to JSON-serializable format"""
        serialized = []
        
        for detection in detections:
            serialized_detection = {}
            for key, value in detection.items():
                if key == 'mask':
                    # Don't serialize large mask arrays, just store metadata
                    if value is not None:
                        serialized_detection['mask_shape'] = value.shape if hasattr(value, 'shape') else None
                        serialized_detection['mask_area'] = detection.get('mask_area', 0)
                    else:
                        serialized_detection['mask_shape'] = None
                        serialized_detection['mask_area'] = 0
                else:
                    # Serialize other fields normally
                    serialized_detection[key] = value
            
            serialized.append(serialized_detection)
        
        return serialized
    
    def _extract_confidence_scores(self, data):
        """Extract confidence scores from detections"""
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
    
    def _assess_analysis_reliability(self, data):
        """Assess overall reliability of the analysis"""
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
    
    def _calculate_total_processing_time(self, data):
        """Calculate total processing time if phase timing info is available"""
        # This would need to be implemented if phase timing is tracked
        return "Not tracked"
    
    def _get_completed_phases(self, data):
        """Get list of completed phases"""
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
            ('Phase 9: Auto-Save Process', 'auto_save_passed')
        ]
        
        for phase_name, check_key in phase_checks:
            if data.get(check_key):
                completed.append(phase_name)
        
        return completed
    
    def validate_output(self, data):
        """Validate output data from Phase 9"""
        required_keys = ['auto_save_passed', 'saved_files', 'gallery_ready']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate saved files structure
        saved_files = data['saved_files']
        if not isinstance(saved_files, list):
            raise ValueError("saved_files must be a list")
        
        if len(saved_files) == 0:
            raise ValueError("No files were saved")
        
        # Check that all saved files exist
        for file_info in saved_files:
            if 'path' not in file_info:
                raise ValueError("File info missing path")
            
            file_path = file_info['path']
            if not os.path.exists(file_path):
                raise ValueError(f"Saved file not found: {file_path}")
        
        return True
