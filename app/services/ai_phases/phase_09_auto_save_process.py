"""
Phase 9: Auto Save Process
Automatically saves analysis results and metadata
"""

import json
import os
from datetime import datetime
from app.utils.config import config

class Phase09AutoSaveProcess:
    """Phase 9: Auto save results"""
    
    def __init__(self):
        self.results_dir = os.path.join(config.UPLOAD_FOLDER, 'analysis_results')
        self._ensure_results_directory()
        print("💾 Phase 9: Auto Save Process initialized")
    
    def auto_save_results(self, analysis_results):
        """Automatically save analysis results"""
        try:
            print("💾 Phase 9: Auto saving results...")
            
            if not analysis_results or not isinstance(analysis_results, dict):
                return {
                    'success': False,
                    'error': 'Invalid analysis results'
                }
            
            # Generate save paths
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            json_filename = f'analysis_{timestamp}.json'
            json_path = os.path.join(self.results_dir, json_filename)
            
            # Prepare save data
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_id': f"analysis_{timestamp}",
                'phase_results': analysis_results.get('phase_results', {}),
                'final_results': analysis_results.get('final_results', {}),
                'metadata': {
                    'image_path': analysis_results.get('image_path', ''),
                    'processing_time': 'calculated_in_phase_10',
                    'model_version': 'phased_pipeline_v1.0',
                    'save_location': json_path
                }
            }
            
            # Save JSON results
            with open(json_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            # Update analysis results with save info
            analysis_results['save_info'] = {
                'saved': True,
                'save_path': json_path,
                'save_time': datetime.now().isoformat(),
                'analysis_id': save_data['analysis_id']
            }
            
            print(f"   ✅ Phase 9: Results saved to {json_filename}")
            
            return {
                'success': True,
                'saved': True,
                'save_path': json_path,
                'analysis_id': save_data['analysis_id']
            }
            
        except Exception as e:
            print(f"   ❌ Phase 9 error: {str(e)}")
            return {
                'success': False,
                'error': f'Auto save failed: {str(e)}'
            }
    
    def _ensure_results_directory(self):
        """Ensure results directory exists"""
        try:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
                print(f"   📁 Created results directory: {self.results_dir}")
        except Exception as e:
            print(f"   ⚠️ Could not create results directory: {e}")
    
    def get_recent_results(self, limit=10):
        """Get recent analysis results"""
        try:
            if not os.path.exists(self.results_dir):
                return []
            
            files = [
                f for f in os.listdir(self.results_dir) 
                if f.startswith('analysis_') and f.endswith('.json')
            ]
            
            # Sort by modification time (newest first)
            files.sort(key=lambda f: os.path.getmtime(os.path.join(self.results_dir, f)), reverse=True)
            
            recent_results = []
            for filename in files[:limit]:
                try:
                    filepath = os.path.join(self.results_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        recent_results.append({
                            'filename': filename,
                            'analysis_id': data.get('analysis_id', 'unknown'),
                            'timestamp': data.get('timestamp', 'unknown'),
                            'image_path': data.get('metadata', {}).get('image_path', ''),
                            'filepath': filepath
                        })
                except Exception as e:
                    print(f"   ⚠️ Error reading {filename}: {e}")
            
            return recent_results
            
        except Exception as e:
            print(f"   ⚠️ Error getting recent results: {e}")
            return []
