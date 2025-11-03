#!/usr/bin/env python3
"""
Standalone AI Debug Script
Run this script to test the AI service independently
"""

import sys
import os
import time
import traceback
import argparse
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the AI service
try:
    from app.services.ai_service import AIService
    from app.utils.config import config
    print("✅ Successfully imported AI service")
except ImportError as e:
    print(f"❌ Failed to import AI service: {e}")
    sys.exit(1)

class AIDebugger:
    """Standalone AI debugging tool"""
    
    def __init__(self):
        self.ai_service = None
        
    def run_comprehensive_test(self):
        """Run comprehensive AI service test"""
        print("="*70)
        print("COMPREHENSIVE AI SERVICE DEBUG TEST")
        print("="*70)
        
        # Test 1: Service initialization
        print("\n1. Testing AI Service Initialization...")
        try:
            start_time = time.time()
            self.ai_service = AIService()
            init_time = time.time() - start_time
            print(f"✅ AI Service initialized in {init_time:.2f}s")
        except Exception as e:
            print(f"❌ AI Service initialization failed: {e}")
            traceback.print_exc()
            return False
        
        # Test 2: Model status check
        print("\n2. Testing Model Status...")
        try:
            status = self.ai_service.get_model_status()
            print(f"   Detectron2 Available: {'✅' if status['detectron2_available'] else '❌'}")
            print(f"   Model Loaded: {'✅' if status['model_loaded'] else '❌'}")
            print(f"   Model Path: {status['model_path']}")
            print(f"   Model Exists: {'✅' if status['model_exists'] else '❌'}")
            print(f"   Classes: {status['class_names']}")
        except Exception as e:
            print(f"❌ Model status check failed: {e}")
            traceback.print_exc()
        
        # Test 3: Debug info
        print("\n3. Testing Debug Info...")
        try:
            debug_info = self.ai_service.get_debug_info()
            system = debug_info.get('system', {})
            model = debug_info.get('model', {})
            
            print(f"   System Memory: {system.get('memory_available_gb', 0):.1f}GB available")
            print(f"   Process Memory: {system.get('process_memory_mb', 0):.1f}MB")
            print(f"   Model Size: {model.get('model_size_mb', 0):.1f}MB")
            print(f"   CPU Count: {system.get('cpu_count', 'Unknown')}")
        except Exception as e:
            print(f"❌ Debug info failed: {e}")
        
        # Test 4: Find test images
        print("\n4. Looking for Test Images...")
        test_images = self.find_test_images()
        if test_images:
            print(f"✅ Found {len(test_images)} test images")
            for img in test_images[:3]:  # Show first 3
                size_kb = os.path.getsize(img) / 1024
                print(f"   - {os.path.basename(img)} ({size_kb:.1f}KB)")
        else:
            print("❌ No test images found")
            return False
        
        # Test 5: Analyze test images
        print("\n5. Testing Image Analysis...")
        for i, test_image in enumerate(test_images[:2]):  # Test first 2 images
            print(f"\n   Testing image {i+1}: {os.path.basename(test_image)}")
            try:
                start_time = time.time()
                result = self.ai_service.analyze_image(test_image)
                analysis_time = time.time() - start_time
                
                if result['success']:
                    print(f"   ✅ Analysis completed in {analysis_time:.2f}s")
                    print(f"   🎯 Detections: {result.get('num_detections', 0)}")
                    print(f"   📐 Dimensions: {result.get('dimensions', {}).get('display', 'N/A')}")
                    print(f"   🧮 Mixture: {result.get('cement_mixture', {}).get('ratio_string', 'N/A')}")
                    print(f"   🤖 Model Type: {result.get('model_type', 'Unknown')}")
                else:
                    print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"   ❌ Analysis exception: {e}")
                traceback.print_exc()
        
        # Test 6: Memory usage check
        print("\n6. Memory Usage Analysis...")
        try:
            if hasattr(self.ai_service, 'get_memory_usage'):
                memory = self.ai_service.get_memory_usage()
                print(f"   Current Process Memory: {memory}MB")
            
            import psutil
            system_memory = psutil.virtual_memory()
            print(f"   System Memory Usage: {system_memory.percent}%")
            print(f"   Available Memory: {system_memory.available / 1024 / 1024 / 1024:.1f}GB")
        except Exception as e:
            print(f"   ⚠️ Memory check failed: {e}")
        
        print("\n" + "="*70)
        print("DEBUG TEST COMPLETED")
        print("="*70)
        
        return True
    
    def find_test_images(self):
        """Find test images in various locations"""
        test_locations = [
            config.UPLOAD_FOLDER,
            'static/captured_images',
            'test_images',
            '.',
            'app/test_images'
        ]
        
        images = []
        
        for location in test_locations:
            if os.path.exists(location):
                for file in os.listdir(location):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        full_path = os.path.join(location, file)
                        images.append(full_path)
        
        # Sort by modification time (newest first)
        images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return images
    
    def test_specific_image(self, image_path):
        """Test analysis on a specific image"""
        print(f"\nTesting specific image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        if not self.ai_service:
            print("Initializing AI service...")
            self.ai_service = AIService()
        
        try:
            # Get image info
            size_kb = os.path.getsize(image_path) / 1024
            print(f"Image size: {size_kb:.1f}KB")
            
            # Load image to check dimensions
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                print(f"Image dimensions: {image.shape}")
            else:
                print("❌ Could not load image with OpenCV")
                return
            
            # Run analysis
            print("Running AI analysis...")
            start_time = time.time()
            result = self.ai_service.analyze_image(image_path)
            analysis_time = time.time() - start_time
            
            print(f"Analysis completed in {analysis_time:.2f}s")
            
            if result['success']:
                print("\n✅ ANALYSIS RESULTS:")
                print(f"   Detections: {result.get('num_detections', 0)}")
                print(f"   Dimensions: {result.get('dimensions', {}).get('display', 'N/A')}")
                print(f"   Cement Mixture: {result.get('cement_mixture', {}).get('ratio_string', 'N/A')}")
                print(f"   Model Type: {result.get('model_type', 'Unknown')}")
                
                if 'analyzed_image_path' in result:
                    print(f"   Analyzed image saved: {result['analyzed_image_path']}")
            else:
                print(f"\n❌ ANALYSIS FAILED:")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception during analysis: {e}")
            traceback.print_exc()
    
    def interactive_mode(self):
        """Interactive debugging mode"""
        print("\n" + "="*50)
        print("INTERACTIVE AI DEBUG MODE")
        print("="*50)
        print("Commands:")
        print("  'test' - Run comprehensive test")
        print("  'status' - Show model status") 
        print("  'memory' - Show memory usage")
        print("  'images' - List available test images")
        print("  'analyze <image>' - Analyze specific image")
        print("  'quit' - Exit")
        print()
        
        if not self.ai_service:
            print("Initializing AI service...")
            self.ai_service = AIService()
        
        while True:
            try:
                command = input("debug> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'test':
                    self.run_comprehensive_test()
                elif command == 'status':
                    status = self.ai_service.get_model_status()
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                elif command == 'memory':
                    debug_info = self.ai_service.get_debug_info()
                    system = debug_info.get('system', {})
                    print(f"  System Memory: {system.get('memory_usage_percent', 0)}% used")
                    print(f"  Available: {system.get('memory_available_gb', 0):.1f}GB")
                    print(f"  Process: {system.get('process_memory_mb', 0):.1f}MB")
                elif command == 'images':
                    images = self.find_test_images()
                    print(f"Found {len(images)} test images:")
                    for img in images[:10]:  # Show first 10
                        print(f"  - {img}")
                elif command.startswith('analyze '):
                    image_path = command[8:].strip()
                    self.test_specific_image(image_path)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='AI Service Debug Tool')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test')
    parser.add_argument('--image', type=str, help='Test specific image file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--status', action='store_true', help='Show status only')
    
    args = parser.parse_args()
    
    debugger = AIDebugger()
    
    if args.test:
        debugger.run_comprehensive_test()
    elif args.image:
        debugger.test_specific_image(args.image)
    elif args.status:
        debugger.ai_service = AIService()
        status = debugger.ai_service.get_model_status()
        for key, value in status.items():
            print(f"{key}: {value}")
    elif args.interactive:
        debugger.interactive_mode()
    else:
        # Default: run comprehensive test
        debugger.run_comprehensive_test()

if __name__ == '__main__':
    main()
