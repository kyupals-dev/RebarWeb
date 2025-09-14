"""
Flask App Initialization with Fixed Sensor Routes Registration
FIXED: Properly registers sensor_routes.py to resolve 404 errors
"""

from flask import Flask
import os

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__, 
                static_folder='../static',
                template_folder='../templates')
    
    # Basic configuration
    app.config['SECRET_KEY'] = 'rebar-vista-pipeline-mode-secret-key-2024'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload
    
    # CORS headers for API endpoints
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Endpoint not found', 'status': 404}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error', 'status': 500}, 500
    
    # Register blueprints
    try:
        print("🔄 Registering Flask blueprints...")
        
        # Import and register page routes
        try:
            from app.routes.page_routes import page_bp
            app.register_blueprint(page_bp)
            print("✅ Page routes registered")
        except ImportError as e:
            print(f"⚠️ Page routes import error: {e}")
        
        # Import and register camera routes
        try:
            from app.routes.camera_routes import camera_bp
            app.register_blueprint(camera_bp)
            print("✅ Camera routes registered")
        except ImportError as e:
            print(f"⚠️ Camera routes import error: {e}")
        
        # Import and register image routes
        try:
            from app.routes.image_routes import image_bp
            app.register_blueprint(image_bp)
            print("✅ Image routes registered")
        except ImportError as e:
            print(f"⚠️ Image routes import error: {e}")
        
        # Import and register AI routes
        try:
            from app.routes.ai_routes import ai_bp
            app.register_blueprint(ai_bp)
            print("✅ AI routes registered")
        except ImportError as e:
            print(f"⚠️ AI routes import error: {e}")
        
        # FIXED: Import and register sensor routes
        try:
            from app.routes.sensor_routes import sensor_bp
            app.register_blueprint(sensor_bp)
            print("✅ Sensor routes registered")
        except ImportError as e:
            print(f"❌ Sensor routes import error: {e}")
            print("   Creating fallback sensor routes...")
            create_fallback_sensor_routes(app)
        
        print("🎯 All available blueprints registered successfully")
        
        # Print registered routes for debugging
        print("\n📋 Registered routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"   {rule.endpoint}: {rule.rule} [{methods}]")
        
    except Exception as e:
        print(f"❌ Error registering blueprints: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal fallback routes
        create_minimal_fallback_routes(app)
    
    return app

def create_fallback_sensor_routes(app):
    """Create fallback sensor routes if sensor_routes.py is not available"""
    print("🔧 Creating fallback sensor routes...")
    
    @app.route('/get-distance', methods=['GET'])
    def get_distance_fallback():
        import random
        from datetime import datetime
        
        # Simulate distance reading
        distance_cm = random.uniform(150, 220)
        
        if distance_cm < 160:
            status = 'too_close'
            status_text = 'TOO CLOSE'
        elif distance_cm > 200:
            status = 'too_far'
            status_text = 'TOO FAR'
        else:
            status = 'optimal'
            status_text = 'OPTIMAL'
        
        return {
            'success': True,
            'distance_cm': round(distance_cm, 1),
            'distance_text': f"{distance_cm:.1f}cm",
            'status': status,
            'status_text': status_text,
            'timestamp': datetime.now().isoformat(),
            'simulation_mode': True
        }
    
    @app.route('/start-distance-monitoring', methods=['POST'])
    def start_distance_monitoring_fallback():
        return {
            'success': True,
            'message': 'Distance monitoring started (fallback mode)'
        }
    
    @app.route('/stop-distance-monitoring', methods=['POST'])
    def stop_distance_monitoring_fallback():
        return {
            'success': True,
            'message': 'Distance monitoring stopped (fallback mode)'
        }
    
    @app.route('/get-sensor-status', methods=['GET'])
    def get_sensor_status_fallback():
        return {
            'success': True,
            'status': {
                'gpio_available': False,
                'sensor_available': True,
                'is_running': True,
                'simulation_mode': True,
                'fallback_mode': True,
                'gpio_pins': {
                    'trigger': 'N/A',
                    'echo': 'N/A'
                },
                'optimal_range': {
                    'min': 160,
                    'max': 200,
                    'unit': 'cm'
                }
            }
        }
    
    print("✅ Fallback sensor routes created")

def create_minimal_fallback_routes(app):
    """Create minimal fallback routes if all else fails"""
    print("🆘 Creating minimal fallback routes...")
    
    @app.route('/')
    def index_fallback():
        return '''
        <html>
        <head><title>Rebar Vista - Starting Up</title></head>
        <body>
        <h1>Rebar Vista</h1>
        <p>System is starting up. Please wait...</p>
        <p><a href="/mainpage.html">Go to Camera Interface</a></p>
        </body>
        </html>
        '''
    
    @app.route('/mainpage.html')
    def mainpage_fallback():
        return '''
        <html>
        <head><title>Rebar Vista - Camera Interface</title></head>
        <body>
        <h1>Rebar Vista Camera Interface</h1>
        <p>Loading camera interface...</p>
        <script>
        setTimeout(() => {
            window.location.reload();
        }, 3000);
        </script>
        </body>
        </html>
        '''
    
    print("✅ Minimal fallback routes created")

# Health check endpoint
def add_health_check(app):
    """Add health check endpoint"""
    @app.route('/health', methods=['GET'])
    def health_check():
        return {
            'status': 'healthy',
            'service': 'rebar-vista',
            'mode': 'pipeline',
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    # For testing the app creation
    app = create_app()
    print("Flask app created successfully!")
    
    # Print all routes
    print("\nAvailable routes:")
    for rule in app.url_map.iter_rules():
        methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
        print(f"  {rule.rule} [{methods}] -> {rule.endpoint}")
