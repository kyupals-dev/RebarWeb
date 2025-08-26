from flask import Flask
import os

def create_app():
    # Get the absolute path to the project directory
    # __file__ points to app/__init__.py, so we need to go up one level
    app_dir = os.path.abspath(os.path.dirname(__file__))  # This is the 'app' directory
    project_root = os.path.dirname(app_dir)  # Go up one level to 'Web App PD'
    
    # Explicitly set template and static folders using absolute paths
    template_dir = os.path.join(project_root, 'templates')
    static_dir = os.path.join(project_root, 'static')
    
    print(f"App directory: {app_dir}")
    print(f"Project root: {project_root}")
    print(f"Template directory: {template_dir}")
    print(f"Static directory: {static_dir}")
    print(f"Template dir exists: {os.path.exists(template_dir)}")
    print(f"Static dir exists: {os.path.exists(static_dir)}")
    
    # Create Flask app with absolute paths
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')
    
    # Verify the Flask app configuration
    print(f"Flask template folder: {app.template_folder}")
    print(f"Flask static folder: {app.static_folder}")
    
    # Create upload directory
    upload_folder = os.path.join(static_dir, 'captured_images')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        print(f"Created upload folder: {upload_folder}")
    
    try:
        # Register blueprints
        from app.routes.page_routes import page_bp
        from app.routes.camera_routes import camera_bp
        from app.routes.image_routes import image_bp
        from app.routes.ai_routes import ai_bp  # Import AI routes
        
        app.register_blueprint(page_bp)
        app.register_blueprint(camera_bp)
        app.register_blueprint(image_bp)
        app.register_blueprint(ai_bp)  # Register AI routes
        
        print("All blueprints registered successfully (including AI routes)")
        
        # List available templates for debugging
        if os.path.exists(template_dir):
            templates = os.listdir(template_dir)
            print(f"Available templates: {templates}")
            
            # Check if mainpage.html specifically exists
            mainpage_path = os.path.join(template_dir, 'mainpage.html')
            print(f"mainpage.html exists: {os.path.exists(mainpage_path)}")
        
        # Check if JavaScript files exist
        js_dir = os.path.join(static_dir, 'javascript')
        if os.path.exists(js_dir):
            js_files = os.listdir(js_dir)
            print(f"Available JavaScript files: {js_files}")
            
            camera_js_path = os.path.join(js_dir, 'camera.js')
            print(f"camera.js exists: {os.path.exists(camera_js_path)}")
        
    except Exception as e:
        print(f"Error registering blueprints: {e}")
        import traceback
        traceback.print_exc()
        
        # At minimum, register page routes so basic pages work
        try:
            from app.routes.page_routes import page_bp
            app.register_blueprint(page_bp)
            print("Page routes registered as fallback")
        except Exception as fallback_error:
            print(f"Fallback registration failed: {fallback_error}")
    
    return app
