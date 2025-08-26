from flask import Blueprint, render_template, send_from_directory

# Create a Blueprint for page routes
page_bp = Blueprint('pages', __name__)

@page_bp.route('/')
def splash():
    """Splash screen route"""
    return render_template('splash.html')

@page_bp.route('/welcome')
def welcome():
    """Welcome page route"""
    return render_template('welcome.html')

@page_bp.route('/mainpage.html')
def mainpage():
    """Main page route"""
    return render_template('mainpage.html')

@page_bp.route('/result.html')
def result():
    """Result/gallery page route"""
    return render_template('result.html')

# Static file routes (if you want to keep them)
@page_bp.route('/manifest.json')
def manifest():
    """PWA manifest file"""
    return send_from_directory('static', 'manifest.json')

@page_bp.route('/sw.js')
def service_worker():
    """Service worker file"""
    return send_from_directory('static', 'sw.js')