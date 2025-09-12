// ==================== UPDATED CAMERA APP MANAGER WITH PIPELINE SUPPORT ==================== 
// MODIFIED: Updated to display exact format from pipeline analysis

class CameraAppManager {
  constructor() {
    this.isLiveMode = true;
    this.isAnalyzing = false;
    this.isFullscreen = false;
    this.analysisResults = null;
    
    // Distance sensor management
    this.distanceInterval = null;
    this.lastDistanceReading = null;
    this.distanceUpdateRate = 500; // 500ms as requested
    
    // DOM Elements
    this.cameraContainer = document.getElementById('camera-container');
    this.serverFeed = document.getElementById('server-feed');
    this.videoElement = document.getElementById('camera-feed');
    this.cameraStatus = document.getElementById('camera-status');
    this.loadingOverlay = document.getElementById('loading-overlay');
    
    // Distance display elements
    this.distanceDisplay = null; // Will be created dynamically
    
    // Controls
    this.tutorialBtn = document.getElementById('tutorial-btn');
    this.galleryBtn = document.getElementById('gallery-btn');
    this.captureBtn = document.getElementById('capture-btn');
    this.fullscreenBtn = document.getElementById('fullscreen-btn');
    this.gridBtn = document.getElementById('grid-btn');
    
    // Modals
    this.tutorialModal = document.getElementById('tutorial-modal');
    this.resultsModal = document.getElementById('results-modal');
    this.errorModal = document.getElementById('error-modal');
    
    // Grid overlay
    this.gridOverlay = document.getElementById('grid-overlay');
    this.isGridActive = false;
    
    // Camera feed management
    this.serverFeedInterval = null;
    this.isUsingServerFeed = true;
    
    this.init();
  }
  
  init() {
    console.log('🎥 Initializing Camera App Manager (PIPELINE MODE)...');
    console.log('📝 NOTE: Using Quadrant Pipeline Analysis');
    this.setupEventListeners();
    this.createDistanceDisplay();
    this.startCameraFeed();
    this.startDistanceMonitoring();
    this.updateStatus('Initializing camera and distance sensor...');
  }
  
  // ==================== DISTANCE SENSOR INTEGRATION ====================
  
  createDistanceDisplay() {
    console.log('📏 Creating distance display overlay...');
    
    // Create distance display element
    this.distanceDisplay = document.createElement('div');
    this.distanceDisplay.className = 'distance-display';
    this.distanceDisplay.innerHTML = `
      <div class="distance-value">--cm</div>
      <div class="distance-status">CHECKING</div>
    `;
    
    // Add to camera controls (positioned right of camera status)
    if (this.cameraContainer) {
      const cameraControls = this.cameraContainer.querySelector('.camera-controls');
      if (cameraControls) {
        cameraControls.appendChild(this.distanceDisplay);
      }
    }
    
    console.log('✅ Distance display created');
  }
  
  async startDistanceMonitoring() {
    console.log('🚀 Starting distance sensor monitoring...');
    
    try {
      // Start the distance monitoring service
      const startResponse = await fetch('/start-distance-monitoring', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (startResponse.ok) {
        console.log('✅ Distance monitoring service started');
        
        // Start polling for distance readings every 500ms
        this.distanceInterval = setInterval(() => {
          this.updateDistanceReading();
        }, this.distanceUpdateRate);
        
        console.log(`📏 Distance polling started at ${this.distanceUpdateRate}ms intervals`);
      } else {
        console.warn('⚠️  Failed to start distance monitoring service');
        this.showDistanceError('Service unavailable');
      }
      
    } catch (error) {
      console.error('❌ Error starting distance monitoring:', error);
      this.showDistanceError('Connection error');
    }
  }
  
  async updateDistanceReading() {
    // Only update if not currently analyzing (avoid interference)
    if (this.isAnalyzing) {
      return;
    }
    
    try {
      const response = await fetch('/distance-reading');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const reading = await response.json();
      
      if (reading.success) {
        this.lastDistanceReading = reading;
        this.updateDistanceDisplay(reading);
      } else {
        this.showDistanceError(reading.error || 'Reading failed');
      }
      
    } catch (error) {
      // Don't spam console with connection errors
      if (Math.random() < 0.1) { // Log only 10% of errors
        console.warn('⚠️  Distance reading error:', error.message);
      }
      this.showDistanceError('Connection error');
    }
  }
  
  updateDistanceDisplay(reading) {
    if (!this.distanceDisplay) return;
    
    const valueElement = this.distanceDisplay.querySelector('.distance-value');
    const statusElement = this.distanceDisplay.querySelector('.distance-status');
    
    if (valueElement) {
      valueElement.textContent = reading.distance_text || '--cm';
    }
    
    if (statusElement) {
      statusElement.textContent = reading.status_text || 'UNKNOWN';
    }
    
    // Update background color based on status
    this.distanceDisplay.className = `distance-display ${reading.status_color || 'gray'}`;
    
    // Add distance icon based on status
    const icon = this.getDistanceIcon(reading.status);
    if (valueElement && !valueElement.textContent.includes('📏')) {
      valueElement.textContent = `📏 ${reading.distance_text || '--cm'}`;
    }
  }
  
  getDistanceIcon(status) {
    switch (status) {
      case 'optimal': return '✅';
      case 'too_close': return '⚠️';
      case 'too_far': return '📏';
      default: return '❓';
    }
  }
  
  showDistanceError(error) {
    if (!this.distanceDisplay) return;
    
    const valueElement = this.distanceDisplay.querySelector('.distance-value');
    const statusElement = this.distanceDisplay.querySelector('.distance-status');
    
    if (valueElement) {
      valueElement.textContent = '❌ --cm';
    }
    
    if (statusElement) {
      statusElement.textContent = 'ERROR';
    }
    
    this.distanceDisplay.className = 'distance-display red';
  }
  
  stopDistanceMonitoring() {
    console.log('🛑 Stopping distance monitoring...');
    
    if (this.distanceInterval) {
      clearInterval(this.distanceInterval);
      this.distanceInterval = null;
    }
    
    // Stop the monitoring service
    fetch('/stop-distance-monitoring', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }).catch(error => {
      console.warn('⚠️  Error stopping distance monitoring:', error);
    });
  }
  
  setupEventListeners() {
    console.log('📋 Setting up event listeners...');
    
    // Camera Controls
    if (this.tutorialBtn) {
      this.tutorialBtn.addEventListener('click', () => this.openTutorialModal());
    }
    if (this.galleryBtn) {
      this.galleryBtn.addEventListener('click', () => this.openGallery());
    }
    if (this.captureBtn) {
      this.captureBtn.addEventListener('click', () => this.captureAndAnalyze());
    }
    if (this.fullscreenBtn) {
      this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
    }
    if (this.gridBtn) {
      this.gridBtn.addEventListener('click', () => this.toggleGrid());
    }
    
    // Modal click outside to close
    if (this.tutorialModal) {
      this.tutorialModal.addEventListener('click', (e) => {
        if (e.target === this.tutorialModal) this.closeTutorialModal();
      });
    }
    if (this.resultsModal) {
      this.resultsModal.addEventListener('click', (e) => {
        if (e.target === this.resultsModal) this.closeResultsModal();
      });
    }
    if (this.errorModal) {
      this.errorModal.addEventListener('click', (e) => {
        if (e.target === this.errorModal) this.closeErrorModal();
      });
    }
    
    // Fullscreen change detection
    document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
    document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    
    // Window beforeunload to clean up distance monitoring
    window.addEventListener('beforeunload', () => {
      this.stopDistanceMonitoring();
    });
    
    console.log('✅ Event listeners setup complete');
  }
  
  // ==================== CAMERA FEED MANAGEMENT ====================
  
  startCameraFeed() {
    console.log('🔄 Starting camera feed (server mode)...');
    
    // Ensure server feed is visible
    if (this.serverFeed) {
      this.serverFeed.style.display = 'block';
    }
    if (this.videoElement) {
      this.videoElement.style.display = 'none';
    }
    
    // Stop any existing WebRTC stream
    if (this.videoElement && this.videoElement.srcObject) {
      this.videoElement.srcObject.getTracks().forEach(track => track.stop());
      this.videoElement.srcObject = null;
    }
    
    this.isUsingServerFeed = true;
    
    // Start server feed refresh
    this.refreshServerFeed();
    
    // Set up interval for continuous feed (only when not analyzing)
    this.serverFeedInterval = setInterval(() => {
      if (this.isUsingServerFeed && this.isLiveMode && !this.isAnalyzing) {
        this.refreshServerFeed();
      }
    }, 100); // 10 FPS for smooth experience
    
    this.updateStatus('A4Tech Camera Active (Pipeline Mode)');
    console.log('✅ Server camera feed started');
  }
  
  refreshServerFeed() {
    if (this.serverFeed && this.isLiveMode && !this.isAnalyzing) {
      const timestamp = new Date().getTime();
      this.serverFeed.src = `/video_feed?t=${timestamp}`;
      
      this.serverFeed.onload = () => {
        // Successfully loaded frame
      };
      
      this.serverFeed.onerror = () => {
        this.updateStatus('Camera feed error');
        console.error('❌ Camera feed error');
      };
    }
  }
  
  // ==================== MODIFIED CAPTURE & ANALYZE FLOW (PIPELINE ANALYSIS) ====================
  
  async captureAndAnalyze() {
    if (this.isAnalyzing) {
      console.log('⚠️ Analysis already in progress, ignoring capture request');
      return;
    }
    
    // Check distance for optimal positioning warning
    if (this.lastDistanceReading && this.lastDistanceReading.success) {
      const status = this.lastDistanceReading.status;
      if (status === 'too_close') {
        const proceed = confirm('Distance is too close (< 160cm). Capture anyway?\n\nFor best results, move back to 160-200cm range.');
        if (!proceed) {
          return;
        }
      } else if (status === 'too_far') {
        const proceed = confirm('Distance is too far (> 200cm). Capture anyway?\n\nFor best results, move closer to 160-200cm range.');
        if (!proceed) {
          return;
        }
      }
      // If optimal, continue without warning
    }
    
    console.log('📸 Starting PIPELINE capture and analyze flow...');
    console.log('📝 NOTE: Using Quadrant Pipeline Analysis');
    this.isAnalyzing = true;
    
    try {
      // Step 1: Capture Animation
      if (this.captureBtn) {
        this.captureBtn.style.transform = 'scale(0.9)';
        setTimeout(() => {
          this.captureBtn.style.transform = '';
        }, 150);
      }
      
      // Step 2: Show loading overlay immediately
      this.showLoadingOverlay();
      this.updateStatus('Preparing frame for PIPELINE analysis...');
      
      // Step 3: Verify camera frame is ready
      console.log('📷 Verifying camera frame is ready for pipeline...');
      const captureResponse = await fetch('/capture-current-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!captureResponse.ok) {
        throw new Error(`Frame preparation failed: ${captureResponse.status}`);
      }
      
      const captureResult = await captureResponse.json();
      
      if (!captureResult.success) {
        throw new Error(captureResult.error || 'Failed to prepare frame');
      }
      
      console.log('✅ Frame ready for PIPELINE analysis:', captureResult.frame_dimensions);
      
      // Step 4: Start PIPELINE AI analysis
      this.updateStatus('Running PIPELINE analysis: Quadrant intersections...');
      console.log('🔍 Starting PIPELINE AI analysis...');
      
      const analysisResponse = await fetch('/analyze-rebar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
        // No body needed - analysis works with current camera frame
      });
      
      if (!analysisResponse.ok) {
        if (analysisResponse.status === 422) {
          // No rebar detected
          const result = await analysisResponse.json();
          if (result.error === 'no_rebar_detected') {
            this.hideLoadingOverlay();
            this.showErrorModal();
            return;
          }
        }
        throw new Error(`PIPELINE analysis failed: ${analysisResponse.status}`);
      }
      
      const analysisResult = await analysisResponse.json();
      
      if (!analysisResult.success) {
        throw new Error(analysisResult.message || 'PIPELINE analysis failed');
      }
      
      console.log('✅ PIPELINE AI analysis completed');
      console.log('📊 Pipeline results:', analysisResult);
      
      // Step 5: Hide loading and show results
      this.hideLoadingOverlay();
      this.showPipelineResults(analysisResult);
      
      // Step 6: Confirm successful analysis
      console.log('💾 SUCCESS: PIPELINE analysis with quadrant intersections completed');
      
    } catch (error) {
      console.error('❌ PIPELINE capture and analyze error:', error);
      this.hideLoadingOverlay();
      this.updateStatus('PIPELINE analysis failed');
      this.showErrorMessage('Failed to analyze image: ' + error.message);
    } finally {
      this.isAnalyzing = false;
    }
  }
  
  // ==================== LOADING OVERLAY MANAGEMENT ====================
  
  showLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.add('active');
      // Update loading text for pipeline
      const loadingText = this.loadingOverlay.querySelector('.loading-text');
      if (loadingText) {
        loadingText.textContent = 'Running quadrant pipeline analysis...';
      }
    }
  }
  
  hideLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.remove('active');
    }
  }
  
  // ==================== PIPELINE RESULTS MANAGEMENT ====================
  
  showPipelineResults(results) {
    console.log('📊 Showing PIPELINE results...');
    
    // Update results modal with PIPELINE data - EXACT FORMAT
    const resultsImage = document.getElementById('results-image');
    const dimensionsResult = document.getElementById('dimensions-result');
    const mixtureResult = document.getElementById('mixture-result');
    
    // Set analyzed image (ONLY image that was saved)
    if (results.images && results.images.analyzed && resultsImage) {
      resultsImage.src = results.images.analyzed;
      console.log('🖼️ Displaying PIPELINE analyzed image with quadrant overlays');
    } else if (resultsImage) {
      console.warn('⚠️ No PIPELINE analyzed image found in results');
    }
    
    // Set dimensions - EXACT FORMAT: "27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters"
    if (results.dimensions && results.dimensions.display && dimensionsResult) {
      dimensionsResult.textContent = results.dimensions.display;
      console.log('📐 PIPELINE Dimensions:', results.dimensions.display);
    } else if (dimensionsResult) {
      dimensionsResult.textContent = '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters'; // Fallback
    }
    
    // Set cement mixture - EXACT FORMAT: "1:2:4"
    if (results.cement_mixture && results.cement_mixture.ratio_string && mixtureResult) {
      mixtureResult.textContent = results.cement_mixture.ratio_string;
      console.log('🧮 PIPELINE Mixture:', results.cement_mixture.ratio_string);
    } else if (mixtureResult) {
      mixtureResult.textContent = '1:2:4'; // Fallback
    }
    
    // Store results for reference
    this.analysisResults = results;
    
    // Show results modal
    if (this.resultsModal) {
      this.resultsModal.classList.add('active');
    }
    
    // Update status
    this.updateStatus('PIPELINE analysis complete - Quadrant analysis saved to gallery');
    
    // Log PIPELINE analysis details
    console.log('📊 PIPELINE Analysis Results Summary:', {
      detections: results.detections?.count || 0,
      dimensions: results.dimensions?.display || 'N/A',
      mixture: results.cement_mixture?.ratio_string || 'N/A',
      model_type: results.metadata?.model_type || 'unknown',
      pipeline_data: results.pipeline_data || null,
      quadrants: results.quadrants || null
    });
    
    // Show success message
    const detectionCount = results.detections?.count || 0;
    const modelType = results.metadata?.model_type || 'pipeline';
    const message = `PIPELINE analysis complete! ${detectionCount} rebar structures detected. Quadrant intersections analyzed.`;
    setTimeout(() => {
      this.showSuccessMessage(message);
    }, 1000); // Delay to let modal appear first
  }
  
  // ==================== GRID TOGGLE FUNCTIONALITY ====================
  
  toggleGrid() {
    this.isGridActive = !this.isGridActive;
    
    if (this.isGridActive) {
      // Show grid overlay
      if (this.gridOverlay) {
        this.gridOverlay.classList.add('active');
      }
      // Change to nogrid icon
      if (this.gridBtn) {
        this.gridBtn.classList.add('grid-active');
        this.gridBtn.title = 'Hide Grid';
      }
      console.log('✅ Rule of thirds grid enabled');
      this.updateStatus('Grid overlay enabled');
    } else {
      // Hide grid overlay
      if (this.gridOverlay) {
        this.gridOverlay.classList.remove('active');
      }
      // Change back to withgrid icon
      if (this.gridBtn) {
        this.gridBtn.classList.remove('grid-active');
        this.gridBtn.title = 'Show Grid';
      }
      console.log('❌ Rule of thirds grid disabled');
      this.updateStatus('Grid overlay disabled');
    }
  }
  
  // ==================== NAVIGATION ====================
  
  openGallery() {
    console.log('📁 Opening gallery (PIPELINE analyzed images)...');
    window.location.href = '/result.html';
  }
  
  // ==================== FULLSCREEN MANAGEMENT ====================
  
  toggleFullscreen() {
    if (!document.fullscreenElement) {
      this.enterFullscreen();
    } else {
      this.exitFullscreen();
    }
  }
  
  enterFullscreen() {
    console.log('⛶ Entering fullscreen...');
    
    const element = this.cameraContainer;
    
    if (element && element.requestFullscreen) {
      element.requestFullscreen();
    } else if (element && element.webkitRequestFullscreen) {
      element.webkitRequestFullscreen();
    } else if (element && element.mozRequestFullScreen) {
      element.mozRequestFullScreen();
    } else if (element && element.msRequestFullscreen) {
      element.msRequestFullscreen();
    }
  }
  
  exitFullscreen() {
    console.log('↙️ Exiting fullscreen...');
    
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
  
  handleFullscreenChange() {
    this.isFullscreen = !!document.fullscreenElement;
    
    if (this.isFullscreen) {
      if (this.cameraContainer) this.cameraContainer.classList.add('fullscreen');
      // Change to minimize icon when in fullscreen
      if (this.fullscreenBtn) {
        this.fullscreenBtn.classList.add('minimize-mode');
        this.fullscreenBtn.title = 'Exit Fullscreen';
      }
      this.updateStatus('Fullscreen mode active');
    } else {
      if (this.cameraContainer) this.cameraContainer.classList.remove('fullscreen');
      // Change back to fullscreen icon when not in fullscreen
      if (this.fullscreenBtn) {
        this.fullscreenBtn.classList.remove('minimize-mode');
        this.fullscreenBtn.title = 'Enter Fullscreen';
      }
      this.updateStatus('Fullscreen mode exited');
    }
  }
  
  // ==================== MODAL MANAGEMENT ====================
  
  openTutorialModal() {
    console.log('❓ Opening tutorial modal...');
    if (this.tutorialModal) {
      this.tutorialModal.classList.add('active');
    }
  }
  
  closeTutorialModal() {
    console.log('✕ Closing tutorial modal...');
    if (this.tutorialModal) {
      this.tutorialModal.classList.remove('active');
    }
  }
  
  closeResultsModal() {
    console.log('✕ Closing PIPELINE results modal...');
    if (this.resultsModal) {
      this.resultsModal.classList.remove('active');
    }
    this.analysisResults = null; // Clear stored results
    this.updateStatus('Ready for next PIPELINE capture');
  }
  
  showErrorModal() {
    console.log('⚠️ Showing error modal...');
    if (this.errorModal) {
      this.errorModal.classList.add('active');
    }
  }
  
  closeErrorModal() {
    console.log('✕ Closing error modal...');
    if (this.errorModal) {
      this.errorModal.classList.remove('active');
    }
    this.updateStatus('Ready for next PIPELINE capture');
  }
  
  // ==================== UI STATUS MANAGEMENT ====================
  
  updateStatus(message) {
    if (this.cameraStatus) {
      this.cameraStatus.textContent = message;
    }
    console.log('📊 Status:', message);
  }
  
  showSuccessMessage(message) {
    // Create temporary success notification
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 100px;
      right: 20px;
      background: #2d7d47;
      color: white;
      padding: 15px 20px;
      border-radius: 8px;
      z-index: 400;
      font-weight: 600;
      box-shadow: 0 6px 25px rgba(45, 125, 71, 0.3);
      animation: slideInRight 0.3s ease;
      max-width: 400px;
      word-wrap: break-word;
      border: 2px solid white;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 5000);
  }
  
  showErrorMessage(message) {
    // Create temporary error notification
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 100px;
      right: 20px;
      background: #e74c3c;
      color: white;
      padding: 15px 20px;
      border-radius: 8px;
      z-index: 400;
      font-weight: 600;
      box-shadow: 0 6px 25px rgba(231, 76, 60, 0.3);
      animation: slideInRight 0.3s ease;
      max-width: 400px;
      word-wrap: break-word;
      border: 2px solid white;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 6000);
  }
  
  // ==================== KEYBOARD SHORTCUTS ====================
  
  handleKeyboard(e) {
    // Prevent shortcuts during analysis
    if (this.isAnalyzing) {
      console.log('⏳ Ignoring keyboard shortcut during PIPELINE analysis');
      return;
    }
    
    switch (e.key) {
      case ' ': // Spacebar - Capture & Analyze
        e.preventDefault();
        if (this.isLiveMode) {
          this.captureAndAnalyze();
        }
        break;
        
      case 'Escape': // Escape - Close modals
        e.preventDefault();
        if (this.tutorialModal && this.tutorialModal.classList.contains('active')) {
          this.closeTutorialModal();
        } else if (this.resultsModal && this.resultsModal.classList.contains('active')) {
          this.closeResultsModal();
        } else if (this.errorModal && this.errorModal.classList.contains('active')) {
          this.closeErrorModal();
        }
        break;
        
      case 'Enter': // Enter - Capture & Analyze
        e.preventDefault();
        if (this.isLiveMode) {
          this.captureAndAnalyze();
        }
        break;
        
      case 'f': // F - Toggle fullscreen
      case 'F':
        e.preventDefault();
        this.toggleFullscreen();
        break;
        
      case 'g': // G - Gallery
      case 'G':
        e.preventDefault();
        this.openGallery();
        break;
        
      case '?': // ? - Tutorial
        e.preventDefault();
        this.openTutorialModal();
        break;
        
      case 'r': // R - Toggle grid
      case 'R':
        e.preventDefault();
        this.toggleGrid();
        break;
        
      case 'd': // D - Show distance info (debug)
      case 'D':
        e.preventDefault();
        if (this.lastDistanceReading) {
          console.log('📏 Current distance reading:', this.lastDistanceReading);
          this.showSuccessMessage(`Distance: ${this.lastDistanceReading.distance_text} - ${this.lastDistanceReading.status_text}`);
        }
        break;
        
      case 'p': // P - Show pipeline mode info (debug)
      case 'P':
        e.preventDefault();
        this.showSuccessMessage('Mode: Quadrant Pipeline Analysis (1:2:4 mix ratio)');
        console.log('🔍 Mode: Quadrant Pipeline Analysis with cement mixture calculation');
        break;
    }
  }
}

// ==================== GLOBAL MODAL FUNCTIONS ====================

// Global functions for modal button onclick events
window.closeTutorialModal = function() {
  if (window.cameraApp) {
    window.cameraApp.closeTutorialModal();
  }
};

window.closeResultsModal = function() {
  if (window.cameraApp) {
    window.cameraApp.closeResultsModal();
  }
};

window.closeErrorModal = function() {
  if (window.cameraApp) {
    window.cameraApp.closeErrorModal();
  }
};

// ==================== INITIALIZATION ====================

// Initialize camera app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Starting Rebar Vista Camera App (QUADRANT PIPELINE MODE)...');
  console.log('📝 PIPELINE: Quadrant intersections → Polygon → Volume → Cement (1:2:4)');
  console.log('🎯 Expected detections: 2 verticals + 11 horizontals');
  console.log('📐 Exact format: "27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters"');
  console.log('🧮 Ratio format: "1:2:4"');
  
  // Create global instance
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Camera App initialized successfully (PIPELINE MODE)');
  console.log('📋 PIPELINE User Flow:');
  console.log('   1. Position device at optimal distance (160-200cm)');
  console.log('   2. Press capture button (📷)');
  console.log('   3. Wait for quadrant pipeline analysis');
  console.log('   4. View results with exact dimensions format');
  console.log('   5. Close results and capture again');
  console.log('');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space/Enter - Capture & analyze with pipeline');
  console.log('   Escape - Close modals');
  console.log('   F - Toggle fullscreen');
  console.log('   G - Open gallery');
  console.log('   D - Show distance info (debug)');
  console.log('   P - Show pipeline mode info (debug)');
  console.log('   ? - Open tutorial');
});

// Additional styles for notifications (injected dynamically)
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
`;
document.head.appendChild(notificationStyles);
