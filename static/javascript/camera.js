// ==================== REFACTORED CAMERA APP MANAGER ====================
// FOCUSED: Only camera integration, delegates buttons and modals to separate modules
// FIXED: Proper error handling and method checking

class CameraAppManager {
  constructor() {
    // Core camera state
    this.isLiveMode = true;
    this.isAnalyzing = false;
    this.isFullscreen = false;
    this.analysisResults = null;
    
    // Distance sensor management
    this.distanceInterval = null;
    this.lastDistanceReading = null;
    this.distanceUpdateRate = 500; // 500ms as requested
    
    // DOM Elements - Camera specific
    this.cameraContainer = document.getElementById('camera-container');
    this.serverFeed = document.getElementById('server-feed');
    this.videoElement = document.getElementById('camera-feed');
    this.cameraStatus = document.getElementById('camera-status');
    this.loadingOverlay = document.getElementById('loading-overlay');
    
    // DOM Elements - Controls
    this.tutorialBtn = document.getElementById('tutorial-btn');
    this.galleryBtn = document.getElementById('gallery-btn');
    this.captureBtn = document.getElementById('capture-btn');
    this.fullscreenBtn = document.getElementById('fullscreen-btn');
    this.gridBtn = document.getElementById('grid-btn');
    
    // DOM Elements - Modals
    this.tutorialModal = document.getElementById('tutorial-modal');
    this.resultsModal = document.getElementById('results-modal');
    this.errorModal = document.getElementById('error-modal');
    
    // Grid overlay
    this.gridOverlay = document.getElementById('grid-overlay');
    this.isGridActive = false;
    
    // Camera feed management
    this.serverFeedInterval = null;
    this.isUsingServerFeed = true;
    
    // Distance display
    this.distanceDisplay = null;
    
    // Initialize modules
    this.buttonManager = null;
    this.modalManager = null;
    
    this.init();
  }
  
  init() {
    console.log('🎥 Initializing Camera App Manager (Pipeline Mode)...');
    console.log('📝 NOTE: Only analyzed images with pipeline steps will be saved');
    
    // Initialize delegated modules - FIXED: Check if classes exist
    try {
      if (typeof ButtonManager !== 'undefined') {
        this.buttonManager = new ButtonManager(this);
        console.log('✅ Button manager initialized');
      } else {
        console.warn('⚠️ ButtonManager class not found, using fallback');
        this.setupFallbackButtonListeners();
      }
      
      if (typeof ModalManager !== 'undefined') {
        this.modalManager = new ModalManager(this);
        console.log('✅ Modal manager initialized');
      } else {
        console.warn('⚠️ ModalManager class not found, using fallback');
        this.setupFallbackModalMethods();
      }
    } catch (error) {
      console.error('❌ Error initializing managers:', error);
      this.setupFallbackButtonListeners();
      this.setupFallbackModalMethods();
    }
    
    // Camera-specific initialization
    this.createDistanceDisplay();
    this.startCameraFeed();
    this.startDistanceMonitoring();
    this.setupCameraEventListeners();
    
    this.updateStatus('Initializing camera and distance sensor...');
  }

  setupFallbackButtonListeners() {
    console.log('🔧 Setting up fallback button listeners...');
    
    // Tutorial Button
    if (this.tutorialBtn) {
      this.tutorialBtn.addEventListener('click', () => this.openTutorialModal());
    }

    // Gallery Button
    if (this.galleryBtn) {
      this.galleryBtn.addEventListener('click', () => this.openGallery());
    }

    // Capture Button
    if (this.captureBtn) {
      this.captureBtn.addEventListener('click', () => this.captureAndAnalyze());
    }

    // Fullscreen Button
    if (this.fullscreenBtn) {
      this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
    }

    // Grid Toggle Button
    if (this.gridBtn) {
      this.gridBtn.addEventListener('click', () => this.toggleGrid());
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    
    console.log('✅ Fallback button listeners setup complete');
  }

  setupFallbackModalMethods() {
    console.log('🔧 Setting up fallback modal methods...');
    
    // Create basic modal methods if ModalManager is not available
    this.openTutorialModal = this.openTutorialModal || (() => {
      if (this.tutorialModal) {
        this.tutorialModal.classList.add('active');
      }
    });
    
    this.closeTutorialModal = this.closeTutorialModal || (() => {
      if (this.tutorialModal) {
        this.tutorialModal.classList.remove('active');
      }
    });
    
    this.closeResultsModal = this.closeResultsModal || (() => {
      if (this.resultsModal) {
        this.resultsModal.classList.remove('active');
      }
      this.analysisResults = null;
    });
    
    this.closeErrorModal = this.closeErrorModal || (() => {
      if (this.errorModal) {
        this.errorModal.classList.remove('active');
      }
    });
    
    console.log('✅ Fallback modal methods setup complete');
  }

  handleKeyboardShortcuts(e) {
    // Ignore if user is typing in an input field
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      return;
    }

    switch (e.key.toLowerCase()) {
      case ' ': // Space - Capture
      case 'enter':
        e.preventDefault();
        if (!this.isAnalyzing) {
          this.captureAndAnalyze();
        }
        break;

      case 'escape': // Escape - Close modals
        e.preventDefault();
        this.closeAllModals();
        break;

      case 'f': // F - Fullscreen
        e.preventDefault();
        this.toggleFullscreen();
        break;

      case 'g': // G - Gallery
        e.preventDefault();
        this.openGallery();
        break;

      case '?': // ? - Tutorial
        e.preventDefault();
        this.openTutorialModal();
        break;

      case 'r': // R - Toggle grid
        e.preventDefault();
        this.toggleGrid();
        break;
    }
  }

  setupCameraEventListeners() {
    console.log('📹 Setting up camera-specific event listeners...');
    
    // Fullscreen change detection
    document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
    document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
    
    // Modal click outside to close - FIXED: Add null checks
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
    
    // Window cleanup
    window.addEventListener('beforeunload', () => {
      this.stopDistanceMonitoring();
    });
    
    console.log('✅ Camera event listeners setup complete');
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
    
    // Set up interval for continuous feed
    this.serverFeedInterval = setInterval(() => {
      if (this.isUsingServerFeed && this.isLiveMode && !this.isAnalyzing) {
        this.refreshServerFeed();
      }
    }, 100); // 10 FPS for smooth experience
    
    this.updateStatus('A4Tech Camera Active');
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

  stopCameraFeed() {
    console.log('⏹️ Stopping camera feed...');
    
    if (this.serverFeedInterval) {
      clearInterval(this.serverFeedInterval);
      this.serverFeedInterval = null;
    }
    
    if (this.videoElement && this.videoElement.srcObject) {
      this.videoElement.srcObject.getTracks().forEach(track => track.stop());
      this.videoElement.srcObject = null;
    }
    
    this.isLiveMode = false;
    this.updateStatus('Camera stopped');
  }

  // ==================== PIPELINE ANALYSIS WORKFLOW ====================
  
  async captureAndAnalyze() {
    if (this.isAnalyzing) {
      console.log('⚠️ Analysis already in progress');
      return;
    }
    
    console.log('📸 Starting pipeline capture and analysis...');
    
    try {
      // Set analyzing state
      this.isAnalyzing = true;
      this.isLiveMode = false;
      
      // Update UI
      this.updateStatus('Running quadrant pipeline analysis...');
      this.showLoadingOverlay();
      this.setButtonsEnabled(false);
      
      // Stop camera feed during analysis
      if (this.serverFeedInterval) {
        clearInterval(this.serverFeedInterval);
        this.serverFeedInterval = null;
      }
      
      // Call AI analysis API
      const response = await fetch('/analyze-rebar', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          mode: 'pipeline',
          capture_timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) {
        throw new Error(`Analysis request failed: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Handle results
      await this.handleAnalysisResults(result);
      
    } catch (error) {
      console.error('❌ Capture and analysis error:', error);
      this.handleAnalysisError(error);
    } finally {
      // Always restore camera state
      this.isAnalyzing = false;
      this.isLiveMode = true;
      this.hideLoadingOverlay();
      this.setButtonsEnabled(true);
      this.startCameraFeed();
    }
  }

  async handleAnalysisResults(result) {
    console.log('📊 Processing pipeline analysis results...');
    
    if (result.success && result.dimensions && result.cement_mixture) {
      console.log('✅ Pipeline analysis successful');
      
      // Save to gallery automatically - FIXED: Check if modalManager exists
      if (this.modalManager && typeof this.modalManager.saveResultsToGallery === 'function') {
        await this.modalManager.saveResultsToGallery(result);
      } else {
        // Fallback gallery save
        await this.saveResultsToGalleryFallback(result);
      }
      
      // Show results modal with pipeline data
      if (this.modalManager && typeof this.modalManager.showResultsModal === 'function') {
        this.modalManager.showResultsModal(result);
      } else {
        this.showResultsModalFallback(result);
      }
      
      // Success feedback
      this.showSuccessMessage('Pipeline analysis complete! Results saved to gallery.');
      
    } else if (result.no_detection || (result.success === false && result.error)) {
      console.log('⚠️ No rebar structures detected');
      
      if (this.modalManager && typeof this.modalManager.showErrorModal === 'function') {
        this.modalManager.showErrorModal(result.error || 'No rebar structures detected in image');
      } else {
        this.showErrorModalFallback(result.error || 'No rebar structures detected in image');
      }
      
    } else {
      console.error('❌ Unexpected analysis result format:', result);
      if (this.modalManager && typeof this.modalManager.showErrorModal === 'function') {
        this.modalManager.showErrorModal('Analysis completed but results are incomplete');
      } else {
        this.showErrorModalFallback('Analysis completed but results are incomplete');
      }
    }
  }

  async saveResultsToGalleryFallback(result) {
    try {
      console.log('💾 Saving analysis results to gallery (fallback)...');
      
      const metadata = {
        timestamp: new Date().toISOString(),
        analysis_type: 'pipeline_quadrant',
        dimensions: result.dimensions,
        cement_mixture: result.cement_mixture,
        detections: result.num_detections,
        model_type: result.model_type,
        quadrant_info: result.quadrant_info
      };

      const response = await fetch('/save-to-gallery', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          analyzed_image_path: result.analyzed_image_path,
          metadata: metadata,
          step_images: result.step_images || {}
        })
      });

      if (response.ok) {
        console.log('✅ Results saved to gallery successfully (fallback)');
      } else {
        console.error('❌ Failed to save to gallery (fallback)');
      }
    } catch (error) {
      console.error('❌ Error saving to gallery (fallback):', error);
    }
  }

  showResultsModalFallback(result) {
    console.log('📊 Showing results modal (fallback)...');
    
    if (this.resultsModal) {
      // Update basic content
      const dimensionElement = document.getElementById('dimension-result');
      const mixtureElement = document.getElementById('mixture-result');
      
      if (dimensionElement && result.dimensions) {
        dimensionElement.textContent = result.dimensions.display || 'N/A';
      }
      
      if (mixtureElement && result.cement_mixture) {
        mixtureElement.textContent = '1:2:4';
      }
      
      this.resultsModal.classList.add('active');
    }
  }

  showErrorModalFallback(errorMessage) {
    console.log('⚠️ Showing error modal (fallback)...');
    
    if (this.errorModal) {
      const errorTextElement = this.errorModal.querySelector('p');
      if (errorTextElement && errorMessage) {
        errorTextElement.textContent = errorMessage;
      }
      this.errorModal.classList.add('active');
    }
  }

  handleAnalysisError(error) {
    console.error('❌ Analysis error:', error);
    if (this.modalManager && typeof this.modalManager.showErrorModal === 'function') {
      this.modalManager.showErrorModal(`Analysis failed: ${error.message}`);
    } else {
      this.showErrorModalFallback(`Analysis failed: ${error.message}`);
    }
    this.showErrorMessage(`Analysis failed: ${error.message}`);
  }

  // ==================== DISTANCE SENSOR INTEGRATION ====================
  
  createDistanceDisplay() {
    console.log('📏 Creating distance sensor display...');
    
    if (this.distanceDisplay) {
      return; // Already created
    }
    
    this.distanceDisplay = document.createElement('div');
    this.distanceDisplay.className = 'distance-display';
    this.distanceDisplay.innerHTML = `
      <div class="distance-value">--cm</div>
      <div class="distance-status">CHECKING</div>
    `;
    
    // Style the distance display
    this.distanceDisplay.style.cssText = `
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px 15px;
      border-radius: 8px;
      font-family: 'Arial', sans-serif;
      z-index: 200;
      border: 2px solid white;
      min-width: 120px;
    `;
    
    if (this.cameraContainer) {
      this.cameraContainer.appendChild(this.distanceDisplay);
    }
    
    console.log('✅ Distance display created');
  }

  startDistanceMonitoring() {
    console.log('📏 Starting distance sensor monitoring...');
    
    // Initial distance reading
    this.updateDistanceReading();
    
    // Set up interval for continuous monitoring
    this.distanceInterval = setInterval(() => {
      this.updateDistanceReading();
    }, this.distanceUpdateRate);
    
    console.log(`✅ Distance monitoring started (${this.distanceUpdateRate}ms interval)`);
  }

  async updateDistanceReading() {
    try {
      const response = await fetch('/get-distance', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        this.lastDistanceReading = data;
        this.updateDistanceDisplay(data);
      } else {
        console.warn('⚠️ Distance sensor API error:', response.status);
      }
    } catch (error) {
      console.warn('⚠️ Distance sensor update error:', error);
    }
  }

  updateDistanceDisplay(data) {
    if (!this.distanceDisplay || !data) return;
    
    const valueElement = this.distanceDisplay.querySelector('.distance-value');
    const statusElement = this.distanceDisplay.querySelector('.distance-status');
    
    if (valueElement && statusElement) {
      valueElement.textContent = data.distance_text || '--cm';
      statusElement.textContent = data.status_text || 'UNKNOWN';
      
      // Update colors based on status
      let backgroundColor = 'rgba(0, 0, 0, 0.8)';
      if (data.status === 'optimal') {
        backgroundColor = 'rgba(46, 125, 50, 0.9)'; // Green
      } else if (data.status === 'too_close' || data.status === 'too_far') {
        backgroundColor = 'rgba(211, 47, 47, 0.9)'; // Red
      }
      
      this.distanceDisplay.style.background = backgroundColor;
    }
  }

  stopDistanceMonitoring() {
    console.log('⏹️ Stopping distance monitoring...');
    
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

  // ==================== UI CONTROLS ====================
  
  toggleGrid() {
    console.log('⚏ Toggling grid overlay...');
    this.isGridActive = !this.isGridActive;
    
    if (this.gridOverlay) {
      this.gridOverlay.style.display = this.isGridActive ? 'block' : 'none';
    }
    
    this.updateStatus(this.isGridActive ? 'Grid overlay enabled' : 'Grid overlay disabled');
  }

  toggleFullscreen() {
    console.log('🔲 Toggling fullscreen mode...');
    
    if (!document.fullscreenElement) {
      if (this.cameraContainer && this.cameraContainer.requestFullscreen) {
        this.cameraContainer.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  handleFullscreenChange() {
    this.isFullscreen = !!document.fullscreenElement;
    
    if (this.isFullscreen) {
      if (this.cameraContainer) this.cameraContainer.classList.add('fullscreen');
      if (this.fullscreenBtn) {
        this.fullscreenBtn.classList.add('minimize-mode');
        this.fullscreenBtn.title = 'Exit Fullscreen';
      }
      this.updateStatus('Fullscreen mode active');
    } else {
      if (this.cameraContainer) this.cameraContainer.classList.remove('fullscreen');
      if (this.fullscreenBtn) {
        this.fullscreenBtn.classList.remove('minimize-mode');
        this.fullscreenBtn.title = 'Enter Fullscreen';
      }
      this.updateStatus('Fullscreen mode exited');
    }
  }

  openGallery() {
    console.log('🖼️ Opening gallery...');
    window.location.href = '/result.html';
  }

  // ==================== MODAL DELEGATION FALLBACKS ====================
  
  openTutorialModal() {
    if (this.modalManager && typeof this.modalManager.openTutorialModal === 'function') {
      this.modalManager.openTutorialModal();
    } else if (this.tutorialModal) {
      this.tutorialModal.classList.add('active');
    }
  }
  
  closeTutorialModal() {
    if (this.modalManager && typeof this.modalManager.closeTutorialModal === 'function') {
      this.modalManager.closeTutorialModal();
    } else if (this.tutorialModal) {
      this.tutorialModal.classList.remove('active');
    }
  }
  
  closeResultsModal() {
    if (this.modalManager && typeof this.modalManager.closeResultsModal === 'function') {
      this.modalManager.closeResultsModal();
    } else if (this.resultsModal) {
      this.resultsModal.classList.remove('active');
      this.analysisResults = null;
    }
  }
  
  closeErrorModal() {
    if (this.modalManager && typeof this.modalManager.closeErrorModal === 'function') {
      this.modalManager.closeErrorModal();
    } else if (this.errorModal) {
      this.errorModal.classList.remove('active');
    }
  }

  closeAllModals() {
    this.closeTutorialModal();
    this.closeResultsModal();
    this.closeErrorModal();
  }

  // ==================== UI FEEDBACK ====================
  
  showLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.style.display = 'flex';
    }
  }
  
  hideLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.style.display = 'none';
    }
  }
  
  updateStatus(message) {
    if (this.cameraStatus) {
      this.cameraStatus.textContent = message;
    }
    console.log('📊 Status:', message);
  }
  
  showSuccessMessage(message) {
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
    }, 8000);
  }

  // Enable/disable buttons during analysis
  setButtonsEnabled(enabled) {
    const buttons = [
      this.captureBtn,
      this.galleryBtn,
      this.fullscreenBtn,
      this.gridBtn
    ];

    buttons.forEach(button => {
      if (button) {
        button.disabled = !enabled;
        if (enabled) {
          button.classList.remove('disabled');
        } else {
          button.classList.add('disabled');
        }
      }
    });

    // Tutorial button should always remain enabled
    if (this.tutorialBtn) {
      this.tutorialBtn.disabled = false;
      this.tutorialBtn.classList.remove('disabled');
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
  console.log('🚀 Starting Rebar Vista Camera App (Pipeline Mode)...');
  console.log('📝 IMPORTANT: Only analyzed images with pipeline steps will be saved');
  console.log('🚫 No original/duplicate images will be created');
  
  // Create global instance
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Camera App initialized successfully');
  console.log('📋 Pipeline Analysis Flow:');
  console.log('   1. Position device at optimal distance (160-200cm)');
  console.log('   2. Press capture button (📷)');
  console.log('   3. Wait for quadrant intersection analysis');
  console.log('   4. View 4-step pipeline results');
  console.log('   5. Automatic save to gallery with metadata');
  console.log('');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space/Enter - Capture & analyze');
  console.log('   Escape - Close modals');
  console.log('   F - Toggle fullscreen');
  console.log('   G - Open gallery');
  console.log('   R - Toggle grid');
  console.log('   ? - Open tutorial');
});

// Additional styles for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  
  .pressed {
    transform: scale(0.95);
    transition: transform 0.1s ease;
  }
  
  .disabled {
    opacity: 0.5;
    pointer-events: none;
  }
`;
document.head.appendChild(notificationStyles);
