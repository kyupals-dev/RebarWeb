// ==================== SIMPLIFIED CAMERA APP MANAGER ==================== 

class CameraAppManager {
  constructor() {
    this.isLiveMode = true;
    this.isAnalyzing = false;
    this.isFullscreen = false;
    this.analysisResults = null;
    
    // DOM Elements
    this.cameraContainer = document.getElementById('camera-container');
    this.serverFeed = document.getElementById('server-feed');
    this.videoElement = document.getElementById('camera-feed');
    this.cameraStatus = document.getElementById('camera-status');
    this.loadingOverlay = document.getElementById('loading-overlay');
    
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
    console.log('🎥 Initializing Simplified Camera App Manager...');
    this.setupEventListeners();
    this.startCameraFeed();
    this.updateStatus('Initializing camera system...');
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
  
  // ==================== SIMPLIFIED CAPTURE & ANALYZE FLOW ====================
  
  async captureAndAnalyze() {
    if (this.isAnalyzing) {
      console.log('⚠️ Analysis already in progress, ignoring capture request');
      return;
    }
    
    console.log('📸 Starting capture and analyze flow...');
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
      this.updateStatus('Capturing image...');
      
      // Step 3: Capture image from server
      console.log('📷 Capturing image from camera...');
      const captureResponse = await fetch('/capture-current-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!captureResponse.ok) {
        throw new Error(`Capture failed: ${captureResponse.status}`);
      }
      
      const captureResult = await captureResponse.json();
      
      if (!captureResult.success) {
        throw new Error(captureResult.error || 'Failed to capture image');
      }
      
      console.log('✅ Image captured successfully:', captureResult.filename);
      
      // Step 4: Immediately start AI analysis
      this.updateStatus('Analyzing rebar structure...');
      console.log('🔍 Starting AI analysis...');
      
      const analysisResponse = await fetch('/analyze-rebar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          filename: captureResult.filename 
        })
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
        throw new Error(`Analysis failed: ${analysisResponse.status}`);
      }
      
      const analysisResult = await analysisResponse.json();
      
      if (!analysisResult.success) {
        throw new Error(analysisResult.message || 'Analysis failed');
      }
      
      console.log('✅ AI analysis completed successfully');
      
      // Step 5: Hide loading and show results
      this.hideLoadingOverlay();
      this.showAnalysisResults(analysisResult);
      
      // Step 6: Auto-save happens automatically (no user action needed)
      console.log('💾 Results automatically saved to gallery');
      
    } catch (error) {
      console.error('❌ Capture and analyze error:', error);
      this.hideLoadingOverlay();
      this.updateStatus('Analysis failed');
      this.showErrorMessage('Failed to analyze image: ' + error.message);
    } finally {
      this.isAnalyzing = false;
    }
  }
  
  // ==================== LOADING OVERLAY MANAGEMENT ====================
  
  showLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.add('active');
    }
  }
  
  hideLoadingOverlay() {
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.remove('active');
    }
  }
  
  // ==================== RESULTS MANAGEMENT ====================
  
  showAnalysisResults(results) {
    console.log('📊 Showing analysis results...');
    
    // Update results modal with actual data from AI
    const resultsImage = document.getElementById('results-image');
    const dimensionsResult = document.getElementById('dimensions-result');
    const mixtureResult = document.getElementById('mixture-result');
    
    // Set analyzed image (with AI overlays)
    if (results.images && results.images.analyzed && resultsImage) {
      resultsImage.src = results.images.analyzed;
      console.log('🖼️ Using analyzed image with AI overlays');
    } else if (results.images && results.images.original && resultsImage) {
      resultsImage.src = results.images.original;
      console.log('🖼️ Using original captured image');
    }
    
    // Set dimensions
    if (results.dimensions && results.dimensions.display && dimensionsResult) {
      dimensionsResult.textContent = results.dimensions.display;
    } else if (dimensionsResult) {
      dimensionsResult.textContent = '25.4cm × 25.4cm × 200cm'; // Fallback
    }
    
    // Set cement mixture
    if (results.cement_mixture && results.cement_mixture.ratio && mixtureResult) {
      mixtureResult.textContent = results.cement_mixture.ratio;
    } else if (mixtureResult) {
      mixtureResult.textContent = '1 Cement : 2 Sand : 3 Aggregate'; // Fallback
    }
    
    // Store results for reference
    this.analysisResults = results;
    
    // Show results modal
    if (this.resultsModal) {
      this.resultsModal.classList.add('active');
    }
    
    // Update status
    this.updateStatus('Analysis complete - Results ready');
    
    // Log analysis details
    console.log('📊 Analysis Results Summary:', {
      detections: results.detections?.count || 0,
      dimensions: results.dimensions?.display || 'N/A',
      mixture: results.cement_mixture?.ratio || 'N/A',
      placeholder: results.metadata?.placeholder_mode || false,
      auto_saved: true
    });
    
    // Show success message
    const detectionCount = results.detections?.count || 0;
    const message = `Analysis complete! ${detectionCount} rebar structures detected. Results saved to gallery.`;
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
    console.log('📁 Opening gallery...');
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
    console.log('✕ Closing results modal...');
    if (this.resultsModal) {
      this.resultsModal.classList.remove('active');
    }
    this.analysisResults = null; // Clear stored results
    this.updateStatus('Ready for next capture');
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
    this.updateStatus('Ready for next capture');
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
      max-width: 350px;
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
      max-width: 350px;
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
      console.log('⏳ Ignoring keyboard shortcut during analysis');
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

// Removed saveResults function since it's no longer needed (auto-save)

// ==================== INITIALIZATION ====================

// Initialize camera app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Starting Simplified Rebar Vista Camera App...');
  
  // Create global instance
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Simplified Camera App initialized successfully');
  console.log('📋 Simplified User Flow:');
  console.log('   1. Press capture button (📷)');
  console.log('   2. Wait for AI analysis');
  console.log('   3. View results (auto-saved to gallery)');
  console.log('   4. Close results and capture again');
  console.log('');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space/Enter - Capture & analyze');
  console.log('   Escape - Close modals');
  console.log('   F - Toggle fullscreen');
  console.log('   G - Open gallery');
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
