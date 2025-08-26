// ==================== CAMERA APP MANAGER ==================== 

class CameraAppManager {
  constructor() {
    this.isLiveMode = true;
    this.isPreviewMode = false;
    this.isFullscreen = false;
    this.isAnalyzing = false;
    this.capturedImageData = null;
    this.analysisResults = null;
    
    // DOM Elements
    this.cameraContainer = document.getElementById('camera-container');
    this.serverFeed = document.getElementById('server-feed');
    this.videoElement = document.getElementById('camera-feed');
    this.capturedPreview = document.getElementById('captured-preview');
    this.cameraStatus = document.getElementById('camera-status');
    this.loadingOverlay = document.getElementById('loading-overlay');
    
    // Controls
    this.tutorialBtn = document.getElementById('tutorial-btn');
    this.closePreviewBtn = document.getElementById('close-preview-btn');
    this.galleryBtn = document.getElementById('gallery-btn');
    this.captureBtn = document.getElementById('capture-btn');
    this.previewControls = document.getElementById('preview-controls');
    this.analyzeBtn = document.getElementById('analyze-btn');
    this.deleteBtn = document.getElementById('delete-btn');
    this.fullscreenBtn = document.getElementById('fullscreen-btn');
    
    // Modals
    this.tutorialModal = document.getElementById('tutorial-modal');
    this.resultsModal = document.getElementById('results-modal');
    this.errorModal = document.getElementById('error-modal');
    
    // Camera feed management
    this.serverFeedInterval = null;
    this.isUsingServerFeed = true;
    this.isUsingWebRTC = false;
    
    this.init();
  }
  
  init() {
    console.log('🎥 Initializing Camera App Manager...');
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
    if (this.closePreviewBtn) {
      this.closePreviewBtn.addEventListener('click', () => this.exitPreviewMode());
    }
    if (this.galleryBtn) {
      this.galleryBtn.addEventListener('click', () => this.openGallery());
    }
    if (this.captureBtn) {
      this.captureBtn.addEventListener('click', () => this.captureImage());
    }
    if (this.analyzeBtn) {
      this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }
    if (this.deleteBtn) {
      this.deleteBtn.addEventListener('click', () => this.deletePreview());
    }
    if (this.fullscreenBtn) {
      this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
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
    this.isUsingWebRTC = false;
    
    // Start server feed refresh
    this.refreshServerFeed();
    
    // Set up interval for continuous feed
    this.serverFeedInterval = setInterval(() => {
      if (this.isUsingServerFeed && this.isLiveMode) {
        this.refreshServerFeed();
      }
    }, 100); // 10 FPS for smooth experience
    
    this.updateStatus('A4Tech Camera Active');
    console.log('✅ Server camera feed started');
  }
  
  refreshServerFeed() {
    if (this.serverFeed && this.isLiveMode) {
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
  
  // ==================== CAPTURE FUNCTIONALITY ====================
  
  async captureImage() {
    console.log('📸 Capturing image...');
    this.updateStatus('Capturing image...');
    
    try {
      // Add capture animation
      if (this.captureBtn) {
        this.captureBtn.style.transform = 'scale(0.9)';
        setTimeout(() => {
          this.captureBtn.style.transform = '';
        }, 150);
      }
      
      // Capture from server feed
      const response = await fetch('/capture-current-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Store captured image data
        this.capturedImageData = {
          filename: result.filename,
          url: `/static/captured_images/${result.filename}`
        };
        
        // Switch to preview mode
        this.enterPreviewMode();
        
        console.log('✅ Image captured successfully:', result.filename);
        this.updateStatus('Image captured - Review or analyze');
      } else {
        throw new Error(result.error);
      }
      
    } catch (error) {
      console.error('❌ Capture error:', error);
      this.updateStatus('Capture failed');
      this.showErrorMessage('Failed to capture image: ' + error.message);
    }
  }
  
  // ==================== PREVIEW MODE MANAGEMENT ====================
  
  enterPreviewMode() {
    console.log('🖼️ Entering preview mode...');
    
    if (!this.capturedImageData) {
      console.error('No captured image data available');
      return;
    }
    
    // Update state
    this.isLiveMode = false;
    this.isPreviewMode = true;
    
    // Hide live feed, show captured image
    if (this.serverFeed) this.serverFeed.style.display = 'none';
    if (this.videoElement) this.videoElement.style.display = 'none';
    if (this.capturedPreview) {
      this.capturedPreview.style.display = 'block';
      this.capturedPreview.src = this.capturedImageData.url;
    }
    
    // Update UI controls
    if (this.captureBtn) this.captureBtn.style.display = 'none';
    if (this.previewControls) this.previewControls.classList.add('active');
    if (this.closePreviewBtn) this.closePreviewBtn.classList.add('active');
    
    // Stop server feed interval
    if (this.serverFeedInterval) {
      clearInterval(this.serverFeedInterval);
      this.serverFeedInterval = null;
    }
    
    // Add fade-in animation
    if (this.capturedPreview) {
      this.capturedPreview.classList.add('fade-in');
      setTimeout(() => {
        this.capturedPreview.classList.remove('fade-in');
      }, 300);
    }
  }
  
  exitPreviewMode() {
    console.log('🔄 Exiting preview mode...');
    
    // Update state
    this.isLiveMode = true;
    this.isPreviewMode = false;
    
    // Show live feed, hide captured image
    if (this.capturedPreview) this.capturedPreview.style.display = 'none';
    if (this.serverFeed) this.serverFeed.style.display = 'block';
    
    // Update UI controls
    if (this.captureBtn) this.captureBtn.style.display = 'flex';
    if (this.previewControls) this.previewControls.classList.remove('active');
    if (this.closePreviewBtn) this.closePreviewBtn.classList.remove('active');
    
    // Clear captured image data
    this.capturedImageData = null;
    
    // Restart camera feed
    this.startCameraFeed();
  }
  
  // ==================== AI ANALYSIS ====================
  
  async analyzeImage() {
    if (!this.capturedImageData) {
      console.error('No image to analyze');
      return;
    }
    
    console.log('🔍 Starting AI analysis...');
    this.isAnalyzing = true;
    
    // Show loading overlay
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.add('active');
    }
    this.updateStatus('Analyzing rebar structure...');
    
    try {
      // Call AI analysis endpoint
      const response = await fetch('/analyze-rebar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          filename: this.capturedImageData.filename 
        })
      });
      
      if (!response.ok) {
        if (response.status === 422) {
          // No rebar detected
          const result = await response.json();
          if (result.error === 'no_rebar_detected') {
            this.showErrorModal();
            return;
          }
        }
        throw new Error(`Analysis failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Show analysis results
        this.showAnalysisResults(result);
        console.log('✅ Analysis completed successfully');
      } else {
        throw new Error(result.message || 'Analysis failed');
      }
      
    } catch (error) {
      console.error('❌ Analysis error:', error);
      this.showErrorModal();
    } finally {
      this.isAnalyzing = false;
      if (this.loadingOverlay) {
        this.loadingOverlay.classList.remove('active');
      }
    }
  }
  
  // ==================== RESULTS MANAGEMENT ====================
  
  showAnalysisResults(results) {
    // Update results modal with actual data from AI
    const resultsImage = document.getElementById('results-image');
    const dimensionsResult = document.getElementById('dimensions-result');
    const mixtureResult = document.getElementById('mixture-result');
    
    // Set analyzed image (with AI overlays)
    if (results.images && results.images.analyzed && resultsImage) {
      resultsImage.src = results.images.analyzed;
    } else if (resultsImage) {
      resultsImage.src = this.capturedImageData.url;
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
    
    // Store results for saving
    this.analysisResults = results;
    
    // Show results modal
    if (this.resultsModal) {
      this.resultsModal.classList.add('active');
    }
    
    // Log analysis details
    console.log('📊 Analysis Results:', {
      detections: results.detections?.count || 0,
      dimensions: results.dimensions?.display || 'N/A',
      mixture: results.cement_mixture?.ratio || 'N/A',
      placeholder: results.metadata?.placeholder_mode || false
    });
  }
  
  async saveResults() {
    if (!this.analysisResults) {
      console.error('No results to save');
      return;
    }
    
    try {
      // The analyzed image is already saved by the AI service
      // Just close modal and return to live mode
      this.closeResultsModal();
      this.exitPreviewMode();
      
      this.updateStatus('Results saved to gallery');
      console.log('💾 Results saved successfully');
      
      // Show success message with analysis details
      const detectionCount = this.analysisResults.detections?.count || 0;
      const message = `Analysis complete! ${detectionCount} rebar structures detected. Results saved to gallery.`;
      this.showSuccessMessage(message);
      
    } catch (error) {
      console.error('❌ Save error:', error);
      this.showErrorMessage('Failed to save results: ' + error.message);
    }
  }
  
  // ==================== IMAGE DELETION ====================
  
  async deletePreview() {
    if (!this.capturedImageData) {
      console.error('No image to delete');
      return;
    }
    
    const confirmed = confirm('Delete this captured image?');
    if (!confirmed) return;
    
    try {
      console.log('🗑️ Deleting preview image...');
      
      const response = await fetch(`/delete-image/${encodeURIComponent(this.capturedImageData.filename)}`, {
        method: 'DELETE'
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ Image deleted successfully');
        this.updateStatus('Image deleted');
        
        // Return to live mode
        this.exitPreviewMode();
        
        // Show brief message
        this.showSuccessMessage('Image deleted successfully');
      } else {
        throw new Error(result.error);
      }
      
    } catch (error) {
      console.error('❌ Delete error:', error);
      this.showErrorMessage('Failed to delete image: ' + error.message);
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
      if (this.fullscreenBtn) this.fullscreenBtn.innerHTML = '↙️';
      this.updateStatus('Fullscreen mode active');
    } else {
      if (this.cameraContainer) this.cameraContainer.classList.remove('fullscreen');
      if (this.fullscreenBtn) this.fullscreenBtn.innerHTML = '⛶';
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
      background: #2ecc71;
      color: white;
      padding: 15px 20px;
      border-radius: 8px;
      z-index: 400;
      font-weight: 500;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      animation: slideInRight 0.3s ease;
      max-width: 300px;
      word-wrap: break-word;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 4000);
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
      font-weight: 500;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      animation: slideInRight 0.3s ease;
      max-width: 300px;
      word-wrap: break-word;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 5000);
  }
  
  // ==================== KEYBOARD SHORTCUTS ====================
  
  handleKeyboard(e) {
    // Prevent shortcuts during analysis
    if (this.isAnalyzing) return;
    
    switch (e.key) {
      case ' ': // Spacebar - Capture
        e.preventDefault();
        if (this.isLiveMode) {
          this.captureImage();
        }
        break;
        
      case 'Escape': // Escape - Close modals or exit preview
        e.preventDefault();
        if (this.tutorialModal && this.tutorialModal.classList.contains('active')) {
          this.closeTutorialModal();
        } else if (this.resultsModal && this.resultsModal.classList.contains('active')) {
          this.closeResultsModal();
        } else if (this.errorModal && this.errorModal.classList.contains('active')) {
          this.closeErrorModal();
        } else if (this.isPreviewMode) {
          this.exitPreviewMode();
        }
        break;
        
      case 'Enter': // Enter - Analyze in preview mode
        e.preventDefault();
        if (this.isPreviewMode && !this.isAnalyzing) {
          this.analyzeImage();
        }
        break;
        
      case 'Delete': // Delete - Delete preview
      case 'Backspace':
        e.preventDefault();
        if (this.isPreviewMode && !this.isAnalyzing) {
          this.deletePreview();
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

window.saveResults = function() {
  if (window.cameraApp) {
    window.cameraApp.saveResults();
  }
};

// ==================== INITIALIZATION ====================

// Initialize camera app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Starting Rebar Vista Camera App...');
  
  // Create global instance
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Camera App initialized successfully');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space - Capture image');
  console.log('   Enter - Analyze image (in preview mode)');
  console.log('   Escape - Close modals or exit preview');
  console.log('   Delete - Delete preview image');
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
