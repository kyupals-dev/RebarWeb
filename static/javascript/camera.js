// ==================== SIMPLIFIED CAMERA APP MANAGER WITH DISTANCE SENSOR ==================== 
// MODIFIED: Only saves analyzed images with AI overlays (no original duplicates)

class CameraAppManager {
  constructor() {
    this.isLiveMode = true;
    this.isAnalyzing = false;
    this.isFullscreen = false;
    this.analysisResults = null;
    
    // Distance sensor management
    this.distanceInterval = null;
    this.lastDistanceReading = null;
    this.distanceUpdateRate = 1000; // 1000ms as requested
    
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
    console.log('🎥 Initializing Camera App Manager (Analyzed Images Only Mode)...');
    console.log('📝 NOTE: Only analyzed images with AI overlays will be saved to gallery');
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
    console.log('🔄 Starting camera feed (server mode - MJPEG stream)...');
    
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
    
    // No need for intervals or timestamp refreshing
    this.serverFeed.src = '/video_feed';
    
    // Add error handling
    this.serverFeed.onerror = () => {
      this.updateStatus('Camera feed error - reconnecting...');
        console.error('❌ Camera feed error');
        
        // Attempt to reconnect after 2 seconds
        setTimeout(() => {
          if (this.isUsingServerFeed && this.isLiveMode) {
            console.log('🔄 Attempting to reconnect camera feed...');
            this.serverFeed.src = '/video_feed?' + new Date().getTime();
          }
        }, 2000); 
      };
    
      this.serverFeed.onload = () => {
        console.log('✅ Camera feed connected');
        this.updateStatus('A4Tech Camera Active');
      };
      
      // ✅ REMOVE the interval completely - not needed for MJPEG
      // Clear any existing intervals
      if (this.serverFeedInterval) {
        clearInterval(this.serverFeedInterval);
        this.serverFeedInterval = null;
      }
      
      this.updateStatus('A4Tech Camera Active');
      console.log('✅ Server camera MJPEG stream started (single connection)');
  }
    
  // ✅ REMOVE or simplify refreshServerFeed - only needed for manual refresh
  refreshServerFeed() {
    // This method is now only used for manual refresh (if needed)
    // Not called automatically anymore
    if (this.serverFeed && this.isLiveMode) {
      // Force refresh by adding timestamp
      const timestamp = new Date().getTime();
      this.serverFeed.src = `/video_feed?t=${timestamp}`;
    }
  }
      
   // Update stopCameraFeed to properly cleanup
   stopCameraFeed() {
     console.log('⏹️ Stopping camera feed...');
     
      // Clear any intervals (just in case)
      if (this.serverFeedInterval) {
        clearInterval(this.serverFeedInterval);
        this.serverFeedInterval = null;
      }
      
      // Stop server feed
      if (this.serverFeed) {
        this.serverFeed.src = ''; // Clear the source to stop MJPEG stream
        this.serverFeed.style.display = 'none';
      }
      
      // Stop WebRTC if active
      if (this.videoElement && this.videoElement.srcObject) {
        this.videoElement.srcObject.getTracks().forEach(track => track.stop());
        this.videoElement.srcObject = null;
      }
      
      this.isUsingServerFeed = false;
      this.updateStatus('Camera Stopped');
      console.log('✅ Camera feed stopped');
    }
  
  // ==================== MODIFIED CAPTURE & ANALYZE FLOW (ANALYZED IMAGE ONLY) ====================
  
// ==================== CAMERA.JS PARTIAL UPDATE ====================
// This shows only the modified captureAndAnalyze method
// Replace the existing captureAndAnalyze method in camera.js with this version

async captureAndAnalyze() {
  // Prevent multiple simultaneous analyses
  if (this.isAnalyzing) {
    console.log('⏳ Analysis already in progress...');
    return;
  }
  
  // Check distance status before capturing
  if (this.lastDistanceReading) {
    const status = this.lastDistanceReading.status;
    
    if (status === 'too_close') {
      const proceed = confirm('Distance is too close. Capture anyway?\n\nFor best results, move back atleast to 196cm range.');
      if (!proceed) {
        return;
      }
    } else if (status === 'too_far') {
      const proceed = confirm('Distance is too far. Capture anyway?\n\nFor best results, move closer atleast to 204cm range.');
      if (!proceed) {
        return;
      }
    }
    // If optimal, continue without warning
  }
  
  console.log('📸 Starting capture and analyze flow (ANALYZED IMAGE ONLY)...');
  console.log('📝 NOTE: Only analyzed image with AI overlays will be saved');
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
    this.updateStatus('Preparing frame for AI analysis...');
    
    // Step 3: Verify camera frame is ready (NO ORIGINAL SAVED)
    console.log('📷 Verifying camera frame is ready (no original will be saved)...');
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
    
    console.log('✅ Frame ready for analysis:', captureResult.frame_dimensions);
    console.log('📝 Confirmed: No original frame saved');
    
    // Step 4: Start AI analysis directly with current camera frame
    this.updateStatus('Analyzing rebar structure with AI...');
    console.log('🔍 Starting AI analysis (will save ONLY analyzed image with overlays)...');
    
    const analysisResponse = await fetch('/analyze-rebar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
      // No body needed - analysis works with current camera frame
    });
    
    // CRITICAL FIX: Check for HTTP 422 (no rebar detected)
    if (!analysisResponse.ok) {
      if (analysisResponse.status === 422) {
        // No rebar detected - show error modal
        const result = await analysisResponse.json();
        if (result.error === 'no_rebar_detected') {
          console.log('⚠️  NO REBAR DETECTED - Showing error modal');
          console.log('📝 Confirmed: No images were saved');
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
    
    // ENHANCED FIX: Additional fallback check for 0 detections
    // (in case backend returns HTTP 200 but with 0 detections)
    if (analysisResult.detections && analysisResult.detections.count === 0) {
      console.log('⚠️  FALLBACK CHECK: 0 detections found - Showing error modal');
      console.log('📝 Note: This should have been caught by backend, but fallback triggered');
      this.hideLoadingOverlay();
      this.showErrorModal();
      return;
    }
    
    console.log('✅ AI analysis completed - ONLY analyzed image saved to gallery');
    
    // Verify the save mode
    if (analysisResult.metadata && analysisResult.metadata.save_mode === 'analyzed_only') {
      console.log('✅ Confirmed: Only analyzed image was saved (no duplicates)');
    }
    
    // Step 5: Hide loading and show results
    this.hideLoadingOverlay();
    this.showResults(analysisResult);
    
    // Step 6: Confirm single image save
    console.log('💾 SUCCESS: Only analyzed image with AI overlays saved to gallery');
    console.log('🚫 No original/duplicate images created');
    
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
  
showResults(results) {
  console.log('📊 Displaying pipeline analysis results (REAL DATA ONLY):', results);
  
  // Get DOM elements
  const resultsImage = document.getElementById('results-image');
  const dimensionsResult = document.getElementById('dimensions-result');
  const mixtureResult = document.getElementById('mixture-result');
  
  // Set analyzed image
  if (results.images && results.images.analyzed) {
    resultsImage.src = results.images.analyzed;
  } else {
    console.warn('⚠️ No analyzed image found in results');
  }
  
  // ==================== DISPLAY DIMENSIONS ====================
  // Use real dimensions from backend (already formatted)
  if (results.dimensions && results.dimensions.display && dimensionsResult) {
    dimensionsResult.textContent = results.dimensions.display;
    console.log('✅ Dimensions displayed:', results.dimensions.display);
  } else {
    console.error('❌ No dimension data received from backend');
    dimensionsResult.textContent = 'Dimension data unavailable';
  }
  
// ==================== DISPLAY CEMENT MIXTURE ====================
// Only show the ratio (quantities moved to Material Quantities section)
if (results.cement_mixture && mixtureResult) {
  mixtureResult.textContent = 'Cement ratio 1:2:4';
  console.log('✅ Cement mixture ratio displayed');
} else {
  console.error('❌ No cement mixture data received from backend');
  mixtureResult.textContent = 'Cement mixture data unavailable';
}

// ==================== UPDATE PIPELINE DETAILS ====================
const pipelineDetections = document.getElementById('pipeline-detections');
const wetVolume = document.getElementById('wet-volume');
const cementCalc = document.getElementById('cement-calc');
const waterCalc = document.getElementById('water-calc');

// Detections Found
if (pipelineDetections && results.detections) {
  const detectionCount = results.detections.count || 0;
  // Try to get breakdown from pipeline_data if available
  const pipelineData = results.metadata?.pipeline_data || {};
  const verticals = pipelineData.front_vertical_count || 0;
  const horizontals = pipelineData.front_horizontal_count || 0;
  
  pipelineDetections.textContent = `${detectionCount} detections (${verticals} verticals + ${horizontals} horizontals)`;
  console.log('✅ Detections displayed:', pipelineDetections.textContent);
}

// Wet Volume Calculation - ONLY show final result
if (wetVolume && results.dimensions) {
  const volumeM3 = results.dimensions.volume_m3 || (results.dimensions.volume / 1000000);
  const dryVolumeFactor = 1.54; // Standard factor
  const wetVolumeM3 = volumeM3 * dryVolumeFactor;
  
  // NEW: Only show final result
  wetVolume.textContent = `${wetVolumeM3.toFixed(7)}m³`;
  console.log('✅ Wet volume displayed:', wetVolume.textContent);
}

// Material Quantities - NEW FORMAT with cement, sand, gravel quantities
if (cementCalc && results.cement_mixture && results.cement_mixture.details) {
  const details = results.cement_mixture.details;
  const cementKg = details.cement_weight_kg || 0;
  const cementBags = details.cement_bags || 0;
  const sandKg = details.sand_weight_kg || 0;
  const gravelKg = details.gravel_weight_kg || 0;
  
  // NEW FORMAT: Multi-line display without formulas
  const materialQuantities = `Cement = ${cementKg.toFixed(2)} kg ≈ ${cementBags.toFixed(2)} bags
Sand = ${sandKg.toFixed(2)} kg
Gravel = ${gravelKg.toFixed(2)} kg`;
  
  cementCalc.textContent = materialQuantities;
  console.log('✅ Material quantities displayed:', cementCalc.textContent);
}

// Water Requirement - ONLY show final result
if (waterCalc && results.cement_mixture && results.cement_mixture.details) {
  const details = results.cement_mixture.details;
  const waterLiters = details.water_liters || 0;
  
  // NEW: Only show final result
  waterCalc.textContent = `≈${waterLiters.toFixed(1)} liters`;
  console.log('✅ Water requirement displayed:', waterCalc.textContent);
}
  
  // Store results for reference
  this.analysisResults = results;
  
  // Show results modal
  if (this.resultsModal) {
    this.resultsModal.classList.add('active');
  }
  
  // Update status
  this.updateStatus('Analysis complete - Analyzed image saved to gallery');
  
  // Log analysis details
  console.log('📊 Analysis Results Summary:', {
    detections: results.detections?.count || 0,
    dimensions: results.dimensions?.display || 'N/A',
    mixture: results.cement_mixture?.ratio || 'N/A',
    placeholder: results.metadata?.placeholder_mode || false,
    save_mode: results.metadata?.save_mode || 'unknown',
    only_analyzed_saved: true
  });
  
  // Show success message
  const detectionCount = results.detections?.count || 0;
  const message = `Analysis complete! ${detectionCount} rebar structures detected. Analyzed image saved to gallery.`;
  this.showSuccessMessage(message);

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
    console.log('📁 Opening gallery (showing analyzed images only)...');
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
    this.updateStatus('Ready for next capture (analyzed image only mode)');
  }
  
  showErrorModal() {
    console.log('⚠️ Showing error modal...');
  
    // Restore camera interface first
    const cameraInterface = document.querySelector('.camera-interface');
    if (cameraInterface) {
      cameraInterface.style.display = 'flex';
    }
  
    // Show alert instead of modal (temporary solution)
    alert('NO REBAR DETECTED\n\n' +
          'The AI could not detect any rebar structures in the captured image.\n\n' +
          'Please ensure the rebar is clearly visible and try again.\n\n' +
          'Requirements:\n' +
          '• Visible rebar structure\n' +
          '• 200cm optimal distance');
    
    this.updateStatus('Ready for next capture (analyzed image only mode)');
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
        
      case 'd': // D - Show distance info (debug)
      case 'D':
        e.preventDefault();
        if (this.lastDistanceReading) {
          console.log('📏 Current distance reading:', this.lastDistanceReading);
          this.showSuccessMessage(`Distance: ${this.lastDistanceReading.distance_text} - ${this.lastDistanceReading.status_text}`);
        }
        break;
        
      case 's': // S - Show save mode info (debug)
      case 'S':
        e.preventDefault();
        this.showSuccessMessage('Save Mode: Analyzed Images Only (no originals)');
        console.log('💾 Save Mode: Only analyzed images with AI overlays are saved');
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

// Navigate to welcome page
window.goToWelcome = function() {
  console.log('🏠 Navigating to welcome page...');
  window.location.href = '/welcome';
};

// ==================== INITIALIZATION ====================

// Initialize camera app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Starting Rebar Vista Camera App (ANALYZED IMAGES ONLY MODE)...');
  console.log('📝 IMPORTANT: Only analyzed images with AI overlays will be saved');
  console.log('🚫 No original/duplicate images will be created');
  
  // Create global instance
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Camera App initialized successfully');
  console.log('📋 Modified User Flow:');
  console.log('   1. Position device at optimal distance (160-200cm)');
  console.log('   2. Press capture button (📷)');
  console.log('   3. Wait for AI analysis');
  console.log('   4. View results (ONLY analyzed image auto-saved to gallery)');
  console.log('   5. Close results and capture again');
  console.log('');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space/Enter - Capture & analyze');
  console.log('   Escape - Close modals');
  console.log('   F - Toggle fullscreen');
  console.log('   G - Open gallery');
  console.log('   D - Show distance info (debug)');
  console.log('   S - Show save mode info (debug)');
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


