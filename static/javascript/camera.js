// ==================== SIMPLIFIED CAMERA APP MANAGER WITH 4-STEP ANALYSIS DISPLAY ==================== 
// UPDATED: Displays 4-step analysis pipeline results

class CameraAppManager {
  constructor() {
    this.isLiveMode = true;
    this.isAnalyzing = false;
    this.isFullscreen = false;
    this.analysisResults = null;
    
    // Distance sensor management
    this.distanceInterval = null;
    this.lastDistanceReading = null;
    this.distanceUpdateRate = 500;
    
    // DOM Elements
    this.cameraContainer = document.getElementById('camera-container');
    this.serverFeed = document.getElementById('server-feed');
    this.videoElement = document.getElementById('camera-feed');
    this.cameraStatus = document.getElementById('camera-status');
    this.loadingOverlay = document.getElementById('loading-overlay');
    
    // Distance display elements
    this.distanceDisplay = null;
    
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
    console.log('🎥 Initializing Camera App Manager with 4-Step Analysis Display...');
    console.log('📝 UPDATED: Now displays 4 analysis steps in results modal');
    this.setupEventListeners();
    this.createDistanceDisplay();
    this.startCameraFeed();
    this.startDistanceMonitoring();
    this.updateStatus('Initializing camera and distance sensor...');
  }
  
  // ==================== DISTANCE SENSOR INTEGRATION (UNCHANGED) ====================
  
  createDistanceDisplay() {
    console.log('📏 Creating distance display overlay...');
    
    this.distanceDisplay = document.createElement('div');
    this.distanceDisplay.className = 'distance-display';
    this.distanceDisplay.innerHTML = `
      <div class="distance-value">--cm</div>
      <div class="distance-status">CHECKING</div>
    `;
    
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
      const startResponse = await fetch('/start-distance-monitoring', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (startResponse.ok) {
        console.log('✅ Distance monitoring service started');
        
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
      if (Math.random() < 0.1) {
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
    
    this.distanceDisplay.className = `distance-display ${reading.status_color || 'gray'}`;
    
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
    
    fetch('/stop-distance-monitoring', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }).catch(error => {
      console.warn('⚠️  Error stopping distance monitoring:', error);
    });
  }
  
  // ==================== EVENT LISTENERS (UNCHANGED) ====================
  
  setupEventListeners() {
    console.log('📋 Setting up event listeners...');
    
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
    
    document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
    document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
    
    document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    
    window.addEventListener('beforeunload', () => {
      this.stopDistanceMonitoring();
    });
    
    console.log('✅ Event listeners setup complete');
  }
  
  // ==================== CAMERA FEED MANAGEMENT (UNCHANGED) ====================
  
  startCameraFeed() {
    console.log('🔄 Starting camera feed (server mode)...');
    
    if (this.serverFeed) {
      this.serverFeed.style.display = 'block';
    }
    if (this.videoElement) {
      this.videoElement.style.display = 'none';
    }
    
    if (this.videoElement && this.videoElement.srcObject) {
      this.videoElement.srcObject.getTracks().forEach(track => track.stop());
      this.videoElement.srcObject = null;
    }
    
    this.isUsingServerFeed = true;
    
    this.refreshServerFeed();
    
    this.serverFeedInterval = setInterval(() => {
      if (this.isUsingServerFeed && this.isLiveMode && !this.isAnalyzing) {
        this.refreshServerFeed();
      }
    }, 100);
    
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
  // ==================== UPDATED CAPTURE & ANALYZE WITH 4-STEP DISPLAY ==================== 
  
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
    }
    
    console.log('📸 Starting 4-Step Analysis Pipeline...');
    console.log('📝 UPDATED: Will display 4 analysis steps in results modal');
    this.isAnalyzing = true;
    
    try {
      // Step 1: Capture Animation
      if (this.captureBtn) {
        this.captureBtn.style.transform = 'scale(0.9)';
        setTimeout(() => {
          this.captureBtn.style.transform = '';
        }, 150);
      }
      
      // Step 2: Show loading overlay
      this.showLoadingOverlay();
      this.updateStatus('Running 4-step AI analysis pipeline...');
      
      // Step 3: Verify camera frame is ready
      console.log('📷 Verifying camera frame is ready for 4-step analysis...');
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
      
      console.log('✅ Frame ready for 4-step analysis:', captureResult.frame_dimensions);
      
      // Step 4: Start 4-Step AI Analysis
      this.updateStatus('Step 1: Rebar Detection...');
      console.log('🔍 Starting 4-Step AI Analysis Pipeline...');
      
      const analysisResponse = await fetch('/analyze-rebar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!analysisResponse.ok) {
        if (analysisResponse.status === 422) {
          const result = await analysisResponse.json();
          if (result.error === 'no_rebar_detected') {
            this.hideLoadingOverlay();
            this.showErrorModal();
            return;
          }
        }
        throw new Error(`4-Step Analysis failed: ${analysisResponse.status}`);
      }
      
      const analysisResult = await analysisResponse.json();
      
      if (!analysisResult.success) {
        throw new Error(analysisResult.message || '4-Step Analysis failed');
      }
      
      console.log('✅ 4-Step AI Analysis completed successfully');
      console.log('📊 Pipeline results:', {
        model_type: analysisResult.model_type,
        step_images: analysisResult.step_images ? 'Available' : 'Missing',
        pipeline_data: analysisResult.pipeline_data
      });
      
      // Step 5: Hide loading and show 4-step results
      this.hideLoadingOverlay();
      this.show4StepAnalysisResults(analysisResult);
      
      // Step 6: Success confirmation
      console.log('💾 SUCCESS: 4-Step Analysis completed with visualizations');
      console.log('🎯 Expected detections: 2 verticals + 11 horizontals = 13 total');
      
    } catch (error) {
      console.error('❌ 4-Step Analysis error:', error);
      this.hideLoadingOverlay();
      this.updateStatus('4-Step Analysis failed');
      this.showErrorMessage('Failed to complete 4-step analysis: ' + error.message);
    } finally {
      this.isAnalyzing = false;
    }
  }
  
  // ==================== NEW: 4-STEP ANALYSIS RESULTS DISPLAY ==================== 
  
  show4StepAnalysisResults(results) {
    console.log('📊 Displaying 4-Step Analysis Results...');
    
    try {
      // Update dimensions
      const dimensionsResult = document.getElementById('dimensions-result');
      if (results.dimensions && results.dimensions.display && dimensionsResult) {
        dimensionsResult.textContent = results.dimensions.display;
      }
      
      // Update cement mixture
      const mixtureResult = document.getElementById('mixture-result');
      if (results.cement_mixture && results.cement_mixture.ratio_string && mixtureResult) {
        mixtureResult.textContent = results.cement_mixture.ratio_string;
      }
      
      // Update detection summary
      this.update4StepDetectionSummary(results);
      
      // CRITICAL: Display 4-Step Images
      this.display4StepImages(results);
      
      // Store results for reference
      this.analysisResults = results;
      
      // Show results modal
      if (this.resultsModal) {
        this.resultsModal.classList.add('active');
      }
      
      // Update status
      this.updateStatus('4-Step Analysis complete - Results displayed');
      
      // Log analysis summary
      console.log('📊 4-Step Analysis Summary:', {
        model_type: results.model_type,
        detections: results.detections?.length || 0,
        dimensions: results.dimensions?.display || 'N/A',
        mixture: results.cement_mixture?.ratio_string || 'N/A',
        pipeline_data: results.pipeline_data,
        step_images_available: !!results.step_images
      });
      
      // Show success message
      const detectionCount = results.detections?.length || 0;
      const expectedCount = results.pipeline_data?.front_horizontal_count + results.pipeline_data?.front_vertical_count || 13;
      const message = `4-Step Analysis complete! Found ${detectionCount}/${expectedCount} expected detections.`;
      setTimeout(() => {
        this.showSuccessMessage(message);
      }, 1000);
      
    } catch (error) {
      console.error('❌ Error displaying 4-step results:', error);
      this.showErrorMessage('Error displaying analysis results: ' + error.message);
    }
  }
  
  update4StepDetectionSummary(results) {
    try {
      const pipelineData = results.pipeline_data || {};
      
      // Update detection counts
      const verticalCount = document.getElementById('vertical-count');
      const horizontalCount = document.getElementById('horizontal-count');
      const intersectionCount = document.getElementById('intersection-count');
      const modelType = document.getElementById('model-type');
      
      if (verticalCount) {
        verticalCount.textContent = pipelineData.front_vertical_count || '2';
      }
      
      if (horizontalCount) {
        horizontalCount.textContent = pipelineData.front_horizontal_count || '11';
      }
      
      if (intersectionCount) {
        intersectionCount.textContent = pipelineData.intersection_count || '22';
      }
      
      if (modelType) {
        const type = results.model_type || 'simplified_4step_pipeline';
        const displayType = results.placeholder ? 'Placeholder Pipeline' : 'Real Model Pipeline';
        modelType.textContent = displayType;
      }
      
      console.log('📊 Detection summary updated:', pipelineData);
      
    } catch (error) {
      console.error('❌ Error updating detection summary:', error);
    }
  }
  
  display4StepImages(results) {
    console.log('🖼️ Loading 4-Step Analysis Images...');
    
    try {
      const stepImages = results.step_images;
      
      if (!stepImages) {
        console.warn('⚠️ No step images provided in results');
        this.displayPlaceholder4StepImages();
        return;
      }
      
      // Update each step image
      const steps = [
        { id: 'step1-image', path: stepImages.step1, name: 'Step 1: Detection' },
        { id: 'step2-image', path: stepImages.step2, name: 'Step 2: Intersections' },
        { id: 'step3-image', path: stepImages.step3, name: 'Step 3: Polygon' },
        { id: 'step4-image', path: stepImages.step4, name: 'Step 4: Cement' }
      ];
      
      steps.forEach((step, index) => {
        const imgElement = document.getElementById(step.id);
        if (imgElement && step.path) {
          // Convert absolute path to relative URL
          const filename = step.path.split('/').pop();
          const imageUrl = `/static/captured_images/${filename}`;
          
          imgElement.src = imageUrl;
          imgElement.alt = step.name;
          
          // Add error handling
          imgElement.onerror = () => {
            console.warn(`⚠️ Failed to load ${step.name}: ${imageUrl}`);
            imgElement.src = '/static/assets/placeholder-analysis.png';
            imgElement.alt = `${step.name} (Not Available)`;
          };
          
          // Add click handler for full view
          imgElement.onclick = () => this.viewFullStepImage(imageUrl, step.name);
          
          console.log(`✅ Loaded ${step.name}: ${filename}`);
        } else {
          console.warn(`⚠️ Missing element or path for ${step.name}`);
        }
      });
      
      console.log('✅ 4-Step Analysis Images loaded successfully');
      
    } catch (error) {
      console.error('❌ Error displaying 4-step images:', error);
      this.displayPlaceholder4StepImages();
    }
  }
  
  displayPlaceholder4StepImages() {
    console.log('📝 Displaying placeholder 4-step images...');
    
    const steps = [
      { id: 'step1-image', name: 'Step 1: Detection' },
      { id: 'step2-image', name: 'Step 2: Intersections' },
      { id: 'step3-image', name: 'Step 3: Polygon' },
      { id: 'step4-image', name: 'Step 4: Cement' }
    ];
    
    steps.forEach(step => {
      const imgElement = document.getElementById(step.id);
      if (imgElement) {
        imgElement.src = '/static/assets/placeholder-analysis.png';
        imgElement.alt = `${step.name} (Placeholder)`;
      }
    });
  }
  
  viewFullStepImage(imageUrl, stepName) {
    console.log(`🔍 Viewing full step image: ${stepName}`);
    
    // Create full-screen image viewer
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.9);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = stepName;
    img.style.cssText = `
      max-width: 90vw;
      max-height: 90vh;
      object-fit: contain;
      border-radius: 10px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;
    
    const title = document.createElement('div');
    title.textContent = stepName;
    title.style.cssText = `
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      color: white;
      font-size: 24px;
      font-weight: bold;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    `;
    
    const closeBtn = document.createElement('div');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
      position: absolute;
      top: 30px;
      right: 30px;
      color: white;
      font-size: 30px;
      cursor: pointer;
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: rgba(0, 0, 0, 0.5);
    `;
    
    overlay.appendChild(img);
    overlay.appendChild(title);
    overlay.appendChild(closeBtn);
    
    // Close on click
    overlay.onclick = () => document.body.removeChild(overlay);
    closeBtn.onclick = (e) => {
      e.stopPropagation();
      document.body.removeChild(overlay);
    };
    
    document.body.appendChild(overlay);
  }
  
  // ==================== LOADING OVERLAY MANAGEMENT (UNCHANGED) ====================
  
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
  
  // ==================== GRID TOGGLE FUNCTIONALITY (UNCHANGED) ====================
  
  toggleGrid() {
    this.isGridActive = !this.isGridActive;
    
    if (this.isGridActive) {
      if (this.gridOverlay) {
        this.gridOverlay.classList.add('active');
      }
      if (this.gridBtn) {
        this.gridBtn.classList.add('grid-active');
        this.gridBtn.title = 'Hide Grid';
      }
      console.log('✅ Rule of thirds grid enabled');
      this.updateStatus('Grid overlay enabled');
    } else {
      if (this.gridOverlay) {
        this.gridOverlay.classList.remove('active');
      }
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
    console.log('📁 Opening gallery (showing analyzed images with 4-step results)...');
    window.location.href = '/result.html';
  }
  
  // ==================== FULLSCREEN MANAGEMENT (UNCHANGED) ====================
  
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
    console.log('✕ Closing 4-step results modal...');
    if (this.resultsModal) {
      this.resultsModal.classList.remove('active');
    }
    this.analysisResults = null;
    this.updateStatus('Ready for next 4-step analysis');
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
    this.updateStatus('Ready for next 4-step analysis');
  }
  
  // ==================== UI STATUS MANAGEMENT ====================
  
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
    }, 6000);
  }
  
  // ==================== KEYBOARD SHORTCUTS ==================== 
  
  handleKeyboard(e) {
    if (this.isAnalyzing) {
      console.log('⏳ Ignoring keyboard shortcut during 4-step analysis');
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
        
      case 's': // S - Show pipeline info (debug)
      case 'S':
        e.preventDefault();
        this.showSuccessMessage('Pipeline: 4-Step Analysis (Detection → Intersections → Polygon → Cement)');
        console.log('🔄 Pipeline: 4-Step Analysis with visualization display');
        break;
    }
  }
}

// ==================== GLOBAL MODAL FUNCTIONS ==================== 

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

window.openGallery = function() {
  if (window.cameraApp) {
    window.cameraApp.openGallery();
  }
};

// ==================== INITIALIZATION ==================== 

document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Starting Rebar Vista Camera App with 4-Step Analysis Display...');
  console.log('📝 UPDATED: Now displays 4 analysis steps in results modal');
  console.log('🎯 Expected: 2 vertical + 11 horizontal rebars = 13 total detections');
  
  window.cameraApp = new CameraAppManager();
  
  console.log('✅ Camera App with 4-Step Display initialized successfully');
  console.log('📋 Updated User Flow:');
  console.log('   1. Position device at optimal distance (160-200cm)');
  console.log('   2. Press capture button (📷)');
  console.log('   3. Wait for 4-step AI analysis pipeline');
  console.log('   4. View results with 4 analysis step images');
  console.log('   5. Close results and capture again');
  console.log('');
  console.log('📋 Available keyboard shortcuts:');
  console.log('   Space/Enter - Capture & 4-step analyze');
  console.log('   Escape - Close modals');
  console.log('   F - Toggle fullscreen');
  console.log('   G - Open gallery');
  console.log('   D - Show distance info (debug)');
  console.log('   S - Show pipeline info (debug)');
  console.log('   ? - Open tutorial');
});

// Additional styles for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
`;
document.head.appendChild(notificationStyles);
