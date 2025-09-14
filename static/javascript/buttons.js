// ==================== BUTTON MANAGEMENT MODULE ====================
// Handles all button interactions for the Rebar Vista camera interface

class ButtonManager {
  constructor(cameraApp) {
    this.cameraApp = cameraApp;
    this.setupButtonListeners();
  }

  setupButtonListeners() {
    console.log('🔘 Setting up button event listeners...');
    
    // Tutorial Button
    if (this.cameraApp.tutorialBtn) {
      this.cameraApp.tutorialBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleTutorialClick();
      });
    }

    // Gallery Button
    if (this.cameraApp.galleryBtn) {
      this.cameraApp.galleryBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleGalleryClick();
      });
    }

    // Capture Button - Main analysis trigger
    if (this.cameraApp.captureBtn) {
      this.cameraApp.captureBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleCaptureClick();
      });
    }

    // Fullscreen Button
    if (this.cameraApp.fullscreenBtn) {
      this.cameraApp.fullscreenBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleFullscreenClick();
      });
    }

    // Grid Toggle Button
    if (this.cameraApp.gridBtn) {
      this.cameraApp.gridBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleGridClick();
      });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    
    console.log('✅ Button listeners setup complete');
  }

  handleTutorialClick() {
    console.log('❓ Tutorial button clicked');
    this.cameraApp.openTutorialModal();
  }

  handleGalleryClick() {
    console.log('🖼️ Gallery button clicked');
    this.cameraApp.openGallery();
  }

  async handleCaptureClick() {
    console.log('📷 Capture button clicked - starting pipeline analysis');
    
    if (this.cameraApp.isAnalyzing) {
      console.log('⚠️ Analysis already in progress');
      this.cameraApp.showErrorMessage('Analysis already in progress. Please wait...');
      return;
    }

    // Check distance before capture
    if (this.cameraApp.lastDistanceReading && this.cameraApp.lastDistanceReading.success) {
      const status = this.cameraApp.lastDistanceReading.status;
      if (status === 'too_close' || status === 'too_far') {
        const message = status === 'too_close' 
          ? 'Distance is too close (< 160cm). For best results, move back to 160-200cm range.'
          : 'Distance is too far (> 200cm). For best results, move closer to 160-200cm range.';
        
        const proceed = confirm(`${message}\n\nCapture anyway?`);
        if (!proceed) {
          return;
        }
      }
    }

    await this.cameraApp.captureAndAnalyze();
  }

  handleFullscreenClick() {
    console.log('🔲 Fullscreen button clicked');
    this.cameraApp.toggleFullscreen();
  }

  handleGridClick() {
    console.log('⚏ Grid button clicked');
    this.cameraApp.toggleGrid();
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
        if (!this.cameraApp.isAnalyzing) {
          this.handleCaptureClick();
        }
        break;

      case 'escape': // Escape - Close modals
        e.preventDefault();
        this.cameraApp.closeAllModals();
        break;

      case 'f': // F - Fullscreen
        e.preventDefault();
        this.handleFullscreenClick();
        break;

      case 'g': // G - Gallery
        e.preventDefault();
        this.handleGalleryClick();
        break;

      case '?': // ? - Tutorial
        e.preventDefault();
        this.handleTutorialClick();
        break;

      case 'r': // R - Toggle grid
        e.preventDefault();
        this.handleGridClick();
        break;

      case 'd': // D - Show distance info (debug)
        e.preventDefault();
        this.showDistanceDebugInfo();
        break;

      case 's': // S - Show save mode info (debug)
        e.preventDefault();
        this.showSaveModeInfo();
        break;
    }
  }

  showDistanceDebugInfo() {
    if (this.cameraApp.lastDistanceReading) {
      console.log('📏 Current distance reading:', this.cameraApp.lastDistanceReading);
      this.cameraApp.showSuccessMessage(
        `Distance: ${this.cameraApp.lastDistanceReading.distance_text} - ${this.cameraApp.lastDistanceReading.status_text}`
      );
    } else {
      this.cameraApp.showErrorMessage('No distance reading available');
    }
  }

  showSaveModeInfo() {
    this.cameraApp.showSuccessMessage('Save Mode: Analyzed Images Only (no originals)');
    console.log('💾 Save Mode: Only analyzed images with AI overlays are saved');
  }

  // Enable/disable buttons during analysis
  setButtonsEnabled(enabled) {
    const buttons = [
      this.cameraApp.captureBtn,
      this.cameraApp.galleryBtn,
      this.cameraApp.fullscreenBtn,
      this.cameraApp.gridBtn
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
    if (this.cameraApp.tutorialBtn) {
      this.cameraApp.tutorialBtn.disabled = false;
      this.cameraApp.tutorialBtn.classList.remove('disabled');
    }
  }

  // Visual feedback for button presses
  addButtonFeedback(button, duration = 200) {
    if (!button) return;
    
    button.classList.add('pressed');
    setTimeout(() => {
      button.classList.remove('pressed');
    }, duration);
  }
}

// Export for use in camera.js
window.ButtonManager = ButtonManager;
