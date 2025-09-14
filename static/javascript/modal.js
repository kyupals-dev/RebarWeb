// ==================== MODAL MANAGEMENT MODULE ====================
// Handles all modal interactions for the Rebar Vista camera interface

class ModalManager {
  constructor(cameraApp) {
    this.cameraApp = cameraApp;
    this.setupModalListeners();
  }

  setupModalListeners() {
    console.log('📋 Setting up modal event listeners...');

    // Tutorial Modal
    if (this.cameraApp.tutorialModal) {
      this.cameraApp.tutorialModal.addEventListener('click', (e) => {
        if (e.target === this.cameraApp.tutorialModal) {
          this.closeTutorialModal();
        }
      });
    }

    // Results Modal
    if (this.cameraApp.resultsModal) {
      this.cameraApp.resultsModal.addEventListener('click', (e) => {
        if (e.target === this.cameraApp.resultsModal) {
          this.closeResultsModal();
        }
      });
    }

    // Error Modal
    if (this.cameraApp.errorModal) {
      this.cameraApp.errorModal.addEventListener('click', (e) => {
        if (e.target === this.cameraApp.errorModal) {
          this.closeErrorModal();
        }
      });
    }

    console.log('✅ Modal listeners setup complete');
  }

  // ==================== TUTORIAL MODAL ====================
  
  openTutorialModal() {
    console.log('❓ Opening tutorial modal...');
    if (this.cameraApp.tutorialModal) {
      this.cameraApp.tutorialModal.classList.add('active');
    }
  }

  closeTutorialModal() {
    console.log('✕ Closing tutorial modal...');
    if (this.cameraApp.tutorialModal) {
      this.cameraApp.tutorialModal.classList.remove('active');
    }
  }

  // ==================== RESULTS MODAL ====================
  
  showResultsModal(analysisResults) {
    console.log('📊 Showing results modal with pipeline data...');
    
    if (!this.cameraApp.resultsModal || !analysisResults) {
      console.error('❌ Cannot show results modal - missing modal or data');
      return;
    }

    // Store results
    this.cameraApp.analysisResults = analysisResults;

    // Update modal content with pipeline results
    this.updateResultsModalContent(analysisResults);

    // Show modal
    this.cameraApp.resultsModal.classList.add('active');

    // Update status
    this.cameraApp.updateStatus('Analysis complete - viewing results');
  }

  updateResultsModalContent(results) {
    try {
      // Update basic detection info
      this.updateElement('pipeline-detections', `${results.num_detections || 0} detections found`);
      this.updateElement('pipeline-method', 'Quadrant Intersection Pipeline');

      // Update dimensions (exact formatting as requested)
      if (results.dimensions) {
        const dimensions = results.dimensions;
        const dimensionText = `${dimensions.length}cm x ${dimensions.width}cm x ${dimensions.height}cm = ${Math.round(dimensions.volume)} cubic centimeters`;
        this.updateElement('dimension-result', dimensionText);
        
        // Update modal content
        this.updateElement('pipeline-quadrants', 
          `BL, BR, TL, TR corners analyzed (${results.quadrant_info?.intersections_found || 0} intersections)`);
        this.updateElement('pipeline-formula', `PX_TO_CM × 1/3.54 + 4.5cm offset`);
      }

      // Update cement mixture (exact formatting as requested)
      if (results.cement_mixture) {
        const mixture = results.cement_mixture;
        this.updateElement('mixture-result', '1:2:4');
        
        // Update detailed ratio text
        if (mixture.ratio_string) {
          this.updateElement('cement-ratio-detail', mixture.ratio_string);
        }
      }

      // Update step images if available
      this.updateStepImages(results);

      console.log('✅ Results modal content updated');
    } catch (error) {
      console.error('❌ Error updating results modal content:', error);
    }
  }

  updateStepImages(results) {
    // Update the 4 pipeline step images as requested
    const stepImages = {
      'step1-detection': results.step_images?.detection || results.analyzed_image_path,
      'step2-quadrants': results.step_images?.quadrants || results.analyzed_image_path,
      'step3-polygon': results.step_images?.polygon || results.analyzed_image_path,
      'step4-cement': results.step_images?.cement || results.analyzed_image_path
    };

    Object.entries(stepImages).forEach(([elementId, imagePath]) => {
      const element = document.getElementById(elementId);
      if (element && imagePath) {
        // Convert to URL if it's a file path
        const imageUrl = imagePath.startsWith('/static/') ? imagePath : `/static/captured_images/${imagePath.split('/').pop()}`;
        element.src = imageUrl;
        element.alt = `Pipeline step ${elementId}`;
      }
    });
  }

  updateElement(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = content;
    } else {
      console.warn(`⚠️ Element not found: ${elementId}`);
    }
  }

  closeResultsModal() {
    console.log('✕ Closing results modal...');
    if (this.cameraApp.resultsModal) {
      this.cameraApp.resultsModal.classList.remove('active');
    }
    this.cameraApp.analysisResults = null;
    this.cameraApp.updateStatus('Ready for next capture (analyzed image only mode)');
  }

  // ==================== ERROR MODAL ====================
  
  showErrorModal(errorMessage = null) {
    console.log('⚠️ Showing error modal...');
    
    if (!this.cameraApp.errorModal) {
      console.error('❌ Error modal not found');
      return;
    }

    // Update error message if provided
    if (errorMessage) {
      const errorTextElement = this.cameraApp.errorModal.querySelector('p');
      if (errorTextElement) {
        errorTextElement.textContent = errorMessage;
      }
    }

    this.cameraApp.errorModal.classList.add('active');
  }

  closeErrorModal() {
    console.log('✕ Closing error modal...');
    if (this.cameraApp.errorModal) {
      this.cameraApp.errorModal.classList.remove('active');
    }
    this.cameraApp.updateStatus('Ready for next capture (analyzed image only mode)');
  }

  // ==================== UTILITY METHODS ====================
  
  closeAllModals() {
    console.log('✕ Closing all modals...');
    this.closeTutorialModal();
    this.closeResultsModal();
    this.closeErrorModal();
  }

  isAnyModalOpen() {
    const modals = [
      this.cameraApp.tutorialModal,
      this.cameraApp.resultsModal,
      this.cameraApp.errorModal
    ];

    return modals.some(modal => 
      modal && modal.classList.contains('active')
    );
  }

  // Save results and images to gallery
  async saveResultsToGallery(results) {
    if (!results || !results.success) {
      console.error('❌ Cannot save invalid results to gallery');
      return false;
    }

    try {
      console.log('💾 Saving analysis results to gallery...');

      // Prepare metadata for gallery storage
      const metadata = {
        timestamp: new Date().toISOString(),
        analysis_type: 'pipeline_quadrant',
        dimensions: results.dimensions,
        cement_mixture: results.cement_mixture,
        detections: results.num_detections,
        model_type: results.model_type,
        quadrant_info: results.quadrant_info
      };

      // Save to gallery via API
      const response = await fetch('/save-to-gallery', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          analyzed_image_path: results.analyzed_image_path,
          metadata: metadata,
          step_images: results.step_images || {}
        })
      });

      if (response.ok) {
        console.log('✅ Results saved to gallery successfully');
        return true;
      } else {
        const error = await response.text();
        console.error('❌ Failed to save to gallery:', error);
        return false;
      }
    } catch (error) {
      console.error('❌ Error saving to gallery:', error);
      return false;
    }
  }
}

// Global modal functions for backward compatibility
window.closeTutorialModal = function() {
  if (window.cameraApp && window.cameraApp.modalManager) {
    window.cameraApp.modalManager.closeTutorialModal();
  }
};

window.closeResultsModal = function() {
  if (window.cameraApp && window.cameraApp.modalManager) {
    window.cameraApp.modalManager.closeResultsModal();
  }
};

window.closeErrorModal = function() {
  if (window.cameraApp && window.cameraApp.modalManager) {
    window.cameraApp.modalManager.closeErrorModal();
  }
};

// Export for use in camera.js
window.ModalManager = ModalManager;
