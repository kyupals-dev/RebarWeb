// ==================== UPDATED RESULT PAGE JAVASCRIPT - PORTRAIT 3x5 GRID WITH ANALYSIS DATA ==================== 

// Global state management
const state = {
  allImages: [],
  filteredImages: [],
  currentPage: 1,
  itemsPerPage: 15, // 3x5 grid = 15 images per page
  totalPages: 1,
  currentFilters: {
    timeframe: 'all',
    sort: 'newest'
  },
  currentModalImage: null
};

// ==================== INITIALIZATION ==================== 
document.addEventListener('DOMContentLoaded', function() {
  console.log('Gallery page loaded, initializing...');
  initializePage();
});

async function initializePage() {
  try {
    showLoadingState();
    await loadImages();
    setupEventListeners();
    applyFilters();
    console.log('Gallery page initialized successfully');
  } catch (error) {
    console.error('Error initializing page:', error);
    showErrorState('Failed to load analysis results');
  }
}

// ==================== EVENT LISTENERS ==================== 
function setupEventListeners() {
  // Filter change listeners
  const timeframeFilter = document.getElementById('timeframe-filter');
  const sortFilter = document.getElementById('sort-filter');
  
  if (timeframeFilter) {
    timeframeFilter.addEventListener('change', handleFilterChange);
  }
  
  if (sortFilter) {
    sortFilter.addEventListener('change', handleFilterChange);
  }
  
  // Keyboard navigation
  document.addEventListener('keydown', handleKeyboard);
  
  // Window resize for responsive behavior
  window.addEventListener('resize', debounce(handleResize, 250));
}

function handleFilterChange() {
  const timeframeFilter = document.getElementById('timeframe-filter');
  const sortFilter = document.getElementById('sort-filter');
  
  state.currentFilters.timeframe = timeframeFilter?.value || 'all';
  state.currentFilters.sort = sortFilter?.value || 'newest';
  state.currentPage = 1; // Reset to first page
  
  applyFilters();
}

function handleKeyboard(e) {
  // Close modal with Escape key
  if (e.key === 'Escape' && isModalOpen()) {
    closeModal();
  }
  
  // Navigate pagination with arrow keys
  if (e.key === 'ArrowLeft' && state.currentPage > 1) {
    goToPage(state.currentPage - 1);
  }
  
  if (e.key === 'ArrowRight' && state.currentPage < state.totalPages) {
    goToPage(state.currentPage + 1);
  }
}

function handleResize() {
  // Keep 3x5 grid (15 items per page) for portrait optimization
  state.itemsPerPage = 15;
  applyFilters(); // Recalculate pagination
}

// ==================== IMAGE LOADING ==================== 
async function loadImages() {
  try {
    console.log('Loading analysis results from server...');
    const response = await fetch('/get-images');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Server response:', result);
    
    if (!result.success) {
      throw new Error(result.error || 'Failed to load analysis results');
    }
    
    // UPDATED: Filter to only show analyzed images (with analysis metadata)
    state.allImages = (result.images || []).filter(image => {
      return image.filename && image.filename.startsWith('analysis_') && image.has_metadata;
    });
    
    console.log(`Loaded ${state.allImages.length} analysis results`);
    
  } catch (error) {
    console.error('Error loading analysis results:', error);
    throw error;
  }
}

// ==================== FILTERING AND SORTING ==================== 
function applyFilters() {
  console.log('Applying filters:', state.currentFilters);
  
  // Start with all analyzed images
  let filtered = [...state.allImages];
  
  // Apply timeframe filter
  filtered = filterByTimeframe(filtered, state.currentFilters.timeframe);
  
  // Apply sorting
  filtered = sortImages(filtered, state.currentFilters.sort);
  
  state.filteredImages = filtered;
  state.totalPages = Math.ceil(filtered.length / state.itemsPerPage);
  
  // Ensure current page is valid
  if (state.currentPage > state.totalPages) {
    state.currentPage = Math.max(1, state.totalPages);
  }
  
  renderGallery();
  renderPagination();
}

function filterByTimeframe(images, timeframe) {
  if (timeframe === 'all') return images;
  
  const now = new Date();
  const cutoff = new Date();
  
  switch (timeframe) {
    case 'today':
      cutoff.setHours(0, 0, 0, 0);
      break;
    case 'week':
      cutoff.setDate(now.getDate() - 7);
      break;
    case 'month':
      cutoff.setMonth(now.getMonth() - 1);
      break;
    case 'year':
      cutoff.setFullYear(now.getFullYear() - 1);
      break;
    default:
      return images;
  }
  
  return images.filter(image => {
    // Use analysis timestamp if available, otherwise fall back to file timestamp
    const timestampToUse = image.analysis?.analysis_timestamp || image.timestamp;
    const imageDate = new Date(timestampToUse);
    return imageDate >= cutoff;
  });
}

function sortImages(images, sortBy) {
  const sorted = [...images];
  
  switch (sortBy) {
    case 'newest':
      return sorted.sort((a, b) => {
        const timestampA = a.analysis?.analysis_timestamp || a.timestamp;
        const timestampB = b.analysis?.analysis_timestamp || b.timestamp;
        return new Date(timestampB) - new Date(timestampA);
      });
    case 'oldest':
      return sorted.sort((a, b) => {
        const timestampA = a.analysis?.analysis_timestamp || a.timestamp;
        const timestampB = b.analysis?.analysis_timestamp || b.timestamp;
        return new Date(timestampA) - new Date(timestampB);
      });
    default:
      return sorted;
  }
}

// ==================== GALLERY RENDERING ==================== 
function renderGallery() {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (!galleryGrid) {
    console.error('Gallery grid element not found');
    return;
  }
  
  // Calculate pagination
  const startIndex = (state.currentPage - 1) * state.itemsPerPage;
  const endIndex = startIndex + state.itemsPerPage;
  const pageImages = state.filteredImages.slice(startIndex, endIndex);
  
  if (pageImages.length === 0) {
    showEmptyState();
    return;
  }
  
  // Create gallery HTML
  const galleryHTML = pageImages.map((image, index) => createImageCard(image, startIndex + index)).join('');
  
  galleryGrid.innerHTML = galleryHTML;
  
  // Add fade-in animation
  galleryGrid.classList.add('fade-in');
  setTimeout(() => galleryGrid.classList.remove('fade-in'), 500);
}

function createImageCard(image, index) {
  // Use analysis timestamp if available
  const timestamp = image.analysis?.analysis_timestamp || image.timestamp;
  const analysisDate = new Date(timestamp).toLocaleDateString();
  
  // Get detection count for display
  const detectionCount = image.analysis?.num_detections || 0;
  const modelType = image.analysis?.model_type || 'unknown';
  
  return `
    <div class="image-card" data-index="${index}">
      <div class="image-container">
        <img src="${image.url}" alt="Analysis Result" loading="lazy" onerror="handleImageError(this)">
        <div class="image-overlay">
          <div class="image-actions">
            <button class="view-btn" onclick="openModal('${image.filename}', '${image.url}', '${analysisDate}', ${index})">
              View Analysis
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
}

function handleImageError(img) {
  console.warn('Failed to load image:', img.src);
  img.style.display = 'none';
  const card = img.closest('.image-card');
  if (card) {
    card.innerHTML = `
      <div class="image-container">
        <div class="error-placeholder">
          <span>⚠️</span>
          <p>Analysis result not found</p>
        </div>
      </div>
    `;
  }
}

// ==================== PAGINATION ==================== 
function renderPagination() {
  const paginationContainer = document.getElementById('pagination');
  
  if (!paginationContainer) {
    console.error('Pagination container not found');
    return;
  }
  
  if (state.totalPages <= 1) {
    paginationContainer.innerHTML = '';
    return;
  }
  
  let paginationHTML = '';
  
  // Previous button
  paginationHTML += `
    <button class="pagination-btn" onclick="goToPage(${state.currentPage - 1})" 
            ${state.currentPage === 1 ? 'disabled' : ''}>
      ‹
    </button>
  `;
  
  // Page numbers (show max 5 pages)
  const maxVisible = 5;
  let startPage = Math.max(1, state.currentPage - Math.floor(maxVisible / 2));
  let endPage = Math.min(state.totalPages, startPage + maxVisible - 1);
  
  // Adjust start if we're near the end
  if (endPage - startPage + 1 < maxVisible) {
    startPage = Math.max(1, endPage - maxVisible + 1);
  }
  
  // First page and ellipsis
  if (startPage > 1) {
    paginationHTML += `<button class="pagination-btn" onclick="goToPage(1)">1</button>`;
    if (startPage > 2) {
      paginationHTML += `<span class="pagination-ellipsis">...</span>`;
    }
  }
  
  // Visible page numbers
  for (let i = startPage; i <= endPage; i++) {
    paginationHTML += `
      <button class="pagination-btn ${i === state.currentPage ? 'active' : ''}" 
              onclick="goToPage(${i})">
        ${i}
      </button>
    `;
  }
  
  // Last page and ellipsis
  if (endPage < state.totalPages) {
    if (endPage < state.totalPages - 1) {
      paginationHTML += `<span class="pagination-ellipsis">...</span>`;
    }
    paginationHTML += `<button class="pagination-btn" onclick="goToPage(${state.totalPages})">${state.totalPages}</button>`;
  }
  
  // Next button
  paginationHTML += `
    <button class="pagination-btn" onclick="goToPage(${state.currentPage + 1})" 
            ${state.currentPage === state.totalPages ? 'disabled' : ''}>
      ›
    </button>
  `;
  
  paginationContainer.innerHTML = paginationHTML;
}

function goToPage(page) {
  if (page < 1 || page > state.totalPages || page === state.currentPage) {
    return;
  }
  
  state.currentPage = page;
  renderGallery();
  renderPagination();
  
  // Scroll to top of gallery
  const gallerySection = document.querySelector('.gallery-section');
  if (gallerySection) {
    gallerySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ==================== UPDATED MODAL FUNCTIONALITY - ENHANCED WITH ANALYSIS DATA ==================== 
function openModal(filename, url, analysisDate, imageIndex) {
  const modal = document.getElementById('image-modal');
  const modalImage = document.getElementById('modal-image');
  const modalDimensions = document.getElementById('modal-dimensions');
  const modalMixture = document.getElementById('modal-mixture');
  const modalAnalysisDate = document.getElementById('modal-analysis-date');
  const modalDetections = document.getElementById('modal-detections');
  const modalModelType = document.getElementById('modal-model-type');
  
  if (!modal || !modalImage) {
    console.error('Modal elements not found');
    return;
  }
  
  // Get image data from current filtered list
  const imageData = state.filteredImages[imageIndex];
  if (!imageData) {
    console.error('Image data not found for index:', imageIndex);
    return;
  }
  
  // Store current image data
  state.currentModalImage = {
    filename,
    url,
    analysisDate,
    imageData
  };
  
  // Update modal image
  modalImage.src = url;
  modalImage.alt = `Analysis Result - ${filename}`;
  
  // UPDATED: Set analysis data from JSON metadata
  if (imageData.analysis) {
    const analysis = imageData.analysis;
    
    // Set dimensions
    if (modalDimensions) {
      const dimensions = analysis.dimensions?.display || 
                        `${analysis.dimensions?.length || 0}cm × ${analysis.dimensions?.width || 0}cm × ${analysis.dimensions?.height || 0}cm`;
      modalDimensions.textContent = dimensions;
    }
    
    // Set cement mixture
    if (modalMixture) {
      modalMixture.textContent = analysis.cement_mixture?.ratio || 'Data not available';
    }
    
    // Set analysis date
    if (modalAnalysisDate) {
      const timestamp = analysis.analysis_timestamp;
      if (timestamp) {
        const date = new Date(timestamp);
        modalAnalysisDate.textContent = date.toLocaleString();
      } else {
        modalAnalysisDate.textContent = analysisDate;
      }
    }
    
    // Set detections count
    if (modalDetections) {
      const count = analysis.num_detections || 0;
      const detectionsText = count === 1 ? '1 structure detected' : `${count} structures detected`;
      modalDetections.textContent = detectionsText;
    }
    
    // Set model type
    if (modalModelType) {
      const modelType = analysis.model_type || 'Unknown';
      modalModelType.textContent = modelType === 'real_trained_model' ? 'AI Model' : 
                                  modelType === 'placeholder' ? 'Simulation' : modelType;
    }
  } else {
    // Fallback if no analysis data
    if (modalDimensions) modalDimensions.textContent = 'Analysis data not available';
    if (modalMixture) modalMixture.textContent = 'Analysis data not available';
    if (modalAnalysisDate) modalAnalysisDate.textContent = analysisDate;
    if (modalDetections) modalDetections.textContent = 'Unknown';
    if (modalModelType) modalModelType.textContent = 'Unknown';
  }
  
  // Show modal
  modal.classList.add('active');
  document.body.style.overflow = 'hidden'; // Prevent background scrolling
  
  console.log('Enhanced modal opened for:', filename);
  if (imageData.analysis) {
    console.log('Analysis data:', imageData.analysis);
  }
}

function closeModal() {
  const modal = document.getElementById('image-modal');
  
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
    state.currentModalImage = null;
    console.log('Modal closed');
  }
}

function isModalOpen() {
  const modal = document.getElementById('image-modal');
  return modal && modal.classList.contains('active');
}

// ==================== MODAL ACTIONS ==================== 
function downloadCurrentImage() {
  if (!state.currentModalImage) {
    console.error('No current modal image to download');
    showNotification('No analysis result selected for download', 'error');
    return;
  }
  
  const { filename, url } = state.currentModalImage;
  
  console.log('Downloading analysis result:', filename, 'from:', url);
  
  try {
    // Method 1: Try using fetch to get the blob first
    fetch(url)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
      })
      .then(blob => {
        // Create blob URL
        const blobUrl = window.URL.createObjectURL(blob);
        
        // Create download link
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = filename;
        link.style.display = 'none';
        
        // Add to DOM, click, and remove
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up blob URL
        setTimeout(() => {
          window.URL.revokeObjectURL(blobUrl);
        }, 100);
        
        showNotification('Analysis result download started successfully', 'success');
        console.log('Download initiated successfully for:', filename);
      })
      .catch(error => {
        console.error('Fetch download failed, trying direct method:', error);
        
        // Method 2: Fallback to direct download
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        
        // Force download by setting proper headers simulation
        link.style.display = 'none';
        document.body.appendChild(link);
        
        // Trigger click
        const event = new MouseEvent('click', {
          bubbles: true,
          cancelable: true,
          view: window
        });
        
        link.dispatchEvent(event);
        document.body.removeChild(link);
        
        showNotification('Analysis result download initiated (fallback method)', 'success');
        console.log('Fallback download initiated for:', filename);
      });
      
  } catch (error) {
    console.error('Download error:', error);
    showNotification('Failed to download analysis result: ' + error.message, 'error');
  }
}

async function deleteCurrentImage() {
  if (!state.currentModalImage) {
    console.error('No current modal image to delete');
    return;
  }
  
  const { filename } = state.currentModalImage;
  
  if (!confirm(`Are you sure you want to delete this analysis result "${filename}"? This action cannot be undone.`)) {
    return;
  }
  
  try {
    console.log('Deleting analysis result:', filename);
    
    const response = await fetch(`/delete-image/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Analysis result deleted successfully:', filename);
      
      // Close modal
      closeModal();
      
      // Reload images and refresh gallery
      await loadImages();
      applyFilters();
      
      showNotification('Analysis result deleted successfully', 'success');
    } else {
      throw new Error(result.error || 'Failed to delete analysis result');
    }
    
  } catch (error) {
    console.error('Error deleting analysis result:', error);
    showNotification('Failed to delete analysis result: ' + error.message, 'error');
  }
}

// ==================== GLOBAL ACTIONS ==================== 
function clearFilters() {
  console.log('Clearing all filters');
  
  const timeframeFilter = document.getElementById('timeframe-filter');
  const sortFilter = document.getElementById('sort-filter');
  
  if (timeframeFilter) timeframeFilter.value = 'all';
  if (sortFilter) sortFilter.value = 'newest';
  
  state.currentFilters = {
    timeframe: 'all',
    sort: 'newest'
  };
  state.currentPage = 1;
  
  applyFilters();
}

async function clearAllImages() {
  if (!confirm('Are you sure you want to delete ALL analysis results? This action cannot be undone!')) {
    return;
  }
  
  try {
    console.log('Clearing all analysis results...');
    
    const response = await fetch('/clear-all-images', {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('All analysis results cleared successfully');
      
      state.allImages = [];
      state.filteredImages = [];
      state.currentPage = 1;
      state.totalPages = 1;
      
      showEmptyState();
      renderPagination();
      
      showNotification('All analysis results cleared successfully', 'success');
    } else {
      throw new Error(result.error || 'Failed to clear analysis results');
    }
    
  } catch (error) {
    console.error('Error clearing analysis results:', error);
    showNotification('Failed to clear analysis results: ' + error.message, 'error');
  }
}

function goToMainPage() {
  console.log('Navigating back to main page...');
  window.location.href = '/mainpage.html';
}

// ==================== UI STATES ==================== 
function showLoadingState() {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (galleryGrid) {
    galleryGrid.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading analysis results...</p>
      </div>
    `;
  }
}

function showEmptyState() {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (galleryGrid) {
    const hasFilters = state.currentFilters.timeframe !== 'all' || state.currentFilters.sort !== 'newest';
    
    galleryGrid.innerHTML = `
      <div class="empty-state">
        <h3>${hasFilters ? 'No results match your filters' : 'No analysis results yet'}</h3>
        <p>${hasFilters ? 'Try adjusting your filters or clear them to see all results.' : 'Go back and analyze some rebar structures!'}</p>
        ${hasFilters ? '<button class="clear-filters-btn" onclick="clearFilters()">Clear Filters</button>' : ''}
      </div>
    `;
  }
}

function showErrorState(message) {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (galleryGrid) {
    galleryGrid.innerHTML = `
      <div class="empty-state">
        <h3>Error loading analysis results</h3>
        <p>${message}</p>
        <button class="clear-filters-btn" onclick="initializePage()">Try Again</button>
      </div>
    `;
  }
}

// ==================== NOTIFICATIONS ==================== 
function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <span>${message}</span>
    <button onclick="this.parentElement.remove()">&times;</button>
  `;
  
  // Add styles if not already present
  if (!document.querySelector('.notification-styles')) {
    const styles = document.createElement('style');
    styles.className = 'notification-styles';
    styles.textContent = `
      .notification {
        position: fixed;
        top: 100px;
        right: 20px;
        background: white;
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 15px;
        max-width: 400px;
        animation: slideInRight 0.3s ease-out;
      }
      .notification-success { border-left: 4px solid #28a745; }
      .notification-error { border-left: 4px solid #dc3545; }
      .notification-info { border-left: 4px solid #007bff; }
      .notification button {
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
        opacity: 0.7;
      }
      .notification button:hover { opacity: 1; }
      @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
    `;
    document.head.appendChild(styles);
  }
  
  // Add to page
  document.body.appendChild(notification);
  
  // Auto remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 5000);
}

// ==================== UTILITY FUNCTIONS ==================== 
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// ==================== LEGACY SUPPORT ==================== 
// For backward compatibility with existing code
window.loadGallery = initializePage;
window.goBackToCamera = goToMainPage;
window.downloadImage = function(url, filename) {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.target = '_blank';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Export functions for global access
window.openModal = openModal;
window.closeModal = closeModal;
window.downloadCurrentImage = downloadCurrentImage;
window.deleteCurrentImage = deleteCurrentImage;
window.clearFilters = clearFilters;
window.clearAllImages = clearAllImages;
window.goToMainPage = goToMainPage;
window.goToPage = goToPage;
window.handleImageError = handleImageError;
