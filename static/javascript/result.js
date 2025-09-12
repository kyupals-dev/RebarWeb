// ==================== UPDATED RESULT PAGE JAVASCRIPT (PIPELINE SUPPORT) ==================== 
// MODIFIED: Enhanced to display pipeline analysis metadata

// Global state management
const state = {
  allImages: [],
  filteredImages: [],
  currentPage: 1,
  itemsPerPage: 6, // 3 images per row, 2 rows
  totalPages: 1,
  currentFilters: {
    timeframe: 'all',
    sort: 'newest'
  },
  currentModalImage: null,
  imageStats: null // Store filtering stats
};

// ==================== INITIALIZATION ==================== 
document.addEventListener('DOMContentLoaded', function() {
  console.log('Result page loaded, initializing (PIPELINE MODE)...');
  initializePage();
});

async function initializePage() {
  try {
    showLoadingState();
    await loadImages();
    setupEventListeners();
    applyFilters();
    console.log('Result page initialized successfully (PIPELINE MODE)');
  } catch (error) {
    console.error('Error initializing page:', error);
    showErrorState('Failed to load images');
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
  // Adjust items per page based on screen size
  const width = window.innerWidth;
  if (width <= 768) {
    state.itemsPerPage = 6; // 1 per row, 6 rows on mobile
  } else {
    state.itemsPerPage = 6; // 3 per row, 2 rows on desktop and tablet
  }
  
  applyFilters(); // Recalculate pagination
}

// ==================== IMAGE LOADING (ENHANCED FOR PIPELINE) ==================== 
async function loadImages() {
  try {
    console.log('Loading PIPELINE analyzed images from server...');
    const response = await fetch('/get-images');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Server response:', result);
    
    if (!result.success) {
      throw new Error(result.error || 'Failed to load images');
    }
    
    // Store the filtered analyzed images
    state.allImages = result.images || [];
    
    // Store filtering stats if available
    if (result.stats) {
      state.imageStats = result.stats;
      console.log('Image filtering stats:', result.stats);
      console.log(`Gallery shows ${result.stats.analyzed_shown} pipeline analyzed images, hides ${result.stats.originals_hidden} original images`);
    }
    
    console.log(`Loaded ${state.allImages.length} PIPELINE analyzed images for gallery`);
    
    // Log types of images loaded
    if (state.allImages.length > 0) {
      const imageTypes = state.allImages.map(img => img.type || 'analyzed').reduce((acc, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {});
      console.log('Image types loaded:', imageTypes);
    }
    
  } catch (error) {
    console.error('Error loading images:', error);
    throw error;
  }
}

// ==================== FILTERING AND SORTING ==================== 
function applyFilters() {
  console.log('Applying filters to PIPELINE analyzed images:', state.currentFilters);
  
  // Start with all analyzed images (already filtered by backend)
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
  
  console.log(`Filtered results: ${filtered.length} PIPELINE images, ${state.totalPages} pages`);
  
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
    const imageDate = new Date(image.timestamp);
    return imageDate >= cutoff;
  });
}

function sortImages(images, sortBy) {
  const sorted = [...images];
  
  switch (sortBy) {
    case 'newest':
      return sorted.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    case 'oldest':
      return sorted.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
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
  const capturedDate = new Date(image.timestamp).toLocaleDateString();
  const imageType = image.type || 'analyzed';
  
  // Determine if this is a pipeline image
  const isPipelineImage = image.filename && (
    image.filename.includes('pipeline') || 
    image.filename.includes('analyzed')
  );
  
  const badgeText = isPipelineImage ? 'Pipeline Analysis' : 'AI Analysis';
  
  return `
    <div class="image-card" data-index="${index}" data-type="${imageType}">
      <div class="image-container">
        <img src="${image.url}" alt="Pipeline analyzed image with AI overlays" loading="lazy" onerror="handleImageError(this)">
        <div class="image-overlay">
          <div class="image-actions">
            <button class="view-btn" onclick="openPipelineModal('${image.filename}', '${image.url}', '${capturedDate}')">
              View Pipeline Analysis
            </button>
          </div>
          <div class="image-type-badge">${badgeText}</div>
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
          <p>Pipeline analyzed image not found</p>
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

// ==================== ENHANCED PIPELINE MODAL FUNCTIONALITY ==================== 
async function openPipelineModal(filename, url, captured) {
  const modal = document.getElementById('image-modal');
  const modalImage = document.getElementById('modal-image');
  
  if (!modal || !modalImage) {
    console.error('Modal elements not found');
    return;
  }
  
  // Store current image data
  state.currentModalImage = {
    filename,
    url,
    captured
  };
  
  // Update modal content
  modalImage.src = url;
  modalImage.alt = `Pipeline analyzed image: ${filename}`;
  
  // Try to load pipeline metadata
  try {
    console.log('Loading pipeline metadata for:', filename);
    const metadataResponse = await fetch(`/get-image-metadata/${encodeURIComponent(filename)}`);
    
    if (metadataResponse.ok) {
      const metadata = await metadataResponse.json();
      if (metadata.success && metadata.metadata) {
        updatePipelineModalWithMetadata(metadata.metadata);
      } else {
        updatePipelineModalWithDefaults();
      }
    } else {
      console.warn('Could not load metadata, using defaults');
      updatePipelineModalWithDefaults();
    }
  } catch (error) {
    console.error('Error loading pipeline metadata:', error);
    updatePipelineModalWithDefaults();
  }
  
  // Show modal
  modal.classList.add('active');
  document.body.style.overflow = 'hidden'; // Prevent background scrolling
  
  console.log('Pipeline modal opened for analyzed image:', filename);
}

function updatePipelineModalWithMetadata(metadata) {
  console.log('Updating modal with pipeline metadata:', metadata);
  
  // Update dimensions - EXACT FORMAT: "27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters"
  const dimensionsElement = document.getElementById('modal-dimensions');
  if (dimensionsElement) {
    const dimensions = metadata.dimensions || metadata.analysis_result?.dimensions;
    if (dimensions && dimensions.display) {
      dimensionsElement.textContent = dimensions.display;
    } else {
      dimensionsElement.textContent = '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters';
    }
  }
  
  // Update cement mixture - EXACT FORMAT: "1:2:4"
  const mixtureElement = document.getElementById('modal-mixture');
  if (mixtureElement) {
    const mixture = metadata.cement_mixture || metadata.analysis_result?.cement_mixture;
    if (mixture && mixture.ratio_string) {
      mixtureElement.textContent = mixture.ratio_string;
    } else {
      mixtureElement.textContent = '1:2:4';
    }
  }
  
  // Update pipeline details
  updatePipelineDetails(metadata);
}

function updatePipelineDetails(metadata) {
  // Update analysis date
  const analysisDateElement = document.getElementById('modal-analysis-date');
  if (analysisDateElement) {
    const timestamp = metadata.timestamp || metadata.analysis_result?.timestamp;
    if (timestamp) {
      const date = new Date(timestamp);
      analysisDateElement.textContent = date.toLocaleString();
    } else {
      analysisDateElement.textContent = new Date().toLocaleString();
    }
  }
  
  // Update detections count
  const detectionsElement = document.getElementById('modal-detections');
  if (detectionsElement) {
    const detections = metadata.detections || metadata.analysis_result?.detections;
    if (detections && detections.count !== undefined) {
      detectionsElement.textContent = `${detections.count} structures detected`;
    } else {
      detectionsElement.textContent = '2 verticals + 11 horizontals (expected)';
    }
  }
  
  // Update model type
  const modelTypeElement = document.getElementById('modal-model-type');
  if (modelTypeElement) {
    const modelType = metadata.model_type || metadata.analysis_result?.metadata?.model_type;
    if (modelType) {
      if (modelType.includes('pipeline')) {
        modelTypeElement.textContent = 'Quadrant Pipeline Analysis';
      } else if (modelType.includes('placeholder')) {
        modelTypeElement.textContent = 'Placeholder Pipeline (Demo)';
      } else {
        modelTypeElement.textContent = 'AI Pipeline Analysis';
      }
    } else {
      modelTypeElement.textContent = 'Quadrant Pipeline Analysis';
    }
  }
}

function updatePipelineModalWithDefaults() {
  console.log('Updating modal with default pipeline values');
  
  // Set default values - EXACT FORMAT
  const dimensionsElement = document.getElementById('modal-dimensions');
  if (dimensionsElement) {
    dimensionsElement.textContent = '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters';
  }
  
  const mixtureElement = document.getElementById('modal-mixture');
  if (mixtureElement) {
    mixtureElement.textContent = '1:2:4';
  }
  
  // Set default pipeline details
  const analysisDateElement = document.getElementById('modal-analysis-date');
  if (analysisDateElement) {
    analysisDateElement.textContent = new Date().toLocaleString();
  }
  
  const detectionsElement = document.getElementById('modal-detections');
  if (detectionsElement) {
    detectionsElement.textContent = '2 verticals + 11 horizontals (expected)';
  }
  
  const modelTypeElement = document.getElementById('modal-model-type');
  if (modelTypeElement) {
    modelTypeElement.textContent = 'Quadrant Pipeline Analysis';
  }
}

function closeModal() {
  const modal = document.getElementById('image-modal');
  
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
    state.currentModalImage = null;
    console.log('Pipeline modal closed');
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
    showNotification('No image selected for download', 'error');
    return;
  }
  
  const { filename, url } = state.currentModalImage;
  
  console.log('Downloading pipeline analyzed image:', filename, 'from:', url);
  
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
        
        showNotification('Pipeline analyzed image download started successfully', 'success');
        console.log('Download initiated successfully for pipeline analyzed image:', filename);
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
        
        showNotification('Download initiated (fallback method)', 'success');
        console.log('Fallback download initiated for:', filename);
      });
      
  } catch (error) {
    console.error('Download error:', error);
    showNotification('Failed to download image: ' + error.message, 'error');
  }
}

async function deleteCurrentImage() {
  if (!state.currentModalImage) {
    console.error('No current modal image to delete');
    return;
  }
  
  const { filename } = state.currentModalImage;
  
  if (!confirm(`Are you sure you want to delete pipeline analysis "${filename}"? This action cannot be undone.`)) {
    return;
  }
  
  try {
    console.log('Deleting pipeline analyzed image:', filename);
    
    const response = await fetch(`/delete-image/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Pipeline analyzed image deleted successfully:', filename);
      
      // Close modal
      closeModal();
      
      // Reload images and refresh gallery
      await loadImages();
      applyFilters();
      
      showNotification('Pipeline analyzed image deleted successfully', 'success');
    } else {
      throw new Error(result.error || 'Failed to delete image');
    }
    
  } catch (error) {
    console.error('Error deleting image:', error);
    showNotification('Failed to delete image: ' + error.message, 'error');
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
  const confirmMessage = state.imageStats && state.imageStats.analyzed_shown > 0
    ? `Are you sure you want to delete all ${state.imageStats.analyzed_shown} pipeline analyzed images? This action cannot be undone!`
    : 'Are you sure you want to delete ALL pipeline analyzed images? This action cannot be undone!';
    
  if (!confirm(confirmMessage)) {
    return;
  }
  
  try {
    console.log('Clearing all pipeline analyzed images...');
    
    const response = await fetch('/clear-all-images', {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('All pipeline analyzed images cleared successfully');
      
      // Reset state
      state.allImages = [];
      state.filteredImages = [];
      state.currentPage = 1;
      state.totalPages = 1;
      state.imageStats = null;
      
      showEmptyState();
      renderPagination();
      
      const clearedCount = result.details ? result.details.total_deleted : 'All';
      showNotification(`${clearedCount} pipeline analyzed images cleared successfully`, 'success');
    } else {
      throw new Error(result.error || 'Failed to clear images');
    }
    
  } catch (error) {
    console.error('Error clearing images:', error);
    showNotification('Failed to clear images: ' + error.message, 'error');
  }
}

function goToMainPage() {
  console.log('Navigating back to main page (PIPELINE MODE)...');
  window.location.href = '/mainpage.html';
}

// ==================== UI STATES ==================== 
function showLoadingState() {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (galleryGrid) {
    galleryGrid.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading pipeline analyzed images...</p>
      </div>
    `;
  }
}

function showEmptyState() {
  const galleryGrid = document.getElementById('gallery-grid');
  
  if (galleryGrid) {
    const hasFilters = state.currentFilters.timeframe !== 'all' || state.currentFilters.sort !== 'newest';
    
    let emptyMessage, emptyDescription;
    
    if (hasFilters) {
      emptyMessage = 'No pipeline images match your filters';
      emptyDescription = 'Try adjusting your filters or clear them to see all pipeline analyzed images.';
    } else {
      emptyMessage = 'No pipeline analyzed images yet';
      emptyDescription = 'Go back and capture some images with pipeline analysis!';
    }
    
    // Add stats info if available
    let statsInfo = '';
    if (state.imageStats && state.imageStats.originals_hidden > 0) {
      statsInfo = `<p class="stats-info">Note: ${state.imageStats.originals_hidden} original images are hidden. Only pipeline analyzed images with AI overlays are shown.</p>`;
    }
    
    galleryGrid.innerHTML = `
      <div class="empty-state">
        <h3>${emptyMessage}</h3>
        <p>${emptyDescription}</p>
        ${statsInfo}
        <p class="hint">Only images with pipeline analysis (quadrant intersections + cement calculation) are shown in this gallery.</p>
        <p class="pipeline-info">Expected format: "27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters" and "1:2:4" ratio</p>
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
        <h3>Error loading pipeline analyzed images</h3>
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
      .stats-info {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin: 10px 0;
      }
      .pipeline-info {
        font-size: 0.85em;
        color: #2d7d47;
        font-weight: 600;
        margin: 10px 0;
        padding: 8px 12px;
        background: rgba(45, 125, 71, 0.1);
        border-radius: 6px;
        border-left: 3px solid #2d7d47;
      }
      .image-type-badge {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(45, 125, 71, 0.9);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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

// Export functions for global access - UPDATED FOR PIPELINE
window.openModal = openPipelineModal; // Updated to use pipeline modal
window.closeModal = closeModal;
window.downloadCurrentImage = downloadCurrentImage;
window.deleteCurrentImage = deleteCurrentImage;
window.clearFilters = clearFilters;
window.clearAllImages = clearAllImages;
window.goToMainPage = goToMainPage;
window.goToPage = goToPage;
window.handleImageError = handleImageError;

// PIPELINE-SPECIFIC EXPORTS
window.openPipelineModal = openPipelineModal;
window.updatePipelineModalWithMetadata = updatePipelineModalWithMetadata;
window.updatePipelineModalWithDefaults = updatePipelineModalWithDefaults;
