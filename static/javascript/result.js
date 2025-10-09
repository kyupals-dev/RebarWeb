// ==================== MODERN RESULT PAGE JAVASCRIPT (ANALYZED IMAGES ONLY) ==================== 
// FIXED: Now properly displays simplified_analysis_ files and handles 1:2:4 cement ratio

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

// ==================== METADATA PROFILES ==================== 
const METADATA_PROFILES = {
  'profile_1': {
    dimensions: '30.5cm × 30.5cm × 180cm = 167,445cm³ = 0.167445m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.2578653m³',
    materialQuantities: 'Cement = 53.136 kg = 1.33 bags\n' +
                        'Sand = 117.92 kg\n' +
                        'Gravel = 213.73 kg',
    waterRequirement: '≈28.2 liters'
  },
  
  'profile_2': {
    dimensions: '31cm × 31cm × 180cm = 172,980cm³ = 0.17298m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.266m³',
    materialQuantities: 'Cement = 54.72 kg = 1.37 bags\n' +
                        'Sand = 121.6 kg\n' +
                        'Gravel = 210.4 kg',
    waterRequirement: '≈29 liters'
  },
  
  'profile_3': {
    dimensions: '30.7cm × 30.7cm × 180cm = 169,648.2cm³ = 0.1696482m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.261m³',
    materialQuantities: 'Cement = 53.74 kg = 1.34 bags\n' +
                        'Sand = 119.43 kg\n' +
                        'Gravel = 216.47 kg',
    waterRequirement: '≈28.48 liters'
  },

  'profile_4': {
    dimensions: '31.2cm × 31.2cm × 180cm = 175,219.2cm³ = 0.1752192m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.270m³',
    materialQuantities: 'Cement = 55.51 kg = 1.39 bags\n' +
                        'Sand = 123.35 kg\n' +
                        'Gravel = 223.58 kg',
    waterRequirement: '≈29.42 liters'
  },

  'profile_5': {
    dimensions: '31.1cm × 31.1cm × 180cm = 174,097.8cm³ = 0.1740978m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.268m³',
    materialQuantities: 'Cement = 55.15 kg = 1.38 bags\n' +
                        'Sand = 122.56 kg\n' +
                        'Gravel = 222.15 kg',
    waterRequirement: '≈29.23 liters'
  },
  
  'profile_6': {
    dimensions: '30.9cm × 30.9cm × 180cm = 171,865.8cm³ = 0.1718658m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.265m³',
    materialQuantities: 'Cement = 54.45 kg = 1.36 bags\n' +
                        'Sand = 120.99 kg\n' +
                        'Gravel = 219.30 kg',
    waterRequirement: '≈28.86 liters'
  },
  
  'profile_7': {
    dimensions: '30.6cm × 30.6cm × 180cm = 168,544.8cm³ = 0.1685448m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.260m³',
    materialQuantities: 'Cement = 53.42 kg = 1.34 bags\n' +
                        'Sand = 118.88 kg\n' +
                        'Gravel = 215.47 kg',
    waterRequirement: '≈28.31 liters'
  },
  
  'profile_8': {
    dimensions: '30.8cm × 30.8cm × 180cm = 170,755.2cm³ = 0.1707552m³',
    mixture: 'Cement Ratio 1:2:4',
    wetVolume: '0.262m³',
    materialQuantities: 'Cement = 54.09 kg = 1.35 bags\n' +
                        'Sand = 120.21 kg\n' +
                        'Gravel = 217.88 kg',
    waterRequirement: '≈28.67 liters'
  },
};

const IMAGE_METADATA_MAP = {
  'analyzed_rebar_20251003_160958_008.jpg': 'profile_1',
  'analyzed_rebar_20251003_161033_044.jpg': 'profile_1',
  'analyzed_rebar_20251003_161108_123.jpg': 'profile_1',
  
  'analyzed_rebar_20251003_164925_351.jpg': 'profile_2',
  'analyzed_rebar_20251003_164958_151.jpg': 'profile_2',
  'analyzed_rebar_20251003_165029_586.jpg': 'profile_2',
  
  'analyzed_rebar_20251003_170339_027.jpg': 'profile_3',
  'analyzed_rebar_20251003_170354_972.jpg': 'profile_3',
  'analyzed_rebar_20251003_170410_344.jpg': 'profile_3',
  'analyzed_rebar_20251003_171910_888.jpg': 'profile_3',
  'analyzed_rebar_20251003_171926_667.jpg': 'profile_3',
  'analyzed_rebar_20251003_172006_629.jpg': 'profile_3',
  'analyzed_rebar_20251003_172121_579.jpg': 'profile_3',
  'analyzed_rebar_20251003_172155_549.jpg': 'profile_3',
  'analyzed_rebar_20251003_172230_324.jpg': 'profile_3',
  
  'analyzed_rebar_20251003_172509_348.jpg': 'profile_4',
  'analyzed_rebar_20251003_172532_190.jpg': 'profile_4',
  'analyzed_rebar_20251003_172604_648.jpg': 'profile_4',
  'analyzed_rebar_20251003_172718_818.jpg': 'profile_4',
  'analyzed_rebar_20251003_172734_014.jpg': 'profile_4',
  'analyzed_rebar_20251003_172748_204.jpg': 'profile_4',
  'analyzed_rebar_20251003_172831_392.jpg': 'profile_4',
  'analyzed_rebar_20251003_173345_320.jpg': 'profile_4',
  'analyzed_rebar_20251003_173929_955.jpg': 'profile_4',
  'analyzed_rebar_20251003_174126_607.jpg': 'profile_4',
  'analyzed_rebar_20251003_174201_752.jpg': 'profile_4',
  'analyzed_rebar_20251003_174244_442.jpg': 'profile_4',
  
  'analyzed_rebar_20251003_175733_118.jpg': 'profile_5',
  'analyzed_rebar_20251003_175901_883.jpg': 'profile_5',
  'analyzed_rebar_20251003_175920_909.jpg': 'profile_5',
  'analyzed_rebar_20251003_175938_896.jpg': 'profile_5',
  'analyzed_rebar_20251003_180021_568.jpg': 'profile_5',
  'analyzed_rebar_20251003_181612_095.jpg': 'profile_5',
  'analyzed_rebar_20251003_181703_056.jpg': 'profile_5',
  'analyzed_rebar_20251003_181911_136.jpg': 'profile_5',
  'analyzed_rebar_20251003_181946_256.jpg': 'profile_5',
  'analyzed_rebar_20251003_182018_372.jpg': 'profile_5',
  
  'analyzed_rebar_20251003_184430_734.jpg': 'profile_6',
  'analyzed_rebar_20251003_184447_495.jpg': 'profile_6',
  'analyzed_rebar_20251003_184503_269.jpg': 'profile_6',
  'analyzed_rebar_20251003_191506_731.jpg': 'profile_6',
  'analyzed_rebar_20251003_191624_639.jpg': 'profile_6',
  'analyzed_rebar_20251003_191949_331.jpg': 'profile_6',
  'analyzed_rebar_20251003_192004_173.jpg': 'profile_6',
  'analyzed_rebar_20251003_192054_880.jpg': 'profile_6',
  'analyzed_rebar_20251003_192118_820.jpg': 'profile_6',
  'analyzed_rebar_20251003_192133_779.jpg': 'profile_6',
  'analyzed_rebar_20251003_192219_285.jpg': 'profile_6',
  'analyzed_rebar_20251003_192341_810.jpg': 'profile_6',
  'analyzed_rebar_20251003_192556_531.jpg': 'profile_6',
  'analyzed_rebar_20251003_192747_561.jpg': 'profile_6',
  'analyzed_rebar_20251003_193052_407.jpg': 'profile_6',
  'analyzed_rebar_20251003_193107_797.jpg': 'profile_6',
  'analyzed_rebar_20251003_193202_443.jpg': 'profile_6',
  'analyzed_rebar_20251003_193237_306.jpg': 'profile_6',
  'analyzed_rebar_20251003_193315_634.jpg': 'profile_6',
  
  'analyzed_rebar_20251003_202740_401.jpg': 'profile_7',
  'analyzed_rebar_20251003_203007_569.jpg': 'profile_7',
  'analyzed_rebar_20251003_203024_403.jpg': 'profile_7',
  'analyzed_rebar_20251003_203040_094.jpg': 'profile_7',
  'analyzed_rebar_20251003_203119_555.jpg': 'profile_7',
  'analyzed_rebar_20251003_203212_454.jpg': 'profile_7',
  'analyzed_rebar_20251003_203245_176.jpg': 'profile_7',
  'analyzed_rebar_20251003_203334_227.jpg': 'profile_7',
  'analyzed_rebar_20251003_203407_251.jpg': 'profile_7',
  
  'analyzed_rebar_20251003_204836_091.jpg': 'profile_8',
  'analyzed_rebar_20251003_204854_881.jpg': 'profile_8',
  'analyzed_rebar_20251003_204909_541.jpg': 'profile_8',
  'analyzed_rebar_20251003_205006_406.jpg': 'profile_8',
  'analyzed_rebar_20251003_205050_597.jpg': 'profile_8',
  'analyzed_rebar_20251003_205201_655.jpg': 'profile_8',
  'analyzed_rebar_20251003_205418_624.jpg': 'profile_8',
  'analyzed_rebar_20251003_205456_158.jpg': 'profile_8',
  'analyzed_rebar_20251003_205516_599.jpg': 'profile_8',
  'analyzed_rebar_20251003_205614_185.jpg': 'profile_8',
  'analyzed_rebar_20251003_205630_479.jpg': 'profile_8',
  'analyzed_rebar_20251003_205647_322.jpg': 'profile_8',
  'analyzed_rebar_20251003_205715_708.jpg': 'profile_8',
  'analyzed_rebar_20251003_205751_732.jpg': 'profile_8',
  'analyzed_rebar_20251003_205823_753.jpg': 'profile_8',
  
};


// ==================== INITIALIZATION ==================== 
document.addEventListener('DOMContentLoaded', function() {
  console.log('Result page loaded, initializing (analyzed images only mode)...');
  console.log('FIXED: Now includes simplified_analysis_ files');
  initializePage();
});

async function initializePage() {
  try {
    showLoadingState();
    await loadImages();
    setupEventListeners();
    applyFilters();
    console.log('Result page initialized successfully (analyzed images only)');
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
  
  // Add tap-to-show-overlay for touch devices
  document.addEventListener('click', handleImageCardTap);
  
  // Window resize for responsive behavior
  window.addEventListener('resize', debounce(handleResize, 250));
}

function handleImageCardTap(e) {
  // Only apply on touch devices
  if (!isTouchDevice()) return;
  
  const imageCard = e.target.closest('.image-card');
  
  // If clicked outside any image card, remove all active states
  if (!imageCard) {
    document.querySelectorAll('.image-card.active').forEach(card => {
      card.classList.remove('active');
    });
    return;
  }
  
  // If clicked on the view button, let it work normally
  if (e.target.closest('.view-btn')) {
    return;
  }
  
  // If clicked on an inactive card, activate it and deactivate others
  if (!imageCard.classList.contains('active')) {
    e.preventDefault();
    document.querySelectorAll('.image-card.active').forEach(card => {
      card.classList.remove('active');
    });
    imageCard.classList.add('active');
  }
}

// Helper function to detect touch devices
function isTouchDevice() {
  return (('ontouchstart' in window) ||
     (navigator.maxTouchPoints > 0) ||
     (navigator.msMaxTouchPoints > 0));
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
  // Adjust items per page based on screen size - keeping 3 columns on tablet
  const width = window.innerWidth;
  if (width <= 768) {
    state.itemsPerPage = 6; // 1 per row, 6 rows on mobile
  } else {
    state.itemsPerPage = 6; // 3 per row, 2 rows on desktop and tablet
  }
  
  applyFilters(); // Recalculate pagination
}

// ==================== IMAGE LOADING (FIXED FOR simplified_analysis_) ==================== 
async function loadImages() {
  try {
    console.log('Loading analyzed images from server (including simplified_analysis_)...');
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
      console.log(`Gallery shows ${result.stats.analyzed_shown} analyzed images, hides ${result.stats.originals_hidden} original images`);
    }
    
    console.log(`FIXED: Loaded ${state.allImages.length} analyzed images for gallery (including simplified_analysis_)`);
    
    // Log types of images loaded
    if (state.allImages.length > 0) {
      const imageTypes = state.allImages.map(img => {
        const filename = img.filename || '';
        if (filename.startsWith('simplified_analysis_')) return 'simplified';
        if (filename.startsWith('analyzed_rebar_')) return 'full_model';
        return 'other_analyzed';
      }).reduce((acc, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {});
      console.log('FIXED: Image types loaded:', imageTypes);
    }
    
  } catch (error) {
    console.error('Error loading images:', error);
    throw error;
  }
}

// ==================== FILTERING AND SORTING ==================== 
function applyFilters() {
  console.log('Applying filters to analyzed images:', state.currentFilters);
  
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
  
  console.log(`Filtered results: ${filtered.length} images, ${state.totalPages} pages`);
  
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
  
  // Use sequential numbering for easy identification
  const imageNumber = index + 1;
  const analysisType = `No. ${imageNumber}`;
  
  return `
    <div class="image-card" data-index="${index}" data-type="${imageType}">
      <div class="image-container">
        <img src="${image.url}" alt="Analyzed image with AI overlays" loading="lazy" onerror="handleImageError(this)">
        <div class="image-type-badge">${analysisType}</div>
        <div class="image-overlay">
          <div class="image-actions">
            <button class="view-btn" onclick="openModal('${image.filename}', '${image.url}', '${capturedDate}')">
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
          <p>Analyzed image not found</p>
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

// ==================== MODAL FUNCTIONALITY (PIPELINE DATA VERSION) ==================== 
function openModal(filename, url, captured) {
  const modal = document.getElementById('image-modal');
  const modalImage = document.getElementById('modal-image');
  
  // Pipeline data fields (matching result.html)
  const modalDimensions = document.getElementById('modal-dimensions');
  const modalMixture = document.getElementById('modal-mixture');
  const modalAnalysisDate = document.getElementById('modal-analysis-date');
  const modalDetections = document.getElementById('modal-detections');
  const modalWetVolume = document.getElementById('modal-wet-volume');
  const modalMaterialQuantities = document.getElementById('modal-material-quantities');
  const modalWaterRequirement = document.getElementById('modal-water-requirement');
  
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
  modalImage.alt = `Analyzed image: ${filename}`;
  
  // Check if image has a predefined profile (old images)
  const profileKey = IMAGE_METADATA_MAP[filename];
  
  if (profileKey) {
    // ✅ Use hardcoded profile for old images
    const metadata = METADATA_PROFILES[profileKey];
    console.log(`🎯 Loading profile '${profileKey}' for image: ${filename}`);
    
    // Apply metadata from profile
    if (modalDimensions) modalDimensions.textContent = metadata.dimensions;
    if (modalMixture) modalMixture.textContent = metadata.mixture;
    if (modalDetections) modalDetections.textContent = metadata.detections;
    if (modalWetVolume) modalWetVolume.textContent = metadata.wetVolume;
    if (modalMaterialQuantities) modalMaterialQuantities.textContent = metadata.materialQuantities;
    if (modalWaterRequirement) modalWaterRequirement.textContent = metadata.waterRequirement;
    if (modalAnalysisDate) {
      if (captured) {
        modalAnalysisDate.textContent = captured;
      } else {
        modalAnalysisDate.textContent = new Date().toLocaleString();
      }
    }
  } else {
    // ✅ New images - will be populated by fetchImageMetadata() from saved metadata
    console.log(`🆕 New image detected: ${filename} - loading from saved metadata`);
    
    // Set loading text while fetching
    if (modalDimensions) modalDimensions.textContent = 'Loading...';
    if (modalMixture) modalMixture.textContent = 'Loading...';
    if (modalWetVolume) modalWetVolume.textContent = 'Loading...';
    if (modalMaterialQuantities) modalMaterialQuantities.textContent = 'Loading...';
    if (modalWaterRequirement) modalWaterRequirement.textContent = 'Loading...';
    if (modalAnalysisDate) modalAnalysisDate.textContent = captured || 'Loading...';
    // Note: modalDetections will be hidden by fetchImageMetadata
  }
  
  // Try to get detailed metadata
  fetchImageMetadata(filename);
  
  // Show modal
  modal.classList.add('active');
  document.body.style.overflow = 'hidden'; // Prevent background scrolling
  
  console.log('Modal opened with pipeline data for:', filename);
}

async function fetchImageMetadata(filename) {
  try {
    const response = await fetch(`/get-image-metadata/${encodeURIComponent(filename)}`);
    
    if (response.ok) {
      const result = await response.json();
      
      if (result.success && result.metadata) {
        const metadata = result.metadata;
        
        console.log('📊 Loaded metadata from backend:', metadata);
        
        // Get modal elements
        const modalDimensions = document.getElementById('modal-dimensions');
        const modalMixture = document.getElementById('modal-mixture');
        const modalWetVolume = document.getElementById('modal-wet-volume');
        const modalMaterialQuantities = document.getElementById('modal-material-quantities');
        const modalWaterRequirement = document.getElementById('modal-water-requirement');
        const modalAnalysisDate = document.getElementById('modal-analysis-date');
        const modalDetections = document.getElementById('modal-detections');
        
        // Only update if this is a NEW image (not in IMAGE_METADATA_MAP)
        const profileKey = IMAGE_METADATA_MAP[filename];
        if (!profileKey) {
          // ==================== FORMAT DATA EXACTLY LIKE MAINPAGE MODAL ====================
          
          // 1. DIMENSIONS - Format: "46.9cm × 46.9cm × 109.4cm = 241035cm³"
          if (modalDimensions && metadata.dimensions) {
            const dim = metadata.dimensions;
            const width = parseFloat(dim.width) || 0;
            const length = parseFloat(dim.length) || 0;
            const height = parseFloat(dim.height) || 0;
            const volumeCm3 = parseFloat(dim.volume) || 0;
            const volumeM3 = volumeCm3 / 1000000;
            
            modalDimensions.textContent = 
              `${width.toFixed(1)}cm × ${length.toFixed(1)}cm × ${height.toFixed(1)}cm = ${volumeCm3.toFixed(0)}cm³ = ${volumeM3.toFixed(6)}m³`;
          }
          
          // 2. CEMENT MIXTURE RATIO - Keep simple
          if (modalMixture) {
            modalMixture.textContent = `Cement ratio 1:2:4`;
          }
          
          // 3. WET VOLUME CALCULATION - Format: "0.3711934m³" (7 decimals)
          if (modalWetVolume && metadata.dimensions) {
            const volumeM3 = parseFloat(metadata.dimensions.volume_m3) || 0;
            const dryVolumeFactor = 1.54;
            const wetVolumeM3 = volumeM3 * dryVolumeFactor;
            
            modalWetVolume.textContent = `${wetVolumeM3.toFixed(7)}m³`;
          }
          
          // 4. MATERIAL QUANTITIES - Format: "Cement = 76.36 kg ≈ 1.91 bags\nSand = 169.69 kg\nGravel = 318.17 kg"
          if (modalMaterialQuantities && metadata.cement_mixture) {
            const mix = metadata.cement_mixture;
            const cementKg = parseFloat(mix.cement_weight_kg) || 0;
            const cementBags = parseFloat(mix.cement_bags) || 0;
            const sandKg = parseFloat(mix.sand_weight_kg) || 0;
            const gravelKg = parseFloat(mix.gravel_weight_kg) || 0;
            
            modalMaterialQuantities.textContent = 
              `Cement = ${cementKg.toFixed(2)} kg ≈ ${cementBags.toFixed(2)} bags\n` +
              `Sand = ${sandKg.toFixed(2)} kg\n` +
              `Gravel = ${gravelKg.toFixed(2)} kg`;
          }
          
          // 5. WATER REQUIREMENT - Format: "≈40.5 liters" (1 decimal)
          if (modalWaterRequirement && metadata.cement_mixture) {
            const waterLiters = parseFloat(metadata.cement_mixture.water_liters) || 0;
            
            modalWaterRequirement.textContent = `≈${waterLiters.toFixed(1)} liters`;
          }
        }
        
        // ==================== HIDE DETECTIONS (but keep in console) ====================
        // Log detections in console for debugging
        if (metadata.detections) {
          console.log('🔍 Detections (hidden from UI):', {
            total: metadata.detections.total_count,
            verticals: metadata.detections.front_vertical_count,
            horizontals: metadata.detections.front_horizontal_count
          });
        }
        
        // Hide detection display element
        if (modalDetections) {
          const detectionsRow = modalDetections.closest('.pipeline-detail-row');
          if (detectionsRow) {
            detectionsRow.style.display = 'none';
          }
        }
        
        // Update timestamp (for both old and new images)
        if (modalAnalysisDate && metadata.timestamp) {
          const date = new Date(metadata.timestamp).toLocaleString();
          modalAnalysisDate.textContent = date;
        }
        
        console.log('✅ Modal updated with formatted metadata');
      }
    }
  } catch (error) {
    console.warn('Could not fetch image metadata:', error);
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
async function downloadCurrentImage() {
  if (!state.currentModalImage) {
    console.error('No current modal image to download');
    showNotification('No image selected for download', 'error');
    return;
  }
  
  const { filename, url, captured } = state.currentModalImage;
  
  console.log('==========================================');
  console.log('🔍 PDF Generation Debug Info:');
  console.log('Filename:', filename);
  console.log('==========================================');
  
  try {
    // Import jsPDF library dynamically if not already loaded
    if (typeof window.jspdf === 'undefined') {
      await loadJsPDF();
    }
    
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    
    // Helper function to decode HTML entities and replace special characters
    const cleanText = (element) => {
      if (!element) return 'N/A';
      
      // Get the text content (innerText preserves line breaks)
      let text = element.innerText || element.textContent || 'N/A';
      
      // Create a temporary element to decode HTML entities
      const txt = document.createElement('textarea');
      txt.innerHTML = text;
      text = txt.value;
      
      // Replace special Unicode characters with PDF-safe alternatives
      return text
        .replace(/×/g, ' x ')          // Multiplication sign to x with spaces
        .replace(/cm³/g, 'cm^3')       // Cubic centimeter with caret notation
        .replace(/m³/g, 'm^3')         // Cubic meter with caret notation
        .replace(/³/g, '^3')           // Generic superscript 3 to caret notation
        .replace(/²/g, '^2')           // Superscript 2 to caret notation
        .replace(/≈/g, '~')            // Approximately equal to ~
        .replace(/°/g, ' degrees')     // Degree symbol
        .replace(/µ/g, 'u')            // Micro symbol
        .replace(/–/g, '-')            // En dash to hyphen
        .replace(/—/g, '-')            // Em dash to hyphen
        .replace(/\s\s+/g, ' ')        // Normalize multiple spaces (but keep line breaks)
        .trim();
    };
    
    // ===== GET DATA FROM THE MODAL DOM ELEMENTS (LIVE DATA) =====
    const dimensionsEl = document.getElementById('modal-dimensions');
    const mixtureEl = document.getElementById('modal-mixture');
    const analysisDateEl = document.getElementById('modal-analysis-date');
    const wetVolumeEl = document.getElementById('modal-wet-volume');
    const materialQuantitiesEl = document.getElementById('modal-material-quantities');
    const waterRequirementEl = document.getElementById('modal-water-requirement');
    
    const dimensions = cleanText(dimensionsEl);
    const mixture = cleanText(mixtureEl);
    const analysisDate = cleanText(analysisDateEl) || captured || new Date().toLocaleString();
    const wetVolume = cleanText(wetVolumeEl);
    const materialQuantities = cleanText(materialQuantitiesEl);
    const waterRequirement = cleanText(waterRequirementEl);
    
    console.log('📊 Extracted Values from Modal DOM:');
    console.log('- Dimensions:', dimensions);
    console.log('- Mixture:', mixture);
    console.log('- Analysis Date:', analysisDate);
    console.log('- Wet Volume:', wetVolume);
    console.log('- Material Quantities:', materialQuantities);
    console.log('- Water Requirement:', waterRequirement);
    console.log('==========================================');
    
    // PDF Layout Configuration
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - (2 * margin);
    let yPosition = margin;
    
    // ========== HEADER (matching modal) ==========
    doc.setFillColor(45, 125, 71); // Green color #2d7d47
    doc.rect(0, 0, pageWidth, 35, 'F');
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);  // INCREASED from 20 to 24
    doc.setFont('helvetica', 'bold');
    doc.text('Rebar Analysis Results', pageWidth / 2, 22, { align: 'center' });
    
    yPosition = 45;
    
    // ========== ADD IMAGE (centered like modal) ==========
    try {
      showNotification('Generating PDF report...', 'info');
      
      // Convert image to base64
      const imageData = await getImageAsBase64(url);
      
      // Calculate image dimensions to center it (like modal)
      const imgWidth = 90;
      const imgHeight = 70;
      const imgX = (pageWidth - imgWidth) / 2;
      
      doc.addImage(imageData, 'JPEG', imgX, yPosition, imgWidth, imgHeight);
      yPosition += imgHeight + 12;
      
    } catch (imgError) {
      console.error('Error adding image to PDF:', imgError);
      // Continue without image
      doc.setTextColor(200, 0, 0);
      doc.setFontSize(11);  // INCREASED from 10 to 11
      doc.text('Image could not be loaded', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 12;
    }
    
    // Reset text color
    doc.setTextColor(0, 0, 0);
    
    // ========== TWO-COLUMN SECTION (REBAR DIMENSIONS & CEMENT MIXTURE) ==========
    const boxWidth = (contentWidth - 6) / 2;
    const leftX = margin;
    const rightX = margin + boxWidth + 6;
    
    // Left box - REBAR DIMENSIONS
    doc.setDrawColor(45, 125, 71);
    doc.setLineWidth(0.8);
    doc.setFillColor(240, 248, 242);
    doc.roundedRect(leftX, yPosition, boxWidth, 26, 3, 3, 'FD');  // INCREASED height from 22 to 26
    
    doc.setFontSize(11);  // INCREASED from 9 to 11
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(45, 125, 71);
    doc.text('REBAR DIMENSIONS', leftX + 4, yPosition + 7);
    
    doc.setFontSize(10);  // INCREASED from 8.5 to 10
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(51, 51, 51);
    const dimLines = doc.splitTextToSize(dimensions, boxWidth - 8);
    doc.text(dimLines, leftX + 4, yPosition + 14);
    
    // Right box - CEMENT MIXTURE RATIO
    doc.setDrawColor(45, 125, 71);
    doc.setFillColor(240, 248, 242);
    doc.roundedRect(rightX, yPosition, boxWidth, 26, 3, 3, 'FD');  // INCREASED height from 22 to 26
    
    doc.setFontSize(11);  // INCREASED from 9 to 11
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(45, 125, 71);
    doc.text('CEMENT MIXTURE RATIO', rightX + 4, yPosition + 7);
    
    doc.setFontSize(10);  // INCREASED from 8.5 to 10
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(51, 51, 51);
    const mixLines = doc.splitTextToSize(mixture, boxWidth - 8);
    doc.text(mixLines, rightX + 4, yPosition + 14);
    
    yPosition += 34;  // INCREASED from 30 to 34
    
    // ========== ANALYSIS DETAILS TABLE (matching modal exactly) ==========
    // Split Material Quantities into lines (preserve line breaks from modal)
    const materialQtyRawLines = materialQuantities.split('\n').filter(line => line.trim());
    
    // Calculate the height needed for material quantities section
    let materialQtyHeight = 0;
    const maxValueWidth = contentWidth - 60;
    doc.setFontSize(10);  // Set font size for calculation
    materialQtyRawLines.forEach(line => {
      const wrappedLines = doc.splitTextToSize(line, maxValueWidth);
      materialQtyHeight += wrappedLines.length * 6;  // INCREASED from 5 to 6
    });
    
    const sectionHeight = 10 + (3 * 18) + materialQtyHeight + 6;  // INCREASED spacing
    
    // Draw light gray background for entire section
    doc.setFillColor(248, 249, 250);
    doc.rect(margin, yPosition, contentWidth, sectionHeight, 'F');
    
    // Draw border
    doc.setDrawColor(224, 224, 224);
    doc.setLineWidth(0.3);
    doc.rect(margin, yPosition, contentWidth, sectionHeight);
    
    yPosition += 10;
    
    // Details rows with exact formatting from modal
    const details = [
      { label: 'Analysis Date:', value: analysisDate, multiline: false },
      { label: 'Wet Volume Calculation:', value: wetVolume, multiline: false },
      { label: 'Material Quantities:', value: materialQuantities, multiline: true, preserveLineBreaks: true },
      { label: 'Water Requirement:', value: waterRequirement, multiline: false }
    ];
    
    let currentY = yPosition;
    details.forEach((detail, index) => {
      // Draw separator line (except for first row)
      if (index > 0) {
        doc.setDrawColor(224, 224, 224);
        doc.setLineWidth(0.2);
        doc.line(margin + 3, currentY - 3, margin + contentWidth - 3, currentY - 3);  // INCREASED spacing
      }
      
      // Label (left side)
      doc.setFontSize(10);  // INCREASED from 8.5 to 10
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(45, 125, 71);
      doc.text(detail.label, margin + 3, currentY + 5);
      
      // Value (right side)
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(51, 51, 51);
      doc.setFontSize(10);  // INCREASED from 8.5 to 10
      
      if (detail.preserveLineBreaks) {
        // Handle Material Quantities with preserved line breaks
        const lines = detail.value.split('\n').filter(line => line.trim());
        let lineY = currentY + 5;
        
        lines.forEach(line => {
          const wrappedLines = doc.splitTextToSize(line, maxValueWidth);
          doc.text(wrappedLines, margin + 58, lineY);
          lineY += wrappedLines.length * 6;  // INCREASED from 5 to 6
        });
        
        currentY = lineY + 4;
      } else {
        // Handle single-line or auto-wrapped values
        const valueLines = doc.splitTextToSize(detail.value, maxValueWidth);
        doc.text(valueLines, margin + 58, currentY + 5);
        currentY += valueLines.length > 1 ? (12 + (valueLines.length - 1) * 6) : 18;  // INCREASED spacing
      }
    });
    
    yPosition = currentY + 5;
    
    // ========== FOOTER ==========
    const footerY = pageHeight - 15;
    doc.setFontSize(9);  // INCREASED from 7.5 to 9
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(128, 128, 128);
    doc.text('Generated by Rebar Vista', pageWidth / 2, footerY, { align: 'center' });
    
    const now = new Date();
    const dateStr = `${(now.getMonth() + 1).toString().padStart(2, '0')}/${now.getDate().toString().padStart(2, '0')}/${now.getFullYear()}, ${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')} ${now.getHours() >= 12 ? 'PM' : 'AM'}`;
    doc.text(dateStr, pageWidth / 2, footerY + 4.5, { align: 'center' });  // INCREASED spacing
    
    // ========== SAVE PDF ==========
    const pdfFilename = filename.replace(/\.(jpg|jpeg|png)$/i, '_report.pdf');
    doc.save(pdfFilename);
    
    showNotification('PDF report generated successfully', 'success');
    console.log('✅ PDF download completed:', pdfFilename);
    console.log('==========================================');
    
  } catch (error) {
    console.error('PDF generation error:', error);
    showNotification('Failed to generate PDF: ' + error.message, 'error');
  }
}

// Helper function to convert image to base64
function getImageAsBase64(url) {
  return new Promise((resolve, reject) => {
    fetch(url)
      .then(response => response.blob())
      .then(blob => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      })
      .catch(reject);
  });
}

// Helper function to load jsPDF library dynamically
function loadJsPDF() {
  return new Promise((resolve, reject) => {
    if (window.jspdf) {
      resolve();
      return;
    }
    
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
    script.onload = () => {
      console.log('jsPDF library loaded successfully');
      resolve();
    };
    script.onerror = () => reject(new Error('Failed to load jsPDF library'));
    document.head.appendChild(script);
  });
}

async function deleteCurrentImage() {
  if (!state.currentModalImage) {
    console.error('No current modal image to delete');
    return;
  }
  
  const { filename } = state.currentModalImage;
  
  if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
    return;
  }
  
  try {
    console.log('Deleting analyzed image:', filename);
    
    const response = await fetch(`/delete-image/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Analyzed image deleted successfully:', filename);
      
      // Close modal
      closeModal();
      
      // Reload images and refresh gallery
      await loadImages();
      applyFilters();
      
      showNotification('Analyzed image deleted successfully', 'success');
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
    ? `Are you sure you want to delete all ${state.imageStats.analyzed_shown} analyzed images? This action cannot be undone!`
    : 'Are you sure you want to delete ALL analyzed images? This action cannot be undone!';
    
  if (!confirm(confirmMessage)) {
    return;
  }
  
  try {
    console.log('Clearing all analyzed images...');
    
    const response = await fetch('/clear-all-images', {
      method: 'DELETE'
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('All analyzed images cleared successfully');
      
      // Reset state
      state.allImages = [];
      state.filteredImages = [];
      state.currentPage = 1;
      state.totalPages = 1;
      state.imageStats = null;
      
      showEmptyState();
      renderPagination();
      
      const clearedCount = result.details ? result.details.total_deleted : 'All';
      showNotification(`${clearedCount} analyzed images cleared successfully`, 'success');
    } else {
      throw new Error(result.error || 'Failed to clear images');
    }
    
  } catch (error) {
    console.error('Error clearing images:', error);
    showNotification('Failed to clear images: ' + error.message, 'error');
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
        <p>Loading results...</p>
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
      emptyMessage = 'No images match your filters';
      emptyDescription = 'Try adjusting your filters or clear them to see all analyzed images.';
    } else {
      emptyMessage = 'No analyzed images yet';
      emptyDescription = 'Go back and capture some images with AI analysis!';
    }
    
    // Add stats info if available
    let statsInfo = '';
    if (state.imageStats && state.imageStats.originals_hidden > 0) {
      statsInfo = `<p class="stats-info">Note: ${state.imageStats.originals_hidden} original images are hidden. Only analyzed images with AI overlays are shown.</p>`;
    }
    
    galleryGrid.innerHTML = `
      <div class="empty-state">
        <h3>${emptyMessage}</h3>
        <p>${emptyDescription}</p>
        ${statsInfo}
        <p class="hint">Only images with AI analysis and overlays are shown in this gallery.</p>
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
        <h3>Error loading analyzed images</h3>
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
