// Modern Welcome Page JavaScript
// Complete replacement for old welcome.js functionality

// Detect mobile/tablet devices
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                 window.innerWidth <= 1024;

// Detect if user prefers reduced motion
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Brand name animation state
let brandNameVisible = true;
let lastScrollY = 0;
const scrollThreshold = 50; // Minimum scroll distance to trigger animation

// Brand name animation functions
function hideBrandName() {
  if (prefersReducedMotion) return; // Respect reduced motion preference
  
  const brandName = document.querySelector('.brand-name');
  if (brandName && brandNameVisible) {
    brandName.classList.add('hidden');
    brandNameVisible = false;
  }
}

function showBrandName() {
  const brandName = document.querySelector('.brand-name');
  if (brandName && !brandNameVisible) {
    brandName.classList.remove('hidden');
    brandNameVisible = true;
  }
}
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

// Scroll to top function for logo and brand name clicks
function scrollToTop() {
  // Disable scroll detection temporarily to prevent tab highlighting during scroll to top
  isUserScrolling = false;
  isScrollingToTop = true;
  
  // Always show brand name when going to top
  showBrandName();
  
  if (!prefersReducedMotion) {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  } else {
    window.scrollTo(0, 0);
  }
  
  // Reset active tab state immediately
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  
  // Reset the scrolling flags after scroll completes
  setTimeout(() => {
    isScrollingToTop = false;
    isUserScrolling = false;
    lastScrollY = 0; // Reset scroll position tracking
  }, prefersReducedMotion ? 100 : 1000); // Longer timeout for smooth scroll
}

// Navigation functionality
function updateActiveTab(targetSection) {
  // Update nav tabs
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  const targetTab = document.querySelector(`[data-target="${targetSection}"]`);
  if (targetTab) {
    targetTab.classList.add('active');
  }
  
  // Hide brand name when navigating to tabs (not when at top)
  if (targetSection && window.scrollY > 100) {
    hideBrandName();
  }
}

// Smooth scroll to element with focus animation
function smoothScrollToElement(element, offset = 100) {
  if (!element) return;
  
  const elementTop = element.offsetTop - offset;
  
  // Use native smooth scrolling if available and user doesn't prefer reduced motion
  if ('scrollBehavior' in document.documentElement.style && !prefersReducedMotion) {
    window.scrollTo({
      top: elementTop,
      behavior: 'smooth'
    });
  } else {
    // Fallback for older browsers or reduced motion preference
    window.scrollTo(0, elementTop);
  }
  
  // Add focus animation to the element (only if not reduced motion)
  if (!prefersReducedMotion) {
    element.classList.add('focusing');
    setTimeout(() => {
      element.classList.remove('focusing');
    }, 600);
  }
}

// Optimized accordion functionality with improved focus behavior
function toggleAccordion(button) {
  // Temporarily disable scroll detection
  isUserScrolling = false;
  
  const item = button.parentElement;
  const content = item.querySelector('.accordion-content');
  const icon = item.querySelector('.accordion-icon');
  
  // Get current state before any changes
  const isCurrentlyActive = item.classList.contains('active');
  
  // Close all other accordions in the same container with optimized animations
  const accordion = item.closest('.accordion');
  const otherActiveItems = accordion.querySelectorAll('.accordion-item.active');
  
  // Check if there are other active items that will be closed
  const hasOtherActiveItems = Array.from(otherActiveItems).some(otherItem => otherItem !== item);
  
  // Batch DOM updates for better performance
  requestAnimationFrame(() => {
    otherActiveItems.forEach(otherItem => {
      if (otherItem !== item) {
        otherItem.classList.remove('active');
        const otherIcon = otherItem.querySelector('.accordion-icon');
        if (otherIcon) {
          otherIcon.textContent = '+';
        }
      }
    });
    
    // Toggle current accordion
    if (isCurrentlyActive) {
      item.classList.remove('active');
      icon.textContent = '+';
    } else {
      item.classList.add('active');
      icon.textContent = '−';
      
      // Apply scroll and focus animation with improved logic
      // Skip animation if other accordions were open (switching between accordions)
      if (!hasOtherActiveItems) {
        // Smooth scroll to the expanded accordion with focus animation
        setTimeout(() => {
          // Calculate optimal scroll position based on current accordion position
          const accordionTop = item.offsetTop;
          const windowHeight = window.innerHeight;
          const headerHeight = 100; // Account for fixed header
          
          // Simple approach: scroll to accordion header with some padding
          const targetScroll = accordionTop - headerHeight - 20; // 20px additional padding
          
          // Smooth scroll with focus animation
          if (!prefersReducedMotion) {
            window.scrollTo({
              top: Math.max(0, targetScroll), // Ensure we don't scroll to negative position
              behavior: 'smooth'
            });
            
            // Add focus animation
            item.classList.add('focusing');
            setTimeout(() => {
              item.classList.remove('focusing');
            }, 600);
          } else {
            // Just scroll without animation for reduced motion preference
            window.scrollTo(0, Math.max(0, targetScroll));
          }
          
        }, isMobile ? 200 : 100); // Longer delay on mobile for accordion animation
      }
    }
  });
  
  // Prevent event bubbling
  if (event) {
    event.stopPropagation();
  }
  
  // Reset scroll detection flag
  const timeout = isMobile ? 400 : 300;
  setTimeout(() => {
    isUserScrolling = false;
  }, timeout);
}

// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
  // Navigation tab click handlers
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const target = tab.getAttribute('data-target');
      
      // Hide brand name when navigating to sections
      hideBrandName();
      
      updateActiveTab(target);
      const element = document.getElementById(target);
      if (element) {
        smoothScrollToElement(element, 100);
      }
    });
  });

  // Hero CTA click handler
  document.querySelectorAll('.hero-cta').forEach(cta => {
    cta.addEventListener('click', (e) => {
      e.preventDefault();
      const target = cta.getAttribute('href').substring(1);
      
      // Hide brand name when navigating via hero CTA
      hideBrandName();
      
      const element = document.getElementById(target);
      if (element) {
        smoothScrollToElement(element, 100);
      }
    });
  });

  // Prevent accordion content clicks from affecting accordion state
  document.querySelectorAll('.accordion-content').forEach(content => {
    content.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  });

  // Prevent accordion body clicks from affecting accordion state
  document.querySelectorAll('.accordion-body').forEach(body => {
    body.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  });

  // Mobile-specific optimizations
  if (isMobile) {
    // Add touch event handling for better responsiveness
    document.querySelectorAll('.accordion-header').forEach(header => {
      let touchStartTime;
      
      header.addEventListener('touchstart', function(e) {
        touchStartTime = Date.now();
        if (!prefersReducedMotion) {
          this.style.backgroundColor = '#f8f9fa';
        }
      }, { passive: true });
      
      header.addEventListener('touchend', function(e) {
        const touchDuration = Date.now() - touchStartTime;
        // Only apply styling if it was a quick touch (not a scroll)
        if (touchDuration < 200 && !prefersReducedMotion) {
          setTimeout(() => {
            this.style.backgroundColor = 'white';
          }, 150);
        } else {
          this.style.backgroundColor = 'white';
        }
      }, { passive: true });
    });

    // Add touch feedback for logo and brand name
    const clickableElements = document.querySelectorAll('.nav-logo, .brand-name');
    clickableElements.forEach(element => {
      element.addEventListener('touchstart', function(e) {
        if (!prefersReducedMotion) {
          this.style.transform = 'scale(0.95)';
        }
      }, { passive: true });
      
      element.addEventListener('touchend', function(e) {
        if (!prefersReducedMotion) {
          setTimeout(() => {
            this.style.transform = '';
          }, 150);
        }
      }, { passive: true });
    });
  }

  // Initialize accessibility features
  improveAccessibility();
});

// Scroll detection for navigation highlighting
let isUserScrolling = false;
let isScrollingToTop = false;
let scrollTimeout;

function updateNavigationOnScroll() {
  // Don't update navigation if user is scrolling to top or not actually scrolling
  if (!isUserScrolling || isScrollingToTop) return;
  
  const currentScrollY = window.scrollY;
  const sections = ['general', 'tutorial'];
  const scrollPos = currentScrollY + 150;
  
  // Handle brand name animation based on scroll behavior
  if (currentScrollY > scrollThreshold && currentScrollY > lastScrollY) {
    // Scrolling down and past threshold - hide brand name
    hideBrandName();
  } else if (currentScrollY <= scrollThreshold || currentScrollY < lastScrollY - 20) {
    // At top or scrolling up significantly - show brand name
    showBrandName();
  }
  
  // Update last scroll position
  lastScrollY = currentScrollY;
  
  // Only highlight tabs if we're actually in a content section, not at the top
  if (scrollPos < 200) {
    // If we're near the top, clear all active tabs and show brand name
    document.querySelectorAll('.nav-tab').forEach(tab => {
      tab.classList.remove('active');
    });
    showBrandName();
    return;
  }
  
  sections.forEach(sectionId => {
    const section = document.getElementById(sectionId);
    if (section && scrollPos >= section.offsetTop && scrollPos < section.offsetTop + section.offsetHeight) {
      updateActiveTab(sectionId);
    }
  });
}

// Optimized scroll event listener with debouncing
const debouncedScrollUpdate = debounce(updateNavigationOnScroll, 50);

window.addEventListener('scroll', function(e) {
  // Don't process scroll events if we're scrolling to top
  if (isScrollingToTop) return;
  
  // Mark that user is scrolling
  isUserScrolling = true;
  
  // Use debounced function for better performance
  debouncedScrollUpdate();
  
  // Reset scrolling flag after a delay
  if (scrollTimeout) {
    clearTimeout(scrollTimeout);
  }
  scrollTimeout = setTimeout(() => {
    isUserScrolling = false;
  }, 150);
}, { passive: true });

// Start app function - redirects to main page
function startApp() {
  window.location.href = '/mainpage.html';
}

// Handle browser back/forward buttons
window.addEventListener('popstate', function(e) {
  if (e.state && e.state.section) {
    updateActiveTab(e.state.section);
  }
});

// Initialize page state
function initializePage() {
  // Ensure brand name is visible on page load
  showBrandName();
  
  // Check if there's a hash in URL and scroll to it
  if (window.location.hash) {
    const target = window.location.hash.substring(1);
    setTimeout(() => {
      const element = document.getElementById(target);
      if (element) {
        smoothScrollToElement(element, 100);
        
        // Update navigation based on hash
        if (target === 'general' || target === 'tutorial') {
          updateActiveTab(target);
        }
      }
    }, 100);
  }
}

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
  // Escape key closes any open mobile menus
  if (e.key === 'Escape') {
    const navTabs = document.querySelector('.nav-tabs');
    if (navTabs) {
      navTabs.classList.remove('mobile-open');
    }
  }
  
  // Enter or Space on accordion headers
  if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('accordion-header')) {
    e.preventDefault();
    toggleAccordion(e.target);
  }

  // Enter or Space on logo or brand name
  if ((e.key === 'Enter' || e.key === ' ') && (e.target.classList.contains('nav-logo') || e.target.classList.contains('brand-name'))) {
    e.preventDefault();
    scrollToTop();
  }
});

// Accessibility improvements
function improveAccessibility() {
  // Add ARIA attributes to accordion items
  document.querySelectorAll('.accordion-item').forEach((item, index) => {
    const header = item.querySelector('.accordion-header');
    const content = item.querySelector('.accordion-content');
    
    if (header && content) {
      const id = `accordion-${index}`;
      header.setAttribute('aria-expanded', item.classList.contains('active'));
      header.setAttribute('aria-controls', id);
      header.setAttribute('tabindex', '0'); // Make focusable
      content.setAttribute('id', id);
      content.setAttribute('aria-hidden', !item.classList.contains('active'));
    }
  });

  // Add accessibility attributes to logo and brand name
  const logo = document.querySelector('.nav-logo');
  const brandName = document.querySelector('.brand-name');
  
  if (logo) {
    logo.setAttribute('tabindex', '0');
    logo.setAttribute('aria-label', 'Rebar Vista logo - scroll to top');
    logo.setAttribute('role', 'button');
  }
  
  if (brandName) {
    brandName.setAttribute('tabindex', '0');
    brandName.setAttribute('aria-label', 'Rebar Vista - scroll to top');
    brandName.setAttribute('role', 'button');
  }
  
  // Update ARIA attributes when accordions change (optimized observer)
  const observer = new MutationObserver(debounce(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
        const item = mutation.target;
        if (item.classList.contains('accordion-item')) {
          const header = item.querySelector('.accordion-header');
          const content = item.querySelector('.accordion-content');
          
          if (header && content) {
            const isActive = item.classList.contains('active');
            header.setAttribute('aria-expanded', isActive);
            content.setAttribute('aria-hidden', !isActive);
          }
        }
      }
    });
  }, 100));
  
  // Start observing
  document.querySelectorAll('.accordion-item').forEach(item => {
    observer.observe(item, { attributes: true, attributeFilter: ['class'] });
  });
}

// Resize handler for responsive behavior
const debouncedResize = debounce(function() {
  // Reset any mobile-specific states if needed
  const navTabs = document.querySelector('.nav-tabs');
  if (navTabs && window.innerWidth > 768) {
    navTabs.classList.remove('mobile-open');
  }
}, 250);

window.addEventListener('resize', debouncedResize, { passive: true });

// Initialize page
document.addEventListener('DOMContentLoaded', initializePage);

// Export functions for global access
window.toggleAccordion = toggleAccordion;
window.startApp = startApp;
window.updateActiveTab = updateActiveTab;
window.scrollToTop = scrollToTop;

// Legacy function support (if needed for backward compatibility)
function toggleCollapse(section) {
  console.warn('toggleCollapse is deprecated. Use toggleAccordion instead.');
}

function goToMain() {
  startApp();
}

// Performance monitoring (development only - remove in production)
if (typeof performance !== 'undefined' && performance.mark) {
  // Mark key performance points
  window.addEventListener('load', () => {
    performance.mark('welcome-page-loaded');
  });
}

// Intersection Observer for better scroll performance (if supported)
if ('IntersectionObserver' in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    // Don't process intersection events if scrolling to top or not user scrolling
    if (!isUserScrolling || isScrollingToTop) return;
    
    entries.forEach(entry => {
      if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
        const sectionId = entry.target.id;
        if (sectionId === 'general' || sectionId === 'tutorial') {
          // Only update if we're not near the top of the page
          if (window.scrollY > 200) {
            updateActiveTab(sectionId);
          }
        }
      }
    });
  }, {
    threshold: [0.5],
    rootMargin: '-100px 0px -100px 0px'
  });

  // Observe sections
  document.addEventListener('DOMContentLoaded', () => {
    const sections = document.querySelectorAll('#general, #tutorial');
    sections.forEach(section => sectionObserver.observe(section));
  });
}