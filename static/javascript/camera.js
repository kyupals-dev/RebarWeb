// Real-time camera.js - No delay streaming approach

let isUsingWebRTC = false;
let isUsingServerFeed = false;  // Disable slow server feed
let streamingCanvas = null;
let streamingContext = null;

// Try WebRTC first for zero-delay local camera access
async function startCamera() {
  console.log('🎥 Starting real-time camera feed...');
  
  // First attempt: Direct WebRTC for zero delay
  try {
    await useWebRTCDirect();
  } catch (error) {
    console.log('WebRTC failed, trying server stream optimization...');
    await useOptimizedServerFeed();
  }
}

// Direct WebRTC camera access (zero delay)
async function useWebRTCDirect() {
  console.log('🔄 Attempting direct WebRTC camera access...');
  
  const video = document.getElementById('camera-feed');
  if (!video) {
    throw new Error('Camera feed element not found');
  }

  // Request camera with specific A4Tech-like settings
  const constraints = {
    video: {
      width: { ideal: 640, min: 480 },
      height: { ideal: 480, min: 360 },
      frameRate: { ideal: 30, min: 15 },
      facingMode: 'environment', // Try back camera first
      deviceId: undefined // Let browser choose best camera
    }
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    
    // Create video element if it doesn't exist or replace img
    let videoElement = document.getElementById('camera-feed');
    if (videoElement.tagName !== 'VIDEO') {
      const newVideo = document.createElement('video');
      newVideo.id = 'camera-feed';
      newVideo.autoplay = true;
      newVideo.playsinline = true;
      newVideo.muted = true;
      newVideo.style.width = '100%';
      newVideo.style.height = '100%';
      newVideo.style.objectFit = 'contain';
      newVideo.style.transform = 'rotate(90deg)'; // Rotate for portrait like A4Tech
      
      videoElement.parentNode.replaceChild(newVideo, videoElement);
      videoElement = newVideo;
    }
    
    videoElement.srcObject = stream;
    
    // Wait for video to start
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play().then(resolve).catch(resolve);
      };
    });
    
    isUsingWebRTC = true;
    isUsingServerFeed = false;
    
    updateCameraStatus('WebRTC Real-time Active (Zero Delay)');
    console.log('✅ WebRTC camera active - Zero delay!');
    
    // Log actual video dimensions
    console.log(`📐 WebRTC Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
    
  } catch (error) {
    console.error('WebRTC setup failed:', error);
    throw error;
  }
}

// Optimized server feed with faster refresh (fallback)
async function useOptimizedServerFeed() {
  console.log('🔄 Setting up optimized server feed...');
  
  const currentElement = document.getElementById('camera-feed');
  if (!currentElement) {
    throw new Error('Camera feed element not found');
  }

  // Create canvas for faster rendering
  const canvas = document.createElement('canvas');
  canvas.id = 'camera-feed';
  canvas.width = 360;
  canvas.height = 480;
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.objectFit = 'contain';
  canvas.style.display = 'block';
  
  const ctx = canvas.getContext('2d');
  
  // Replace current element with canvas
  currentElement.parentNode.replaceChild(canvas, currentElement);
  
  streamingCanvas = canvas;
  streamingContext = ctx;
  
  isUsingWebRTC = false;
  isUsingServerFeed = true;
  
  // Start rapid image updates
  updateCanvasStream();
  
  updateCameraStatus('Server Feed Optimized (Low Delay)');
  console.log('✅ Optimized server feed active');
}

// Ultra-fast canvas streaming
async function updateCanvasStream() {
  if (!isUsingServerFeed || !streamingCanvas || !streamingContext) return;
  
  try {
    // Create a new image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      // Clear canvas and draw new image
      streamingContext.clearRect(0, 0, streamingCanvas.width, streamingCanvas.height);
      streamingContext.drawImage(img, 0, 0, streamingCanvas.width, streamingCanvas.height);
      
      // Schedule next update immediately
      if (isUsingServerFeed) {
        requestAnimationFrame(updateCanvasStream);
      }
    };
    
    img.onerror = () => {
      // Retry on error
      setTimeout(updateCanvasStream, 50);
    };
    
    // Load new frame with timestamp
    const timestamp = performance.now();
    img.src = `/video_feed?t=${timestamp}`;
    
  } catch (error) {
    console.error('Canvas stream error:', error);
    setTimeout(updateCanvasStream, 100);
  }
}

// Update camera status in UI
function updateCameraStatus(status) {
  const statusElements = document.querySelectorAll('.camera-label, .camera-status');
  statusElements.forEach(element => {
    if (element) {
      element.textContent = status;
      element.style.color = status.includes('Error') || status.includes('Failed') ? '#ff0000' : '#00ff00';
    }
  });
  
  console.log('📊 Camera status:', status);
}

// Capture image - works with both methods
async function captureImage() {
  const resultText = document.querySelector('.result-text p');
  
  try {
    if (resultText) {
      resultText.textContent = 'Capturing real-time image...';
      resultText.style.color = 'orange';
    }
    
    if (isUsingWebRTC) {
      console.log('📸 Capturing from WebRTC (real-time)...');
      await captureWebRTCImage();
    } else {
      console.log('📸 Capturing from server...');
      await captureServerFrame();
    }
    
  } catch (error) {
    console.error('Capture error:', error);
    if (resultText) {
      resultText.textContent = `Capture failed: ${error.message}`;
      resultText.style.color = 'red';
    }
  }
}

// Capture from WebRTC video stream
async function captureWebRTCImage() {
  const video = document.getElementById('camera-feed');
  if (!video || !video.srcObject) {
    throw new Error('No WebRTC video stream available');
  }

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  
  // Set canvas to match video dimensions
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  
  // Draw current video frame
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Convert to JPEG with high quality
  const imageData = canvas.toDataURL('image/jpeg', 0.95);
  
  // Send to server
  const response = await fetch('/upload-image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageData
    })
  });
  
  const result = await response.json();
  
  if (result.success) {
    const resultText = document.querySelector('.result-text p');
    if (resultText) {
      resultText.textContent = `✅ Real-time capture! ${result.filename}`;
      resultText.style.color = 'green';
    }
    console.log('✅ WebRTC capture successful:', result.filename);
  } else {
    throw new Error(result.error);
  }
}

// Capture from server (fallback)
async function captureServerFrame() {
  const response = await fetch('/capture-current-frame', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    }
  });
  
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  
  const result = await response.json();
  
  if (result.success) {
    const resultText = document.querySelector('.result-text p');
    if (resultText) {
      resultText.textContent = `✅ Server capture! ${result.filename}`;
      resultText.style.color = 'green';
    }
    console.log('✅ Server capture successful:', result.filename);
  } else {
    throw new Error(result.error);
  }
}

// Start camera immediately when page loads
window.addEventListener('load', function() {
  console.log('🚀 Starting real-time camera system...');
  startCamera();
});

// Add capture button event listener
document.addEventListener('DOMContentLoaded', function() {
  const captureBtn = document.getElementById('capture-btn');
  if (captureBtn) {
    captureBtn.addEventListener('click', captureImage);
    console.log('✅ Real-time capture button ready');
  }
  
  console.log('🎥 Real-time camera system initialized');
});

// Add camera switching function for testing
window.switchCameraMode = async function() {
  if (isUsingWebRTC) {
    // Switch to server feed
    const video = document.getElementById('camera-feed');
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
    }
    await useOptimizedServerFeed();
  } else {
    // Switch to WebRTC
    await useWebRTCDirect();
  }
};
