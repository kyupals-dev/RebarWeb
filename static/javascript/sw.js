// Simple service worker for caching
const CACHE_NAME = 'camera-app-v1';
const urlsToCache = [
  '/',
  '/static/css/mainpage.css',
  '/static/css/welcome.css',
  '/static/javascript/camera.js'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});