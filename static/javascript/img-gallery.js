// Add event listener for the "VIEW PREVIOUS IMAGES" button
document.addEventListener('DOMContentLoaded', function() {
    const viewImagesBtn = document.getElementById('view-images-btn');
    
    if (viewImagesBtn) {
        viewImagesBtn.addEventListener('click', function() {
            // Redirect to image gallery page
            window.location.href = '../templates/result.html'; // Adjust path as needed
        });
    }
});

function goToGallery() {
    window.location.href = '../templates/result.html'; // Adjust path as needed
}