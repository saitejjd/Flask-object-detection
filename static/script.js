let cameraEnabled = false;
const videoFeed = document.getElementById('videoFeed');
const cameraPanel = document.getElementById('cameraPanel');


async function toggleCamera() {
    try {
        if (!cameraEnabled) {
            // Start camera
            await fetch('/start_camera');
            document.getElementById('cameraPanel').style.display = 'block';
            document.getElementById('videoFeed').src = '/video_feed';
            cameraEnabled = true;
        } else {
            // Stop camera
            document.getElementById('cameraPanel').style.display = 'none';
            document.getElementById('videoFeed').src = '';
            await fetch('/stop_camera');
            cameraEnabled = false;
        }
    } catch (error) {
        console.error('Camera error:', error);
        alert('Camera error: ' + error.message);
    }
}

function closeCamera() {
    cameraPanel.style.display = 'none';
    videoFeed.src = '';
    if (cameraEnabled) {
        fetch('/stop_camera').then(() => cameraEnabled = false);
    }
}

window.addEventListener('beforeunload', () => {
    if (cameraEnabled) {
        fetch('/stop_camera');
    }
});