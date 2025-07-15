const videoElement = document.getElementById('video');
const toggleCameraButton = document.getElementById('toggle-camera');

let cameraOn = false;

toggleCameraButton.addEventListener('click', () => {
  if (!cameraOn) {
    startCamera();
    toggleCameraButton.textContent = 'Toggle Camera';
  } else {
    stopCamera();
    toggleCameraButton.textContent = 'Toggle Camera';
  }
});

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    cameraOn = true;
  } catch (error) {
    console.error('Error accessing camera:', error);
  }
}

function stopCamera() {
  const stream = videoElement.srcObject;
  const tracks = stream.getTracks();

  tracks.forEach(track => track.stop());

  videoElement.srcObject = null;
  cameraOn = false;
}
