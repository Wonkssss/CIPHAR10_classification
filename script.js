const video = document.getElementById('webcam');
const switchCamBtn = document.getElementById('SwitchCam');
const predictionText = document.getElementById('predictedClass');
let currentStream;
let model;

const class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

// Initialize the plot
const initialData = [{
    y: Array(10).fill(0),  
    type: 'bar'           
}];
Plotly.newPlot('ClassGraph', initialData);


// Initialize ONNX model
async function initModel() {
    model = await ort.InferenceSession.create('model.onnx');
}

// Get webcam access
async function getWebcam(streamName = 'user') {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: streamName
        }
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;

    video.onloadedmetadata = function() {
        video.play();
        inferenceLoop();
    };
}

function preprocessFrame() {
    // Get video element
    const video = document.getElementById('webcam');

    // Create a temporary canvas to capture a frame from the video
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0);

    // Create an off-screen canvas for resizing
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = 32;
    resizeCanvas.height = 32;
    const resizeCtx = resizeCanvas.getContext('2d');

    resizeCtx.drawImage(captureCanvas, video.videoWidth / 2 - 140, video.videoHeight / 2 - 140, 320, 320, 0, 0, 32, 32);

    // Get resized image data
    const resizedImageData = resizeCtx.getImageData(0, 0, 32, 32);
    

    const redArray = [];
    const greenArray = [];
    const blueArray = [];

    // Remove alpha channel and normalize
    for (let i = 0; i < resizedImageData.data.length; i += 4) {
        redArray.push((resizedImageData.data[i] / 255 - 0.4914) / 0.2023);
        greenArray.push((resizedImageData.data[i + 1] / 255 - 0.4822) / 0.1994);
        blueArray.push((resizedImageData.data[i + 2] / 255 - 0.4465) / 0.2010);
        // Skip data[i + 3] to filter out the alpha channel
    }

    // Concatenate RGB arrays
    const input = new Float32Array([...redArray, ...greenArray, ...blueArray]);

    // Display the grayscaled and resized image on the actual canvas in the HTML
    const displayCanvas = document.getElementById('preview');
    const displayCtx = displayCanvas.getContext('2d');
    displayCtx.putImageData(resizedImageData, 0, 0);
    const output_tensor = new ort.Tensor('float32', input,  [1, 3, 32, 32]);
    return output_tensor;
}

// Inference loop
async function inferenceLoop() {

    const tensorInput = preprocessFrame();
    const res = await model.run({'input': tensorInput});
    output = res.output.data; // ‼️
    const max_index = output.indexOf(Math.max(...output));
    predictionText.innerHTML = class_names[max_index];


    // Process logits & plot
    const updatedData = [{ y: output , type: 'bar'}];
    Plotly.react('ClassGraph', updatedData);

    const now = Date.now();
    const fps = 1000 / (now - (window.lastInferenceTime || now));
    window.lastInferenceTime = now;

    requestAnimationFrame(inferenceLoop);
}

switchCamBtn.addEventListener('click', () => {
    const facingMode = video.srcObject.getVideoTracks()[0].getSettings().facingMode;
    getWebcam(facingMode === 'user' ? 'environment' : 'user');
});

initModel().then(() => {
    getWebcam();
});