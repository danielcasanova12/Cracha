// Face Detection with MediaPipe - Non-module version
let faceDetector;
let runningMode = "IMAGE";
let currentDetections = []; // Store current detections for cropping

// Initialize the face detector
const initializeFaceDetector = async () => {
    try {
        // Import MediaPipe dynamically
        const { FaceDetector, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0");
        
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            runningMode: runningMode
        });
        console.log("Face detector initialized successfully");
        
        // Enable the detect button if it exists
        const detectBtn = document.getElementById('detectFacesBtn');
        if (detectBtn) {
            detectBtn.disabled = false;
            detectBtn.innerHTML = '<i class="fas fa-search"></i> Detectar Rostos';
        }
    } catch (error) {
        console.error("Error initializing face detector:", error);
        const detectBtn = document.getElementById('detectFacesBtn');
        if (detectBtn) {
            detectBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Erro ao carregar detector';
            detectBtn.disabled = true;
        }
    }
};

// Function to detect faces - Global scope
window.detectFaces = async function() {
    const image = document.getElementById('uploadedImage');
    const detectBtn = document.getElementById('detectFacesBtn');
    const clearBtn = document.getElementById('clearDetectionBtn');
    const statusDiv = document.getElementById('detectionStatus');
    
    if (!image) {
        alert('Nenhuma imagem encontrada para detectar rostos.');
        return;
    }

    if (!faceDetector) {
        statusDiv.innerHTML = '<div class="alert alert-warning">Aguarde o detector carregar antes de tentar detectar rostos...</div>';
        return;
    }

    // Clear previous detections
    window.clearDetections();

    // Show loading status
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detectando...';
    statusDiv.innerHTML = '<div class="alert alert-info">Analisando imagem...</div>';

    try {
        // Ensure we're in image mode
        if (runningMode === "VIDEO") {
            runningMode = "IMAGE";
            await faceDetector.setOptions({ runningMode: "IMAGE" });
        }

        // Detect faces
        const detections = faceDetector.detect(image).detections;
        console.log('Detections found:', detections);

        if (detections.length > 0) {
            currentDetections = detections; // Store detections for cropping
            displayImageDetections(detections, image);
            statusDiv.innerHTML = `<div class="alert alert-success">Detectados ${detections.length} rosto(s) na imagem!</div>`;
            clearBtn.style.display = 'inline-block';
            
            // Show crop buttons
            const cropBtn = document.getElementById('cropFaceBtn');
            const cropRoundBtn = document.getElementById('cropRoundBtn');
            if (cropBtn) {
                cropBtn.style.display = 'inline-block';
            }
            if (cropRoundBtn) {
                cropRoundBtn.style.display = 'inline-block';
            }
        } else {
            currentDetections = []; // Clear detections
            statusDiv.innerHTML = '<div class="alert alert-warning">Nenhum rosto detectado na imagem.</div>';
        }

    } catch (error) {
        console.error('Error during face detection:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao detectar rostos. Tente novamente.</div>';
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = '<i class="fas fa-search"></i> Detectar Rostos';
    }
};

// Function to clear detections - Global scope
window.clearDetections = function() {
    const container = document.querySelector('.image-container');
    if (!container) return;

    // Clear stored detections
    currentDetections = [];

    // Remove all detection elements
    const highlighters = container.querySelectorAll('.highlighter');
    const infos = container.querySelectorAll('.info');
    const keyPoints = container.querySelectorAll('.key-point');

    highlighters.forEach(el => el.remove());
    infos.forEach(el => el.remove());
    keyPoints.forEach(el => el.remove());

    // Hide buttons
    const clearBtn = document.getElementById('clearDetectionBtn');
    const cropBtn = document.getElementById('cropFaceBtn');
    const cropRoundBtn = document.getElementById('cropRoundBtn');
    if (clearBtn) {
        clearBtn.style.display = 'none';
    }
    if (cropBtn) {
        cropBtn.style.display = 'none';
    }
    if (cropRoundBtn) {
        cropRoundBtn.style.display = 'none';
    }

    // Hide cropped image containers
    const croppedContainer = document.getElementById('croppedImageContainer');
    const roundCroppedContainer = document.getElementById('roundCroppedContainer');
    if (croppedContainer) {
        croppedContainer.style.display = 'none';
    }
    if (roundCroppedContainer) {
        roundCroppedContainer.style.display = 'none';
    }

    // Clear status
    const statusDiv = document.getElementById('detectionStatus');
    if (statusDiv) {
        statusDiv.innerHTML = '';
    }
};

function displayImageDetections(detections, resultElement) {
    const container = resultElement.parentNode;
    const ratio = resultElement.height / resultElement.naturalHeight;

    detections.forEach((detection, index) => {
        // Create confidence text
        const confidence = Math.round(parseFloat(detection.categories[0].score) * 100);
        const infoEl = document.createElement("div");
        infoEl.className = "info";
        infoEl.innerHTML = `<span class="badge bg-primary">Rosto ${index + 1}: ${confidence}%</span>`;
        infoEl.style.cssText = `
            position: absolute;
            left: ${detection.boundingBox.originX * ratio}px;
            top: ${(detection.boundingBox.originY * ratio) - 35}px;
            z-index: 10;
        `;

        // Create bounding box
        const highlighter = document.createElement("div");
        highlighter.className = "highlighter";
        highlighter.style.cssText = `
            position: absolute;
            left: ${detection.boundingBox.originX * ratio}px;
            top: ${detection.boundingBox.originY * ratio}px;
            width: ${detection.boundingBox.width * ratio}px;
            height: ${detection.boundingBox.height * ratio}px;
            border: 3px solid #00ff00;
            border-radius: 5px;
            background: rgba(0, 255, 0, 0.1);
            z-index: 5;
        `;

        container.appendChild(highlighter);
        container.appendChild(infoEl);

        // Add keypoints if available
        if (detection.keypoints) {
            detection.keypoints.forEach(keypoint => {
                const keypointEl = document.createElement("div");
                keypointEl.className = "key-point";
                keypointEl.style.cssText = `
                    position: absolute;
                    top: ${(keypoint.y * resultElement.height) - 3}px;
                    left: ${(keypoint.x * resultElement.width) - 3}px;
                    width: 6px;
                    height: 6px;
                    background: #ff0000;
                    border-radius: 50%;
                    z-index: 15;
                `;
                container.appendChild(keypointEl);
            });
        }
    });
}

// Function to crop face image in 4:3 ratio - Global scope
window.cropFaceImage = function() {
    const image = document.getElementById('uploadedImage');
    const cropBtn = document.getElementById('cropFaceBtn');
    const statusDiv = document.getElementById('detectionStatus');
    
    if (!image || currentDetections.length === 0) {
        alert('Nenhuma detecção de rosto disponível para recorte.');
        return;
    }

    cropBtn.disabled = true;
    cropBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Recortando...';

    try {
        // Use the first detected face (you could modify this to let user choose)
        const detection = currentDetections[0];
        const boundingBox = detection.boundingBox;
        
        // Calculate face center
        const faceCenterX = boundingBox.originX + (boundingBox.width / 2);
        const faceCenterY = boundingBox.originY + (boundingBox.height / 2);
        
        // Calculate 4:3 crop dimensions
        // We'll make the crop area larger than the face for better framing
        const faceSize = Math.max(boundingBox.width, boundingBox.height);
        const cropPadding = faceSize * 0.8; // Add 80% padding around the face
        
        // Calculate crop dimensions maintaining 4:3 ratio
        let cropWidth = faceSize + cropPadding;
        let cropHeight = cropWidth * 0.75; // 4:3 ratio (3/4 = 0.75)
        
        // Ensure crop doesn't exceed image boundaries
        const imageWidth = image.naturalWidth;
        const imageHeight = image.naturalHeight;
        
        // Adjust crop size if it exceeds image boundaries
        if (cropWidth > imageWidth) {
            cropWidth = imageWidth;
            cropHeight = cropWidth * 0.75;
        }
        if (cropHeight > imageHeight) {
            cropHeight = imageHeight;
            cropWidth = cropHeight / 0.75;
        }
        
        // Calculate crop position (centered on face)
        let cropX = faceCenterX - (cropWidth / 2);
        let cropY = faceCenterY - (cropHeight / 2);
        
        // Adjust position to stay within image boundaries
        cropX = Math.max(0, Math.min(cropX, imageWidth - cropWidth));
        cropY = Math.max(0, Math.min(cropY, imageHeight - cropHeight));
        
        // Create canvas for cropping
        const canvas = document.getElementById('croppedCanvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions
        canvas.width = cropWidth;
        canvas.height = cropHeight;
        
        // Draw cropped image
        ctx.drawImage(
            image,
            cropX, cropY, cropWidth, cropHeight,  // Source rectangle
            0, 0, cropWidth, cropHeight           // Destination rectangle
        );
        
        // Show cropped image container
        const croppedContainer = document.getElementById('croppedImageContainer');
        if (croppedContainer) {
            croppedContainer.style.display = 'block';
        }
        
        statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check"></i> Imagem recortada com sucesso! Proporção 4:3 mantida.</div>';
        
    } catch (error) {
        console.error('Error cropping image:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao recortar imagem. Tente novamente.</div>';
    } finally {
        cropBtn.disabled = false;
        cropBtn.innerHTML = '<i class="fas fa-crop"></i> Recortar Rosto 4:3';
    }
};

// Function to download cropped image - Global scope
window.downloadCroppedImage = function() {
    const canvas = document.getElementById('croppedCanvas');
    if (!canvas) {
        alert('Nenhuma imagem recortada disponível para download.');
        return;
    }
    
    // Create download link
    const link = document.createElement('a');
    link.download = `rosto-recortado-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

// Function to crop round face image - Global scope
window.cropRoundFace = function() {
    const image = document.getElementById('uploadedImage');
    const cropRoundBtn = document.getElementById('cropRoundBtn');
    const statusDiv = document.getElementById('detectionStatus');
    
    if (!image || currentDetections.length === 0) {
        alert('Nenhuma detecção de rosto disponível para recorte.');
        return;
    }

    cropRoundBtn.disabled = true;
    cropRoundBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Recortando...';

    try {
        // Use the first detected face
        const detection = currentDetections[0];
        const boundingBox = detection.boundingBox;
        
        // Calculate face center
        const faceCenterX = boundingBox.originX + (boundingBox.width / 2);
        const faceCenterY = boundingBox.originY + (boundingBox.height / 2);
        
        // Calculate circle radius - use the larger dimension of the face + padding
        const faceSize = Math.max(boundingBox.width, boundingBox.height);
        const padding = faceSize * 1.2; // More generous padding for round crop
        const radius = (faceSize + padding) / 2;
        
        // Ensure circle doesn't exceed image boundaries
        const imageWidth = image.naturalWidth;
        const imageHeight = image.naturalHeight;
        
        // Adjust radius if needed
        const maxRadiusX = Math.min(faceCenterX, imageWidth - faceCenterX);
        const maxRadiusY = Math.min(faceCenterY, imageHeight - faceCenterY);
        const maxRadius = Math.min(maxRadiusX, maxRadiusY);
        const finalRadius = Math.min(radius, maxRadius);
        
        // Calculate crop area (square that contains the circle)
        const cropSize = finalRadius * 2;
        const cropX = faceCenterX - finalRadius;
        const cropY = faceCenterY - finalRadius;
        
        // Create canvas for round cropping
        const canvas = document.getElementById('roundCroppedCanvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions (square)
        canvas.width = cropSize;
        canvas.height = cropSize;
        
        // Clear canvas
        ctx.clearRect(0, 0, cropSize, cropSize);
        
        // Create circular clipping mask
        ctx.save();
        ctx.beginPath();
        ctx.arc(finalRadius, finalRadius, finalRadius, 0, 2 * Math.PI);
        ctx.closePath();
        ctx.clip();
        
        // Draw the image within the circular mask
        ctx.drawImage(
            image,
            cropX, cropY, cropSize, cropSize,  // Source rectangle
            0, 0, cropSize, cropSize           // Destination rectangle
        );
        
        ctx.restore();
        
        // Add a subtle border around the circle
        ctx.beginPath();
        ctx.arc(finalRadius, finalRadius, finalRadius - 2, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 4;
        ctx.stroke();
        
        // Show round cropped image container
        const roundCroppedContainer = document.getElementById('roundCroppedContainer');
        if (roundCroppedContainer) {
            roundCroppedContainer.style.display = 'block';
        }
        
        statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Imagem redonda recortada com sucesso!</div>';
        
    } catch (error) {
        console.error('Error cropping round image:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao recortar imagem redonda. Tente novamente.</div>';
    } finally {
        cropRoundBtn.disabled = false;
        cropRoundBtn.innerHTML = '<i class="fas fa-circle"></i> Recortar Redondo';
    }
};

// Function to download round cropped image - Global scope
window.downloadRoundCroppedImage = function() {
    const canvas = document.getElementById('roundCroppedCanvas');
    if (!canvas) {
        alert('Nenhuma imagem redonda recortada disponível para download.');
        return;
    }
    
    // Create download link
    const link = document.createElement('a');
    link.download = `rosto-redondo-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

// Preview function for file input - Global scope
window.previewImage = function(input) {
    const preview = document.getElementById('preview');
    const previewContainer = document.getElementById('imagePreview');
    
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
        }
        
        reader.readAsDataURL(input.files[0]);
    } else {
        previewContainer.style.display = 'none';
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing face detection...');
    initializeFaceDetector();
});
