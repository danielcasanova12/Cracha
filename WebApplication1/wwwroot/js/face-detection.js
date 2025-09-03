// Face Detection with MediaPipe - Non-module version
let faceDetector;
let imageSegmenter;
let biRefNetSession; // ONNX Runtime session for BiRefNet
let runningMode = "IMAGE";
let currentDetections = []; // Store current detections for cropping

// Initialize the face detector
const initializeFaceDetector = async () => {
    try {
        // Import MediaPipe dynamically
        const { FaceDetector, ImageSegmenter, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0");

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        // Initialize Face Detector
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            runningMode: runningMode
        });

        // Initialize Image Segmenter for background removal
        imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
                delegate: "GPU"
            },
            runningMode: runningMode,
            outputCategoryMask: true,
            outputConfidenceMasks: false
        });

        console.log("Face detector and Image segmenter initialized successfully");

        // Initialize BiRefNet ONNX model
        await initializeBiRefNet();

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

// Initialize BiRefNet ONNX model
const initializeBiRefNet = async () => {
    try {
        // Load ONNX Runtime Web from CDN
        if (typeof ort === 'undefined') {
            console.log("Loading ONNX Runtime...");
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
            document.head.appendChild(script);
            await new Promise((resolve, reject) => {
                script.onload = resolve;
                script.onerror = reject;
            });
            console.log("ONNX Runtime loaded successfully");
        }

        // Configure ONNX Runtime
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

        // Fetch model directly
        console.log("Fetching BiRefNet model...");
        const modelUrl = `/models/BiRefNet-portrait-epoch_150.onnx?v=${Date.now()}`;
        const response = await fetch(modelUrl);

        if (!response.ok) {
            console.error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            console.error(`URL attempted: ${modelUrl}`);
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }

        console.log("Model response received, loading as ArrayBuffer...");
        const modelArrayBuffer = await response.arrayBuffer();
        console.log("Model fetched successfully, size:", modelArrayBuffer.byteLength);

        // Create ONNX Runtime session from ArrayBuffer
        console.log("Creating ONNX session...");
        biRefNetSession = await ort.InferenceSession.create(modelArrayBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        // Log input and output names for debugging
        console.log("BiRefNet model loaded successfully");
        console.log("Input names:", biRefNetSession.inputNames);
        console.log("Output names:", biRefNetSession.outputNames);

    } catch (error) {
        console.error("Error loading BiRefNet model:", error);
        console.error("Stack trace:", error.stack);
        biRefNetSession = null;
    }
};

// Function to detect faces - Global scope
window.detectFaces = async function () {
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
            const removeBackgroundBtn = document.getElementById('removeBackgroundBtn');
            const removeBackgroundRoundBtn = document.getElementById('removeBackgroundRoundBtn');
            const removeBackgroundBiRefNetBtn = document.getElementById('removeBackgroundBiRefNetBtn');
            const removeBackgroundRMBGBtn = document.getElementById('removeBackgroundRMBGBtn');
            const removeBackgroundMODNetBtn = document.getElementById('removeBackgroundMODNetBtn');
            const removeBackgroundMODNetRoundBtn = document.getElementById('removeBackgroundMODNetRoundBtn');
            if (cropBtn) {
                cropBtn.style.display = 'inline-block';
            }
            if (cropRoundBtn) {
                cropRoundBtn.style.display = 'inline-block';
            }
            if (removeBackgroundBtn) {
                removeBackgroundBtn.style.display = 'inline-block';
            }
            if (removeBackgroundRoundBtn) {
                removeBackgroundRoundBtn.style.display = 'inline-block';
            }
            if (removeBackgroundBiRefNetBtn) {
                removeBackgroundBiRefNetBtn.style.display = 'inline-block';
            }
            if (removeBackgroundRMBGBtn) {
                removeBackgroundRMBGBtn.style.display = 'inline-block';
            }
            if (removeBackgroundMODNetBtn) {
                removeBackgroundMODNetBtn.style.display = 'inline-block';
            }
            if (removeBackgroundMODNetRoundBtn) {
                removeBackgroundMODNetRoundBtn.style.display = 'inline-block';
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
window.clearDetections = function () {
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
    const removeBackgroundBtn = document.getElementById('removeBackgroundBtn');
    const removeBackgroundRoundBtn = document.getElementById('removeBackgroundRoundBtn');
    const removeBackgroundBiRefNetBtn = document.getElementById('removeBackgroundBiRefNetBtn');
    const removeBackgroundRMBGBtn = document.getElementById('removeBackgroundRMBGBtn');
    const removeBackgroundMODNetBtn = document.getElementById('removeBackgroundMODNetBtn');
    const removeBackgroundMODNetRoundBtn = document.getElementById('removeBackgroundMODNetRoundBtn');
    if (clearBtn) {
        clearBtn.style.display = 'none';
    }
    if (cropBtn) {
        cropBtn.style.display = 'none';
    }
    if (cropRoundBtn) {
        cropRoundBtn.style.display = 'none';
    }
    if (removeBackgroundBtn) {
        removeBackgroundBtn.style.display = 'none';
    }
    if (removeBackgroundRoundBtn) {
        removeBackgroundRoundBtn.style.display = 'none';
    }
    if (removeBackgroundBiRefNetBtn) {
        removeBackgroundBiRefNetBtn.style.display = 'none';
    }
    if (removeBackgroundRMBGBtn) {
        removeBackgroundRMBGBtn.style.display = 'none';
    }
    if (removeBackgroundMODNetBtn) {
        removeBackgroundMODNetBtn.style.display = 'none';
    }
    if (removeBackgroundMODNetRoundBtn) {
        removeBackgroundMODNetRoundBtn.style.display = 'none';
    }
    
    // Esconder botão do crachá
    hideBadgeButton();

    // Hide cropped image containers
    const croppedContainer = document.getElementById('croppedImageContainer');
    const roundCroppedContainer = document.getElementById('roundCroppedContainer');
    const noBackgroundContainer = document.getElementById('noBackgroundContainer');
    const combinedContainer = document.getElementById('combinedContainer');
    const biRefNetContainer = document.getElementById('biRefNetContainer');
    const rmbgContainer = document.getElementById('rmbgContainer');
    const modnetContainer = document.getElementById('modnetContainer');
    const modnetRoundContainer = document.getElementById('modnetRoundContainer');
    if (croppedContainer) {
        croppedContainer.style.display = 'none';
    }
    if (roundCroppedContainer) {
        roundCroppedContainer.style.display = 'none';
    }
    if (noBackgroundContainer) {
        noBackgroundContainer.style.display = 'none';
    }
    if (combinedContainer) {
        combinedContainer.style.display = 'none';
    }
    if (biRefNetContainer) {
        biRefNetContainer.style.display = 'none';
    }
    if (rmbgContainer) {
        rmbgContainer.style.display = 'none';
    }
    if (modnetContainer) {
        modnetContainer.style.display = 'none';
    }
    if (modnetRoundContainer) {
        modnetRoundContainer.style.display = 'none';
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
window.cropFaceImage = function () {
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
        
        // Mostrar botões do crachá após processamento bem-sucedido
        showBadgeButton();
        showSpecificBadgeButtons();

    } catch (error) {
        console.error('Error cropping image:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao recortar imagem. Tente novamente.</div>';
    } finally {
        cropBtn.disabled = false;
        cropBtn.innerHTML = '<i class="fas fa-crop"></i> Recortar Rosto 4:3';
    }
};

// Function to download cropped image - Global scope
window.downloadCroppedImage = function () {
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
window.cropRoundFace = function () {
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
        
        // Mostrar botões específicos do crachá após processamento bem-sucedido
        showBadgeButton();
        showSpecificBadgeButtons();

    } catch (error) {
        console.error('Error cropping round image:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao recortar imagem redonda. Tente novamente.</div>';
    } finally {
        cropRoundBtn.disabled = false;
        cropRoundBtn.innerHTML = '<i class="fas fa-circle"></i> Recortar Redondo';
    }
};

// Function to download round cropped image - Global scope
window.downloadRoundCroppedImage = function () {
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

// Function to remove background using image segmentation - Global scope
window.removeBackground = function () {
    const image = document.getElementById('uploadedImage');
    const removeBackgroundBtn = document.getElementById('removeBackgroundBtn');
    const statusDiv = document.getElementById('detectionStatus');

    if (!image) {
        alert('Nenhuma imagem encontrada para remover fundo.');
        return;
    }

    if (!imageSegmenter) {
        statusDiv.innerHTML = '<div class="alert alert-warning">Aguarde o segmentador carregar antes de tentar remover o fundo...</div>';
        return;
    }

    removeBackgroundBtn.disabled = true;
    removeBackgroundBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Removendo Fundo...';
    statusDiv.innerHTML = '<div class="alert alert-info">Processando segmentação da imagem...</div>';

    try {
        // Ensure we're in image mode
        if (runningMode === "VIDEO") {
            runningMode = "IMAGE";
            imageSegmenter.setOptions({ runningMode: "IMAGE" });
        }

        // Segment the image
        const segmentationResult = imageSegmenter.segment(image);

        if (segmentationResult && segmentationResult.categoryMask) {
            processSegmentationResult(segmentationResult, image);
            statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check"></i> Fundo removido com sucesso!</div>';
            
            // Mostrar botão do crachá após processamento bem-sucedido
            showBadgeButton();
        } else {
            statusDiv.innerHTML = '<div class="alert alert-warning">Não foi possível segmentar a imagem.</div>';
        }

    } catch (error) {
        console.error('Error removing background:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro ao remover fundo. Tente novamente.</div>';
    } finally {
        removeBackgroundBtn.disabled = false;
        removeBackgroundBtn.innerHTML = '<i class="fas fa-magic"></i> Remover Fundo';
    }
};

// Process segmentation result to remove background
function processSegmentationResult(result, originalImage) {
    const canvas = document.getElementById('noBackgroundCanvas');
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions to match the original image
    canvas.width = originalImage.naturalWidth;
    canvas.height = originalImage.naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the original image
    ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);

    // Get image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Get segmentation mask
    const mask = result.categoryMask.getAsUint8Array();
    const { width, height } = result.categoryMask;

    // Process each pixel
    for (let i = 0; i < mask.length; i++) {
        const maskValue = mask[i];

        // If mask value is 0 (background) or not person (15 is person class in DeepLab)
        // Make the pixel transparent
        if (maskValue === 0 || maskValue !== 15) {
            const pixelIndex = i * 4;
            if (pixelIndex < data.length) {
                data[pixelIndex + 3] = 0; // Set alpha to 0 (transparent)
            }
        }
    }

    // Put the modified image data back to canvas
    ctx.putImageData(imageData, 0, 0);

    // Show the no background container
    const noBackgroundContainer = document.getElementById('noBackgroundContainer');
    if (noBackgroundContainer) {
        noBackgroundContainer.style.display = 'block';
    }
}

// Function to download image without background - Global scope
window.downloadNoBackgroundImage = function () {
    const canvas = document.getElementById('noBackgroundCanvas');
    if (!canvas) {
        alert('Nenhuma imagem sem fundo disponível para download.');
        return;
    }

    // Create download link
    const link = document.createElement('a');
    link.download = `sem-fundo-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

// Function to remove background and crop round face - Global scope
window.removeBackgroundAndCropRound = function () {
    if (!imageSegmenter) {
        alert('Detector de imagem não inicializado ainda. Aguarde um momento.');
        return;
    }

    if (currentDetections.length === 0) {
        alert('Nenhum rosto detectado. Execute a detecção primeiro.');
        return;
    }

    const img = document.getElementById('uploadedImage');
    if (!img) {
        alert('Nenhuma imagem carregada.');
        return;
    }

    const statusDiv = document.getElementById('detectionStatus');
    const removeBackgroundRoundBtn = document.getElementById('removeBackgroundRoundBtn');

    removeBackgroundRoundBtn.disabled = true;
    removeBackgroundRoundBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando...';

    try {
        statusDiv.innerHTML = '<div class="alert alert-info">Removendo fundo e criando recorte redondo...</div>';

        // Process the image for segmentation
        const segmentationResults = imageSegmenter.segment(img);

        if (segmentationResults && segmentationResults.categoryMask) {
            // Get the first detected face
            const detection = currentDetections[0];
            const faceBox = detection.boundingBox;

            // Process the combined result
            processCombinedResult(segmentationResults, img, faceBox);

            statusDiv.innerHTML = '<div class="alert alert-success">Processamento concluído!</div>';
            
            // Mostrar botão do crachá após processamento bem-sucedido
            showBadgeButton();
        } else {
            throw new Error('Falha na segmentação da imagem');
        }

    } catch (error) {
        console.error('Error in combined processing:', error);
        statusDiv.innerHTML = '<div class="alert alert-danger">Erro no processamento. Tente novamente.</div>';
    } finally {
        removeBackgroundRoundBtn.disabled = false;
        removeBackgroundRoundBtn.innerHTML = '<i class="fas fa-user-circle"></i> Fundo + Recorte';
    }
};

// Process combined result (remove background + round crop with face centered)
function processCombinedResult(segmentationResult, originalImage, faceBox) {
    const canvas = document.getElementById('combinedCanvas');
    const ctx = canvas.getContext('2d');

    // Calculate face center
    const faceCenterX = faceBox.originX + faceBox.width / 2;
    const faceCenterY = faceBox.originY + faceBox.height / 2;

    // Calculate crop size (larger than face for better composition)
    const cropSize = Math.max(faceBox.width, faceBox.height) * 2.5;

    // Set canvas dimensions to square
    const finalSize = 400;
    canvas.width = finalSize;
    canvas.height = finalSize;

    // Clear canvas with transparency
    ctx.clearRect(0, 0, finalSize, finalSize);

    // Create a temporary canvas for background removal
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = originalImage.naturalWidth;
    tempCanvas.height = originalImage.naturalHeight;

    // Draw original image on temp canvas
    tempCtx.drawImage(originalImage, 0, 0, tempCanvas.width, tempCanvas.height);

    // Get image data for background removal
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;

    // Get segmentation mask
    const mask = segmentationResult.categoryMask.getAsUint8Array();

    // Remove background (keep only person - category 15)
    for (let i = 0; i < mask.length; i++) {
        const pixelIndex = i * 4;
        if (mask[i] !== 15) { // Not a person
            data[pixelIndex + 3] = 0; // Make transparent
        }
    }

    // Put the processed image data back
    tempCtx.putImageData(imageData, 0, 0);

    // Calculate crop area with face centered
    const scale = originalImage.naturalWidth / originalImage.width;
    const scaledFaceCenterX = faceCenterX * scale;
    const scaledFaceCenterY = faceCenterY * scale;
    const scaledCropSize = cropSize * scale;

    const cropX = Math.max(0, scaledFaceCenterX - scaledCropSize / 2);
    const cropY = Math.max(0, scaledFaceCenterY - scaledCropSize / 2);
    const cropWidth = Math.min(scaledCropSize, tempCanvas.width - cropX);
    const cropHeight = Math.min(scaledCropSize, tempCanvas.height - cropY);

    // Create circular clipping path
    ctx.save();
    ctx.beginPath();
    ctx.arc(finalSize / 2, finalSize / 2, finalSize / 2, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.clip();

    // Draw the cropped and background-removed image within the circle
    ctx.drawImage(
        tempCanvas,
        cropX, cropY, cropWidth, cropHeight,
        0, 0, finalSize, finalSize
    );

    ctx.restore();

    // Show the combined container
    const combinedContainer = document.getElementById('combinedContainer');
    if (combinedContainer) {
        combinedContainer.style.display = 'block';
    }
}

// Function to download combined image - Global scope
window.downloadCombinedImage = function () {
    const canvas = document.getElementById('combinedCanvas');
    if (!canvas) {
        alert('Nenhuma imagem combinada disponível para download.');
        return;
    }

    // Create download link
    const link = document.createElement('a');
    link.download = `foto-perfil-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

// Function to remove background using BiRefNet - Global scope
window.removeBackgroundBiRefNet = async function () {
    if (!biRefNetSession) {
        alert('Modelo BiRefNet não inicializado ainda. Aguarde um momento.');
        return;
    }

    const img = document.getElementById('uploadedImage');
    if (!img) {
        alert('Nenhuma imagem carregada.');
        return;
    }

    const statusDiv = document.getElementById('detectionStatus');
    const removeBackgroundBiRefNetBtn = document.getElementById('removeBackgroundBiRefNetBtn');

    removeBackgroundBiRefNetBtn.disabled = true;
    removeBackgroundBiRefNetBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando BiRefNet...';

    try {
        statusDiv.innerHTML = '<div class="alert alert-info">Processando com BiRefNet (alta precisão)...</div>';

        // Preprocess image for BiRefNet
        console.log("Starting image preprocessing...");
        const inputTensor = await preprocessImageForBiRefNet(img);
        console.log("Input tensor created:", inputTensor.dims, inputTensor.type);
        console.log("Input data sample:", Array.from(inputTensor.data.slice(0, 10)));

        // Get the correct input name from the model
        const inputName = biRefNetSession.inputNames[0];
        console.log("Using input name:", inputName);

        // Validate tensor before inference
        if (!inputTensor || !inputTensor.data || inputTensor.data.length === 0) {
            throw new Error("Invalid input tensor");
        }

        // Run inference with proper error handling
        console.log("Starting ONNX inference...");
        const feeds = {};
        feeds[inputName] = inputTensor;

        // Try to validate the model before inference
        try {
            console.log("Validating model compatibility...");

            // Create a small test tensor to validate the model
            const testTensor = new ort.Tensor('float32', new Float32Array(3 * 1024 * 1024), [1, 3, 1024, 1024]);
            const testFeeds = {};
            testFeeds[inputName] = testTensor;

            console.log("Running test inference with dummy data...");
            const testResults = await biRefNetSession.run(testFeeds);
            console.log("Test inference successful! Model is compatible.");
            console.log("Test output keys:", Object.keys(testResults));

            // If test passes, run with real data
            console.log("Running inference with real image data...");
            results = await biRefNetSession.run(feeds);

        } catch (testError) {
            console.error("Test inference failed:", testError);

            // Try with different tensor formats
            console.log("Trying alternative tensor format (NHWC instead of NCHW)...");
            try {
                const nhwcTensor = await createNHWCTensor(img);
                feeds[inputName] = nhwcTensor;
                results = await biRefNetSession.run(feeds);
                console.log("NHWC format successful!");
            } catch (nhwcError) {
                console.error("NHWC format also failed:", nhwcError);

                // Final attempt with minimal preprocessing
                console.log("Trying with minimal preprocessing...");
                try {
                    const simpleTensor = await createSimpleTensor(img);
                    feeds[inputName] = simpleTensor;
                    results = await biRefNetSession.run(feeds);
                    console.log("Simple tensor format successful!");
                } catch (simpleError) {
                    console.error("All tensor formats failed:", simpleError);
                    throw new Error(`Model incompatible: ${testError.message || testError}`);
                }
            }
        }

        // Debug: log output tensor info
        console.log("BiRefNet inference completed");
        console.log("Available outputs:", Object.keys(results));

        // Get the output tensor (BiRefNet usually outputs 'output' or the first key)
        const outputKey = Object.keys(results)[0];
        const outputTensor = results[outputKey];
        console.log("Using output key:", outputKey);
        console.log("Output tensor shape:", outputTensor.dims);
        console.log("Output tensor type:", outputTensor.type);
        console.log("Output data length:", outputTensor.data.length);
        console.log("Output data sample (first 10 values):", Array.from(outputTensor.data.slice(0, 10)));

        // Process the output
        console.log("Starting result processing...");
        await processBiRefNetResult(outputTensor, img);
        console.log("Result processing completed");

        statusDiv.innerHTML = '<div class="alert alert-success">Processamento BiRefNet concluído!</div>';

    } catch (error) {
        console.error('Error in BiRefNet processing:');
        console.error('Error type:', typeof error);
        console.error('Error message:', error.message || error);
        console.error('Error stack:', error.stack);
        console.error('Full error object:', error);

        let errorMessage = 'Erro no processamento BiRefNet.';
        if (error.message) {
            errorMessage += ` Detalhes: ${error.message}`;
        }
        statusDiv.innerHTML = `<div class="alert alert-danger">${errorMessage}</div>`;
    } finally {
        removeBackgroundBiRefNetBtn.disabled = false;
        removeBackgroundBiRefNetBtn.innerHTML = '<i class="fas fa-brain"></i> BiRefNet';
    }
};

// Preprocess image for BiRefNet
async function preprocessImageForBiRefNet(img) {
    try {
        console.log("Starting BiRefNet preprocessing...");
        console.log("Original image dimensions:", img.naturalWidth, "x", img.naturalHeight);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // BiRefNet expects 1024x1024 input
        const targetSize = 1024;
        canvas.width = targetSize;
        canvas.height = targetSize;

        // Draw image with padding to maintain aspect ratio
        const scale = Math.min(targetSize / img.naturalWidth, targetSize / img.naturalHeight);
        const scaledWidth = img.naturalWidth * scale;
        const scaledHeight = img.naturalHeight * scale;
        const offsetX = (targetSize - scaledWidth) / 2;
        const offsetY = (targetSize - scaledHeight) / 2;

        console.log("Scaling:", scale, "New size:", scaledWidth, "x", scaledHeight);
        console.log("Offset:", offsetX, offsetY);

        // Fill with gray background (0.5 normalized)
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Draw scaled image
        ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

        // Get image data
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const data = imageData.data;
        console.log("Image data length:", data.length);

        // Convert to tensor format [1, 3, 1024, 1024]
        const tensorSize = 1 * 3 * targetSize * targetSize;
        const tensor = new Float32Array(tensorSize);

        console.log("Creating tensor with size:", tensorSize);

        // Convert RGBA to CHW format with correct indexing - FIXED VERSION
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const pixelIndex = y * targetSize + x;
                const dataIndex = pixelIndex * 4;

                // Simple [0,1] normalization
                const r = data[dataIndex] / 255.0;
                const g = data[dataIndex + 1] / 255.0;
                const b = data[dataIndex + 2] / 255.0;

                // CHW layout: channels first - CORRECTED INDEXING
                tensor[0 * targetSize * targetSize + pixelIndex] = r;  // R channel
                tensor[1 * targetSize * targetSize + pixelIndex] = g;  // G channel  
                tensor[2 * targetSize * targetSize + pixelIndex] = b;  // B channel
            }
        }

        console.log("Tensor created successfully");
        console.log("First few tensor values:", Array.from(tensor.slice(0, 10)));

        // Calculate min/max without spread operator to avoid stack overflow
        let minVal = tensor[0];
        let maxVal = tensor[0];
        for (let i = 1; i < tensor.length; i++) {
            if (tensor[i] < minVal) minVal = tensor[i];
            if (tensor[i] > maxVal) maxVal = tensor[i];
        }
        console.log("Tensor range - Min:", minVal, "Max:", maxVal);

        // Validate tensor
        if (tensor.length !== tensorSize) {
            throw new Error(`Tensor size mismatch: expected ${tensorSize}, got ${tensor.length}`);
        }

        const result = new ort.Tensor('float32', tensor, [1, 3, targetSize, targetSize]);
        console.log("ONNX Tensor created with shape:", result.dims);

        return result;

    } catch (error) {
        console.error("Error in preprocessImageForBiRefNet:", error);
        throw error;
    }
}

// Alternative preprocessing with ImageNet normalization (fallback)
async function preprocessImageForBiRefNetImageNet(img) {
    try {
        console.log("Starting alternative BiRefNet preprocessing with ImageNet normalization...");

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // BiRefNet expects 1024x1024 input
        const targetSize = 1024;
        canvas.width = targetSize;
        canvas.height = targetSize;

        // Draw image with padding to maintain aspect ratio
        const scale = Math.min(targetSize / img.naturalWidth, targetSize / img.naturalHeight);
        const scaledWidth = img.naturalWidth * scale;
        const scaledHeight = img.naturalHeight * scale;
        const offsetX = (targetSize - scaledWidth) / 2;
        const offsetY = (targetSize - scaledHeight) / 2;

        // Fill with gray background
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Draw scaled image
        ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

        // Get image data
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const data = imageData.data;

        // Convert to tensor format [1, 3, 1024, 1024] with ImageNet normalization
        const tensorSize = 1 * 3 * targetSize * targetSize;
        const tensor = new Float32Array(tensorSize);

        console.log("Using ImageNet normalization");

        for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            const y = Math.floor(pixelIndex / targetSize);
            const x = pixelIndex % targetSize;

            // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            const r = (data[i] / 255.0 - 0.485) / 0.229;
            const g = (data[i + 1] / 255.0 - 0.456) / 0.224;
            const b = (data[i + 2] / 255.0 - 0.406) / 0.225;

            // CHW format: [C, H, W]
            const rIndex = 0 * targetSize * targetSize + y * targetSize + x;
            const gIndex = 1 * targetSize * targetSize + y * targetSize + x;
            const bIndex = 2 * targetSize * targetSize + y * targetSize + x;

            tensor[rIndex] = r;
            tensor[gIndex] = g;
            tensor[bIndex] = b;
        }

        // Calculate min/max without spread operator to avoid stack overflow
        let minVal = tensor[0];
        let maxVal = tensor[0];
        for (let i = 1; i < tensor.length; i++) {
            if (tensor[i] < minVal) minVal = tensor[i];
            if (tensor[i] > maxVal) maxVal = tensor[i];
        }
        console.log("Alternative tensor range - Min:", minVal, "Max:", maxVal);

        const result = new ort.Tensor('float32', tensor, [1, 3, targetSize, targetSize]);
        console.log("Alternative ONNX Tensor created with shape:", result.dims);

        return result;

    } catch (error) {
        console.error("Error in alternative preprocessing:", error);
        throw error;
    }
}

// Create NHWC tensor (alternative format)
async function createNHWCTensor(img) {
    console.log("Creating NHWC tensor...");
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const targetSize = 1024;
    canvas.width = targetSize;
    canvas.height = targetSize;

    // Draw image
    const scale = Math.min(targetSize / img.naturalWidth, targetSize / img.naturalHeight);
    const scaledWidth = img.naturalWidth * scale;
    const scaledHeight = img.naturalHeight * scale;
    const offsetX = (targetSize - scaledWidth) / 2;
    const offsetY = (targetSize - scaledHeight) / 2;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetSize, targetSize);
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data;

    // NHWC format: [N, H, W, C]
    const tensor = new Float32Array(1 * targetSize * targetSize * 3);

    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const pixelIndex = y * targetSize + x;
            const dataIndex = pixelIndex * 4;

            const r = data[dataIndex] / 255.0;
            const g = data[dataIndex + 1] / 255.0;
            const b = data[dataIndex + 2] / 255.0;

            const tensorIndex = pixelIndex * 3;
            tensor[tensorIndex] = r;
            tensor[tensorIndex + 1] = g;
            tensor[tensorIndex + 2] = b;
        }
    }

    return new ort.Tensor('float32', tensor, [1, targetSize, targetSize, 3]);
}

// Create simple tensor with minimal processing
async function createSimpleTensor(img) {
    console.log("Creating simple tensor...");
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Try smaller size first
    const targetSize = 512;
    canvas.width = targetSize;
    canvas.height = targetSize;

    // Simple resize without padding
    ctx.drawImage(img, 0, 0, targetSize, targetSize);

    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data;

    // Simple CHW format with basic normalization
    const tensor = new Float32Array(1 * 3 * targetSize * targetSize);

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        const y = Math.floor(pixelIndex / targetSize);
        const x = pixelIndex % targetSize;

        // Simple [0,1] normalization
        const r = data[i] / 255.0;
        const g = data[i + 1] / 255.0;
        const b = data[i + 2] / 255.0;

        tensor[0 * targetSize * targetSize + y * targetSize + x] = r;
        tensor[1 * targetSize * targetSize + y * targetSize + x] = g;
        tensor[2 * targetSize * targetSize + y * targetSize + x] = b;
    }

    return new ort.Tensor('float32', tensor, [1, 3, targetSize, targetSize]);
}

// Create tensor with specific size (adaptive sizing)
async function createTensorWithSize(img, targetSize) {
    console.log(`Creating tensor with size ${targetSize}x${targetSize}...`);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = targetSize;
    canvas.height = targetSize;

    // Draw image with aspect ratio preservation
    const scale = Math.min(targetSize / img.naturalWidth, targetSize / img.naturalHeight);
    const scaledWidth = img.naturalWidth * scale;
    const scaledHeight = img.naturalHeight * scale;
    const offsetX = (targetSize - scaledWidth) / 2;
    const offsetY = (targetSize - scaledHeight) / 2;

    // Fill with neutral gray background
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetSize, targetSize);

    // Draw scaled image
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data;

    // Convert to CHW tensor format with [0,1] normalization
    const tensor = new Float32Array(1 * 3 * targetSize * targetSize);

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        const y = Math.floor(pixelIndex / targetSize);
        const x = pixelIndex % targetSize;

        // Simple [0,1] normalization - most compatible
        const r = data[i] / 255.0;
        const g = data[i + 1] / 255.0;
        const b = data[i + 2] / 255.0;

        // CHW format: [C, H, W]
        tensor[0 * targetSize * targetSize + y * targetSize + x] = r;
        tensor[1 * targetSize * targetSize + y * targetSize + x] = g;
        tensor[2 * targetSize * targetSize + y * targetSize + x] = b;
    }

    console.log(`Tensor created successfully for size ${targetSize}`);
    return new ort.Tensor('float32', tensor, [1, 3, targetSize, targetSize]);
}

// Process BiRefNet result
async function processBiRefNetResult(outputTensor, originalImage) {
    try {
        console.log("Processing BiRefNet result...");

        const canvas = document.getElementById('biRefNetCanvas');
        if (!canvas) {
            throw new Error("BiRefNet canvas not found");
        }

        const ctx = canvas.getContext('2d');

        // Set canvas dimensions to match original image
        canvas.width = originalImage.naturalWidth;
        canvas.height = originalImage.naturalHeight;

        console.log("Canvas dimensions:", canvas.width, "x", canvas.height);

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw original image
        ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
        console.log("Original image drawn to canvas");

        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        console.log("Image data retrieved, length:", data.length);

        // Get mask data - BiRefNet outputs various formats
        if (!outputTensor || !outputTensor.data) {
            throw new Error("Invalid output tensor");
        }

        const maskData = outputTensor.data;
        const dims = outputTensor.dims;
        console.log("Mask dimensions:", dims);
        console.log("Mask data type:", typeof maskData);
        console.log("Mask data length:", maskData.length);

        // Handle different output formats
        let maskHeight, maskWidth;
        if (dims.length === 4) {
            // Format: [batch, channels, height, width]
            maskHeight = dims[2];
            maskWidth = dims[3];
        } else if (dims.length === 3) {
            // Format: [batch, height, width]  
            maskHeight = dims[1];
            maskWidth = dims[2];
        } else if (dims.length === 2) {
            // Format: [height, width]
            maskHeight = dims[0];
            maskWidth = dims[1];
        } else {
            throw new Error(`Unexpected mask dimensions: ${dims}`);
        }

        console.log("Mask size:", maskWidth, "x", maskHeight);

        // Calculate scaling factors
        const scaleX = maskWidth / canvas.width;
        const scaleY = maskHeight / canvas.height;

        console.log("Scale factors:", scaleX, scaleY);

        // Apply mask to remove background
        let processedPixels = 0;
        let transparentPixels = 0;

        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const pixelIndex = (y * canvas.width + x) * 4;

                // Get corresponding mask position
                const maskX = Math.min(Math.floor(x * scaleX), maskWidth - 1);
                const maskY = Math.min(Math.floor(y * scaleY), maskHeight - 1);
                const maskIndex = maskY * maskWidth + maskX;

                // Validate mask index
                if (maskIndex >= maskData.length) {
                    console.warn(`Mask index out of bounds: ${maskIndex} >= ${maskData.length}`);
                    continue;
                }

                // Get mask value
                let maskValue = maskData[maskIndex];

                // Apply sigmoid if values seem to be logits (outside 0-1 range)
                if (maskValue > 1 || maskValue < 0) {
                    maskValue = 1 / (1 + Math.exp(-maskValue));
                }

                // Apply threshold and set alpha
                if (maskValue < 0.5) {
                    data[pixelIndex + 3] = 0; // Make transparent
                    transparentPixels++;
                } else {
                    // Keep original alpha or apply soft masking
                    data[pixelIndex + 3] = Math.min(255, Math.max(0, data[pixelIndex + 3] * maskValue));
                }

                processedPixels++;
            }
        }

        console.log("Processed pixels:", processedPixels);
        console.log("Transparent pixels:", transparentPixels);
        console.log("Foreground pixels:", processedPixels - transparentPixels);

        // Put processed image data back
        ctx.putImageData(imageData, 0, 0);
        console.log("Image data applied to canvas");

        // Show the BiRefNet container
        const biRefNetContainer = document.getElementById('biRefNetContainer');
        if (biRefNetContainer) {
            biRefNetContainer.style.display = 'block';
            console.log("BiRefNet container shown");
        } else {
            throw new Error("BiRefNet container not found");
        }

    } catch (error) {
        console.error("Error in processBiRefNetResult:");
        console.error("Error type:", typeof error);
        console.error("Error message:", error.message || error);
        console.error("Error stack:", error.stack);
        throw error; // Re-throw to be caught by the main function
    }
}

// Function to download BiRefNet image - Global scope
window.downloadBiRefNetImage = function () {
    const canvas = document.getElementById('biRefNetCanvas');
    if (!canvas) {
        alert('Nenhuma imagem BiRefNet disponível para download.');
        return;
    }

    // Create download link
    const link = document.createElement('a');
    link.download = `birefnet-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

// Preview function for file input - Global scope
window.previewImage = function (input) {
    const preview = document.getElementById('preview');
    const previewContainer = document.getElementById('imagePreview');

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
        }

        reader.readAsDataURL(input.files[0]);
    } else {
        previewContainer.style.display = 'none';
    }
};
window.removeBackgroundRMBG = async function () {
    try {
        const uploadedImage = document.getElementById('uploadedImage');
        if (!uploadedImage || !uploadedImage.src) {
            alert('Por favor, faça upload de uma imagem primeiro.');
            return;
        }

        // Mostrar loading
        const button = document.querySelector('button[onclick="removeBackgroundRMBG()"]');
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando RMBG...';
        button.disabled = true;

        console.log('🎯 Starting RMBG background removal...');

        // Inicializar RMBG se necessário
        if (!window.rmbgRemover.isInitialized) {
            await window.rmbgRemover.initialize();
        }

        // Processar imagem
        const resultDataUrl = await window.rmbgRemover.removeBackground(uploadedImage);

        // Mostrar resultado
        const container = document.getElementById('rmbgContainer');
        const canvas = document.getElementById('rmbgCanvas');
        const downloadBtn = document.getElementById('downloadRMBGBtn');

        if (container && canvas && downloadBtn) {
            container.style.display = 'block';

            const img = new Image();
            img.onload = function () {
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                // Limpar canvas e definir fundo transparente
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Desenhar imagem com transparência
                ctx.drawImage(img, 0, 0);

                console.log('✅ Image displayed on canvas:', canvas.width, 'x', canvas.height);
            };
            img.onerror = function () {
                console.error('❌ Error loading processed image');
            };
            img.src = resultDataUrl;

            downloadBtn.style.display = 'inline-block';
            downloadBtn.onclick = () => downloadRMBGImage(resultDataUrl);
        } else {
            console.error('❌ Missing UI elements:', { container, canvas, downloadBtn });
        }

        console.log('✅ RMBG processing completed successfully!');
        
        // Mostrar botão do crachá após processamento bem-sucedido
        showBadgeButton();

    } catch (error) {
        console.error('❌ Error in RMBG processing:', error);
        alert(`Erro no processamento RMBG: ${error.message}`);
    } finally {
        // Restaurar botão
        const button = document.querySelector('button[onclick="removeBackgroundRMBG()"]');
        button.innerHTML = '<i class="fas fa-eraser"></i> Remove Background (RMBG)';
        button.disabled = false;
    }
};

// Function to remove background using MODNet - Global scope
window.removeBackgroundMODNet = async function () {
    try {
        const uploadedImage = document.getElementById('uploadedImage');
        if (!uploadedImage || !uploadedImage.src) {
            alert('Por favor, faça upload de uma imagem primeiro.');
            return;
        }

        // Mostrar loading
        const button = document.querySelector('button[onclick="removeBackgroundMODNet()"]');
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando MODNet...';
        button.disabled = true;

        console.log('🎯 Starting MODNet background removal...');

        // Inicializar MODNet se necessário
        if (!window.modnetRemover.isInitialized) {
            await window.modnetRemover.initialize();
        }

        // Processar imagem
        const resultDataUrl = await window.modnetRemover.removeBackground(uploadedImage);

        // Mostrar resultado
        const container = document.getElementById('modnetContainer');
        const canvas = document.getElementById('modnetCanvas');
        const downloadBtn = document.getElementById('downloadMODNetBtn');

        if (container && canvas && downloadBtn) {
            container.style.display = 'block';

            const img = new Image();
            img.onload = function () {
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                // Limpar canvas e definir fundo transparente
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Desenhar imagem com transparência
                ctx.drawImage(img, 0, 0);

                console.log('✅ MODNet image displayed on canvas:', canvas.width, 'x', canvas.height);
            };
            img.onerror = function () {
                console.error('❌ Error loading MODNet processed image');
            };
            img.src = resultDataUrl;

            downloadBtn.style.display = 'inline-block';
            downloadBtn.onclick = () => downloadMODNetImage(resultDataUrl);
        } else {
            console.error('❌ Missing MODNet UI elements:', { container, canvas, downloadBtn });
        }

        console.log('✅ MODNet processing completed successfully!');
        
        // Mostrar botão do crachá após processamento bem-sucedido
        showBadgeButton();

    } catch (error) {
        console.error('❌ Error in MODNet processing:', error);
        alert(`Erro no processamento MODNet: ${error.message}`);
    } finally {
        // Restaurar botão
        const button = document.querySelector('button[onclick="removeBackgroundMODNet()"]');
        button.innerHTML = '<i class="fas fa-user"></i> Remove Background (MODNet)';
        button.disabled = false;
    }
};

// Function to remove background with MODNet + center face + round crop - Global scope
window.removeBackgroundMODNetRound = async function () {
    try {
        const uploadedImage = document.getElementById('uploadedImage');
        if (!uploadedImage || !uploadedImage.src) {
            alert('Por favor, faça upload de uma imagem primeiro.');
            return;
        }

        // Mostrar loading
        const button = document.querySelector('button[onclick="removeBackgroundMODNetRound()"]');
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando MODNet + Redondo...';
        button.disabled = true;

        console.log('🎯 Starting MODNet background removal + round crop...');

        // Inicializar MODNet se necessário
        if (!window.modnetRemover.isInitialized) {
            await window.modnetRemover.initialize();
        }

        // Obter detecção de rosto se disponível
        let faceDetection = null;
        if (currentDetections && currentDetections.length > 0) {
            faceDetection = currentDetections[0]; // Usar primeira detecção
        }

        // Processar imagem com MODNet + crop redondo
        const resultDataUrl = await window.modnetRemover.removeBackgroundAndCropRound(uploadedImage, faceDetection);

        // Mostrar resultado
        const container = document.getElementById('modnetRoundContainer');
        const canvas = document.getElementById('modnetRoundCanvas');
        const downloadBtn = document.getElementById('downloadMODNetRoundBtn');

        if (container && canvas && downloadBtn) {
            container.style.display = 'block';

            const img = new Image();
            img.onload = function () {
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                // Limpar canvas e definir fundo transparente
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Desenhar imagem com transparência
                ctx.drawImage(img, 0, 0);

                console.log('✅ MODNet round image displayed on canvas:', canvas.width, 'x', canvas.height);
            };
            img.onerror = function () {
                console.error('❌ Error loading MODNet round processed image');
            };
            img.src = resultDataUrl;

            downloadBtn.style.display = 'inline-block';
            downloadBtn.onclick = () => downloadMODNetRoundImage(resultDataUrl);
        } else {
            console.error('❌ Missing MODNet Round UI elements:', { container, canvas, downloadBtn });
        }

        console.log('✅ MODNet round processing completed successfully!');
        
        // Mostrar botão do crachá após processamento bem-sucedido
        showBadgeButton();

    } catch (error) {
        console.error('❌ Error in MODNet round processing:', error);
        alert(`Erro no processamento MODNet Round: ${error.message}`);
    } finally {
        // Restaurar botão
        const button = document.querySelector('button[onclick="removeBackgroundMODNetRound()"]');
        button.innerHTML = '<i class="fas fa-user-circle"></i> MODNet + Redondo';
        button.disabled = false;
    }
};


// Download RMBG image
window.downloadRMBGImage = function () {
    const canvas = document.getElementById('rmbgCanvas');
    if (!canvas || !canvas.getContext) {
        alert('Nenhuma imagem RMBG disponível para download.');
        return;
    }

    const link = document.createElement('a');
    link.download = 'imagem_rmbg.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
};

// Download MODNet image
window.downloadMODNetImage = function () {
    const canvas = document.getElementById('modnetCanvas');
    if (!canvas || !canvas.getContext) {
        alert('Nenhuma imagem MODNet disponível para download.');
        return;
    }

    const link = document.createElement('a');
    link.download = 'imagem_modnet.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
};

// Download MODNet Round image
window.downloadMODNetRoundImage = function () {
    const canvas = document.getElementById('modnetRoundCanvas');
    if (!canvas || !canvas.getContext) {
        alert('Nenhuma imagem MODNet Round disponível para download.');
        return;
    }

    const link = document.createElement('a');
    link.download = 'foto_perfil_modnet.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
};

// Function to generate badge from any processed result - Global scope
window.generateBadgeFromResult = function(sourceType = 'auto') {
    // Detectar automaticamente qual resultado usar se não especificado
    let sourceCanvas = null;
    let sourceName = '';

    if (sourceType === 'auto') {
        // Priorizar resultados na ordem: MODNet Round > MODNet > RMBG > Round Crop > Face Crop
        const canvases = [
            { id: 'modnetRoundCanvas', name: 'MODNet Round' },
            { id: 'modnetCanvas', name: 'MODNet' },
            { id: 'rmbgCanvas', name: 'RMBG' },
            { id: 'roundCroppedCanvas', name: 'Round Crop' },
            { id: 'croppedCanvas', name: 'Face Crop' },
            { id: 'biRefNetCanvas', name: 'BiRefNet' }
        ];

        for (const canvasInfo of canvases) {
            const canvas = document.getElementById(canvasInfo.id);
            if (canvas && canvas.width > 0 && canvas.height > 0) {
                sourceCanvas = canvas;
                sourceName = canvasInfo.name;
                break;
            }
        }
    } else {
        // Usar tipo específico
        sourceCanvas = document.getElementById(sourceType);
        sourceName = sourceType;
    }

    if (!sourceCanvas) {
        alert('Nenhuma imagem processada encontrada. Execute primeiro algum processamento de imagem.');
        return;
    }

    console.log(`🎯 Generating badge from ${sourceName}...`);

    // Mostrar modal para entrada do nome
    showBadgeModal(sourceCanvas);
};

// ✅ Botão 1: Gera crachá com recorte redondo, sem fundo, face centralizada
window.generateBadgeFromRoundCrop = function() {
    console.log('🎯 Generating badge from round crop (no background)...');
    
    // Verificar se existe round crop
    const roundCanvas = document.getElementById('roundCroppedCanvas');
    if (!roundCanvas || roundCanvas.width === 0) {
        // Se não existe round crop, criar um
        alert('Primeiro é necessário detectar o rosto e fazer o recorte redondo. Clique em "Detect Faces" e depois em "Crop Round".');
        return;
    }
    
    // Usar o resultado do round crop
    showBadgeModal(roundCanvas, 'round-no-bg');
};

// ✅ Novo Botão: Gerar Crachá Redondo (simples, usando o canvas do recorte redondo)
window.generateBadgeFromRoundSimple = function() {
    console.log('🎯 Generating simple badge from round crop...');
    
    // Verificar se existe round crop
    const roundCanvas = document.getElementById('roundCroppedCanvas');
    if (!roundCanvas || roundCanvas.width === 0) {
        alert('Primeiro é necessário detectar o rosto e fazer o recorte redondo. Clique em "Detect Faces" e depois em "Crop Round".');
        return;
    }
    
    // Usar o resultado do round crop com o processamento padrão
    showBadgeModal(roundCanvas, 'round-simple');
};

// ✅ Botão 2: Gera crachá com fundo + recorte redondo + face centralizada  
window.generateBadgeFromRoundCropWithBackground = function() {
    console.log('🎯 Generating badge from round crop (with background)...');
    
    // Verificar se foi detectado um rosto
    if (!window.detectedFaces || window.detectedFaces.length === 0) {
        alert('Primeiro é necessário detectar o rosto. Clique em "Detect Faces".');
        return;
    }
    
    // Criar um round crop com fundo preservado
    createRoundCropWithBackground().then(canvas => {
        if (canvas) {
            showBadgeModal(canvas, 'round-with-bg');
        } else {
            alert('Erro ao criar recorte redondo com fundo. Tente novamente.');
        }
    });
};

// Função auxiliar para criar round crop preservando o fundo
async function createRoundCropWithBackground() {
    try {
        const uploadedImage = document.querySelector('.uploaded-image');
        if (!uploadedImage || !window.detectedFaces || window.detectedFaces.length === 0) {
            return null;
        }
        
        const face = window.detectedFaces[0];
        
        // Criar canvas temporário
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Definir tamanho do recorte redondo
        const size = 400;
        tempCanvas.width = size;
        tempCanvas.height = size;
        
        // Calcular área do rosto com margem
        const faceWidth = face.width * uploadedImage.naturalWidth;
        const faceHeight = face.height * uploadedImage.naturalHeight;
        const faceX = face.x * uploadedImage.naturalWidth;
        const faceY = face.y * uploadedImage.naturalHeight;
        
        // Adicionar margem
        const margin = Math.max(faceWidth, faceHeight) * 0.3;
        const cropSize = Math.max(faceWidth, faceHeight) + margin * 2;
        
        // Centralizar o crop no rosto
        const cropX = faceX + faceWidth/2 - cropSize/2;
        const cropY = faceY + faceHeight/2 - cropSize/2;
        
        // Criar máscara circular
        tempCtx.save();
        tempCtx.beginPath();
        tempCtx.arc(size/2, size/2, size/2, 0, Math.PI * 2);
        tempCtx.clip();
        
        // Desenhar a imagem original com fundo
        tempCtx.drawImage(
            uploadedImage,
            cropX, cropY, cropSize, cropSize,
            0, 0, size, size
        );
        
        tempCtx.restore();
        
        console.log('✅ Round crop with background created successfully');
        return tempCanvas;
        
    } catch (error) {
        console.error('❌ Error creating round crop with background:', error);
        return null;
    }
}

// Show badge generation modal
function showBadgeModal(sourceCanvas, processType = 'auto') {
    // Verificar se modal já existe
    let modal = document.getElementById('badgeModal');
    if (!modal) {
        // Criar modal dinamicamente
        modal = createBadgeModal();
        document.body.appendChild(modal);
    }

    // Mostrar modal
    modal.style.display = 'flex';
    
    // Limpar input anterior
    const nameInput = document.getElementById('candidateNameInput');
    nameInput.value = '';
    nameInput.focus();

    // Configurar botões
    const confirmBtn = document.getElementById('confirmBadgeBtn');
    const cancelBtn = document.getElementById('cancelBadgeBtn');
    const previewBtn = document.getElementById('previewBadgeBtn');

    // Remover event listeners anteriores
    confirmBtn.replaceWith(confirmBtn.cloneNode(true));
    cancelBtn.replaceWith(cancelBtn.cloneNode(true));
    previewBtn.replaceWith(previewBtn.cloneNode(true));

    // Reobter referências após clonagem
    const newConfirmBtn = document.getElementById('confirmBadgeBtn');
    const newCancelBtn = document.getElementById('cancelBadgeBtn');
    const newPreviewBtn = document.getElementById('previewBadgeBtn');

    // Adicionar novos event listeners
    newConfirmBtn.addEventListener('click', () => generateFinalBadge(sourceCanvas, processType));
    newCancelBtn.addEventListener('click', () => closeBadgeModal());
    newPreviewBtn.addEventListener('click', () => generateBadgePreview(sourceCanvas, processType));

    // Enter key para confirmar
    nameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            generateFinalBadge(sourceCanvas, processType);
        }
    });
}

// Create badge modal HTML
function createBadgeModal() {
    const modal = document.createElement('div');
    modal.id = 'badgeModal';
    modal.style.cssText = `
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
        align-items: center;
        justify-content: center;
        font-family: Arial, sans-serif;
    `;

    modal.innerHTML = `
        <div style="
            background: white;
            border-radius: 15px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            position: relative;
        ">
            <div style="text-align: center; margin-bottom: 25px;">
                <h3 style="color: #e74c3c; margin: 0 0 10px 0; font-size: 24px;">
                    <i class="fas fa-id-card" style="margin-right: 10px;"></i>
                    Gerar Crachá
                </h3>
                <p style="color: #7f8c8d; margin: 0; font-size: 14px;">
                    Digite o nome do candidato para gerar o crachá profissional
                </p>
            </div>

            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; color: #2c3e50; font-weight: bold;">
                    Nome do Candidato:
                </label>
                <input 
                    type="text" 
                    id="candidateNameInput" 
                    placeholder="Ex: João Silva Santos"
                    style="
                        width: 100%;
                        padding: 12px;
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        font-size: 16px;
                        box-sizing: border-box;
                        transition: border-color 0.3s;
                    "
                    maxlength="100"
                />
                <small style="color: #7f8c8d; font-size: 12px;">
                    Mínimo 2 caracteres, máximo 100 caracteres
                </small>
            </div>

            <div id="badgePreviewContainer" style="
                text-align: center;
                margin: 20px 0;
                display: none;
            ">
                <h5 style="color: #2c3e50; margin-bottom: 10px;">Preview:</h5>
                <div style="
                    border: 2px dashed #e0e0e0;
                    border-radius: 10px;
                    padding: 15px;
                    background: #f8f9fa;
                ">
                    <canvas id="badgePreviewCanvas" style="
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    "></canvas>
                </div>
            </div>

            <div style="
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 25px;
            ">
                <button 
                    id="previewBadgeBtn"
                    style="
                        background: #3498db;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.3s;
                        min-width: 120px;
                    "
                    onmouseover="this.style.background='#2980b9'"
                    onmouseout="this.style.background='#3498db'"
                >
                    <i class="fas fa-eye"></i> Preview
                </button>
                
                <button 
                    id="confirmBadgeBtn"
                    style="
                        background: #e74c3c;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.3s;
                        min-width: 120px;
                    "
                    onmouseover="this.style.background='#c0392b'"
                    onmouseout="this.style.background='#e74c3c'"
                >
                    <i class="fas fa-download"></i> Gerar
                </button>
                
                <button 
                    id="cancelBadgeBtn"
                    style="
                        background: #95a5a6;
                        color: white;
                        border: none;
                        padding: 12px 20px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.3s;
                        min-width: 120px;
                    "
                    onmouseover="this.style.background='#7f8c8d'"
                    onmouseout="this.style.background='#95a5a6'"
                >
                    <i class="fas fa-times"></i> Cancelar
                </button>
            </div>
        </div>
    `;

    return modal;
}

// Generate badge preview
async function generateBadgePreview(sourceCanvas, processType = 'auto') {
    const nameInput = document.getElementById('candidateNameInput');
    const candidateName = nameInput.value.trim();

    if (!window.badgeGenerator.validateCandidateName(candidateName)) {
        alert('Por favor, digite um nome válido (mínimo 2 caracteres).');
        nameInput.focus();
        return;
    }

    try {
        // Mostrar loading no botão
        const previewBtn = document.getElementById('previewBadgeBtn');
        const originalText = previewBtn.innerHTML;
        previewBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Gerando...';
        previewBtn.disabled = true;

        // Gerar preview com tipo específico
        const previewDataUrl = await window.badgeGenerator.generatePreview(sourceCanvas, candidateName, processType);

        // Mostrar preview
        const previewContainer = document.getElementById('badgePreviewContainer');
        const previewCanvas = document.getElementById('badgePreviewCanvas');
        
        const img = new Image();
        img.onload = function() {
            previewCanvas.width = img.width;
            previewCanvas.height = img.height;
            const ctx = previewCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            previewContainer.style.display = 'block';
        };
        img.src = previewDataUrl;

        // Restaurar botão
        previewBtn.innerHTML = originalText;
        previewBtn.disabled = false;

    } catch (error) {
        console.error('Error generating preview:', error);
        alert('Erro ao gerar preview. Tente novamente.');
        
        // Restaurar botão
        const previewBtn = document.getElementById('previewBadgeBtn');
        previewBtn.innerHTML = '<i class="fas fa-eye"></i> Preview';
        previewBtn.disabled = false;
    }
}

// Generate final high-quality badge
async function generateFinalBadge(sourceCanvas, processType = 'auto') {
    const nameInput = document.getElementById('candidateNameInput');
    const candidateName = nameInput.value.trim();

    if (!window.badgeGenerator.validateCandidateName(candidateName)) {
        alert('Por favor, digite um nome válido (mínimo 2 caracteres).');
        nameInput.focus();
        return;
    }

    try {
        // Mostrar loading no botão
        const confirmBtn = document.getElementById('confirmBadgeBtn');
        const originalText = confirmBtn.innerHTML;
        confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Gerando...';
        confirmBtn.disabled = true;

        // Gerar badge em alta qualidade com tipo específico
        const badgeDataUrl = await window.badgeGenerator.generateHighQuality(sourceCanvas, candidateName, processType);

        // Download automático com nome específico baseado no tipo
        let fileName = `cracha-${candidateName.replace(/\s+/g, '-').toLowerCase()}-${Date.now()}`;
        if (processType === 'round-no-bg') {
            fileName += '-sem-fundo';
        } else if (processType === 'round-with-bg') {
            fileName += '-com-fundo';
        } else if (processType === 'round-simple') {
            fileName += '-redondo';
        }
        fileName += '.png';
        
        window.badgeGenerator.downloadBadge(badgeDataUrl, fileName);

        // Fechar modal
        closeBadgeModal();

        // Mostrar sucesso
        setTimeout(() => {
            alert('✅ Crachá gerado e baixado com sucesso!');
        }, 500);

    } catch (error) {
        console.error('Error generating badge:', error);
        alert('Erro ao gerar crachá. Tente novamente.');
        
        // Restaurar botão
        const confirmBtn = document.getElementById('confirmBadgeBtn');
        confirmBtn.innerHTML = originalText;
        confirmBtn.disabled = false;
    }
}

// Close badge modal
function closeBadgeModal() {
    const modal = document.getElementById('badgeModal');
    if (modal) {
        modal.style.display = 'none';
        
        // Limpar preview
        const previewContainer = document.getElementById('badgePreviewContainer');
        if (previewContainer) {
            previewContainer.style.display = 'none';
        }
    }
}

// Show badge button when there's a processed result
function showBadgeButton() {
    const badgeBtn = document.getElementById('generateBadgeBtn');
    if (badgeBtn) {
        badgeBtn.style.display = 'inline-block';
    }
}

// Show specific badge buttons based on processing type
function showSpecificBadgeButtons() {
    // Mostrar botão simples para round crop
    const roundSimpleBtn = document.getElementById('generateBadgeRoundSimpleBtn');
    const roundCanvas = document.getElementById('roundCroppedCanvas');
    if (roundSimpleBtn && roundCanvas && roundCanvas.width > 0) {
        roundSimpleBtn.style.display = 'inline-block';
    }
    
    // Mostrar botão para round crop sem fundo
    const roundBtn = document.getElementById('generateBadgeRoundBtn');
    if (roundBtn && roundCanvas && roundCanvas.width > 0) {
        roundBtn.style.display = 'inline-block';
    }
    
    // Mostrar botão para round crop com fundo (sempre disponível se há detecção de rosto)
    const roundWithBgBtn = document.getElementById('generateBadgeRoundWithBgBtn');
    if (roundWithBgBtn && window.detectedFaces && window.detectedFaces.length > 0) {
        roundWithBgBtn.style.display = 'inline-block';
    }
}

// Hide badge button
function hideBadgeButton() {
    const badgeBtn = document.getElementById('generateBadgeBtn');
    if (badgeBtn) {
        badgeBtn.style.display = 'none';
    }
    
    const roundSimpleBtn = document.getElementById('generateBadgeRoundSimpleBtn');
    if (roundSimpleBtn) {
        roundSimpleBtn.style.display = 'none';
    }
    
    const roundBtn = document.getElementById('generateBadgeRoundBtn');
    if (roundBtn) {
        roundBtn.style.display = 'none';
    }
    
    const roundWithBgBtn = document.getElementById('generateBadgeRoundWithBgBtn');
    if (roundWithBgBtn) {
        roundWithBgBtn.style.display = 'none';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing face detection...');
    initializeFaceDetector();

    // Initialize Badge Generator
    if (typeof BadgeGenerator !== 'undefined') {
        try {
            window.badgeGenerator = new BadgeGenerator();
            console.log('✅ Badge Generator initialized');
        } catch (error) {
            console.log('⚠️ Badge Generator not available:', error.message);
        }
    }

    // Initialize RMBG in background
    setTimeout(async () => {
        try {
            await rmbgBackgroundRemover.loadModel();
            console.log('✅ RMBG model preloaded successfully');
        } catch (error) {
            console.log('⚠️ RMBG model not available:', error.message);
        }
    }, 2000);

    // Initialize MODNet in background
    setTimeout(async () => {
        try {
            if (window.modnetRemover) {
                await window.modnetRemover.initialize();
                console.log('✅ MODNet model preloaded successfully');
            }
        } catch (error) {
            console.log('⚠️ MODNet model not available:', error.message);
        }
    }, 3000);
});
