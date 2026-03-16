import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
  } from "@mediapipe/tasks-vision";
  
  const video = document.getElementById("webcam");
  const canvasElement = document.getElementById("output_canvas");
  const canvasCtx = canvasElement.getContext("2d");
  const enableWebcamButton = document.getElementById("webcamButton");
  const loadingIndicator = document.getElementById("loading");
  const permissionRequest = document.getElementById("permission-request");
  const statusText = document.getElementById("status-text");
  const pulseDot = document.querySelector(".pulse-dot");
  const predictionText = document.getElementById("prediction-text");
  
  // Data Collection Elements
  const labelInput = document.getElementById("label-input");
  const recordBtn = document.getElementById("record-btn");
  const recordJsonBtn = document.getElementById("record-json-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  const downloadBtn = document.getElementById("download-btn");
  const collectionStatus = document.getElementById("collection-status");
  
  let handLandmarker = undefined;
  let runningMode = "VIDEO";
  let webcamRunning = false;
  let lastVideoTime = -1;
  let results = undefined;
  
  // Custom AI Model state
  let gestureModel = null;
  let gestureClasses = [];
  
  // Data Collection State
  let isRecording = false;
  let recordingMode = 'csv'; // 'csv' or 'json'
  let recordedFrames = 0;
  const targetFrames = 200;
  let currentLabel = "";
  let collectedDataCsv = "label,Lx0,Ly0,Lz0,Lx1,Ly1,Lz1,Lx2,Ly2,Lz2,Lx3,Ly3,Lz3,Lx4,Ly4,Lz4,Lx5,Ly5,Lz5,Lx6,Ly6,Lz6,Lx7,Ly7,Lz7,Lx8,Ly8,Lz8,Lx9,Ly9,Lz9,Lx10,Ly10,Lz10,Lx11,Ly11,Lz11,Lx12,Ly12,Lz12,Lx13,Ly13,Lz13,Lx14,Ly14,Lz14,Lx15,Ly15,Lz15,Lx16,Ly16,Lz16,Lx17,Ly17,Lz17,Lx18,Ly18,Lz18,Lx19,Ly19,Lz19,Lx20,Ly20,Lz20,Rx0,Ry0,Rz0,Rx1,Ry1,Rz1,Rx2,Ry2,Rz2,Rx3,Ry3,Rz3,Rx4,Ry4,Rz4,Rx5,Ry5,Rz5,Rx6,Ry6,Rz6,Rx7,Ry7,Rz7,Rx8,Ry8,Rz8,Rx9,Ry9,Rz9,Rx10,Ry10,Rz10,Rx11,Ry11,Rz11,Rx12,Ry12,Rz12,Rx13,Ry13,Rz13,Rx14,Ry14,Rz14,Rx15,Ry15,Rz15,Rx16,Ry16,Rz16,Rx17,Ry17,Rz17,Rx18,Ry18,Rz18,Rx19,Ry19,Rz19,Rx20,Ry20,Rz20\n";
  let collectedDataJson = {}; // E.g., { "HELLO": [[{x,y}, {x,y}...], [{x,y}...]] }
  
  // --- 0. Data Collection Logic ---
  recordBtn.addEventListener("click", () => {
    if (!webcamRunning) {
        alert("Please enable the camera first!");
        return;
    }
    const label = labelInput.value.trim().toUpperCase();
    if (label.length !== 1 || !label.match(/[A-Z]/i)) {
        alert("For CSV training, please enter a single valid letter (A-Z)!");
        return;
    }
    
    currentLabel = label;
    isRecording = true;
    recordingMode = 'csv';
    recordedFrames = 0;
    labelInput.disabled = true;
    recordBtn.style.display = "none";
    recordJsonBtn.style.display = "none";
    cancelBtn.style.display = "inline-block";
    downloadBtn.disabled = true;
    
    collectionStatus.style.color = "#f97316"; 
    collectionStatus.innerText = `Waiting for your hand to record static CSV '${currentLabel}' (${recordedFrames}/${targetFrames})...`;
  });

  recordJsonBtn.addEventListener("click", () => {
    if (!webcamRunning) {
        alert("Please enable the camera first!");
        return;
    }
    const label = labelInput.value.trim().toUpperCase();
    if (label.length === 0) {
        alert("Please enter a word or sign name (e.g. THANK_YOU)!");
        return;
    }
    
    currentLabel = label;
    if (!collectedDataJson[currentLabel]) {
        collectedDataJson[currentLabel] = [];
    }
    
    isRecording = true;
    recordingMode = 'json';
    recordedFrames = 0;
    labelInput.disabled = true;
    recordBtn.style.display = "none";
    recordJsonBtn.style.display = "none";
    cancelBtn.style.display = "inline-block";
    downloadBtn.disabled = true;
    
    collectionStatus.style.color = "#f97316"; 
    collectionStatus.innerText = `Waiting for your hand to record JSON Sequence for '${currentLabel}' (${recordedFrames}/${targetFrames})...`;
  });
  
  cancelBtn.addEventListener("click", () => {
      isRecording = false;
      labelInput.disabled = false;
      recordBtn.style.display = "inline-block";
      recordJsonBtn.style.display = "inline-block";
      cancelBtn.style.display = "none";
      collectionStatus.style.color = "#fca5a5"; 
      collectionStatus.innerText = `Recording cancelled.`;
      
      if (collectedDataCsv.split("\n").length > 2 || Object.keys(collectedDataJson).length > 0) {
          downloadBtn.disabled = false;
      }
  });

  downloadBtn.addEventListener("click", () => {
      // Download CSV if there's data
      if (collectedDataCsv.split("\n").length > 2) {
          const csvBlob = new Blob([collectedDataCsv], { type: 'text/csv;charset=utf-8;' });
          const csvUrl = URL.createObjectURL(csvBlob);
          const csvLink = document.createElement("a");
          csvLink.href = csvUrl;
          csvLink.download = "hand_landmarks_dataset.csv";
          document.body.appendChild(csvLink);
          csvLink.click();
          document.body.removeChild(csvLink);
      }
      
      // Download JSON if there's data
      if (Object.keys(collectedDataJson).length > 0) {
          const jsonStr = JSON.stringify(collectedDataJson, null, 2);
          const jsonBlob = new Blob([jsonStr], { type: 'application/json' });
          const jsonUrl = URL.createObjectURL(jsonBlob);
          const jsonLink = document.createElement("a");
          jsonLink.href = jsonUrl;
          jsonLink.download = "gesture_sequences.json";
          document.body.appendChild(jsonLink);
          jsonLink.click();
          document.body.removeChild(jsonLink);
      }
  });
  
  // --- 1. Load the Custom AI Model ---
  async function loadGestureModel() {
      try {
          const response = await fetch('/tfjs_model/weights.json');
          if (response.ok) {
              gestureModel = await response.json();
              gestureClasses = gestureModel.classes;
              console.log("Custom gesture model loaded successfully!", gestureClasses);
          } else {
              console.log("No custom model found at /tfjs_model/weights.json. Please train one first!");
          }
      } catch (e) {
          console.log("Error loading custom model.", e);
      }
  }
  loadGestureModel();
  
  // --- 2. Neural Network Inference Logic ---
  // A simple function to predict via our exported Dense weights
  function predictNeuralNetwork(features) {
      if (!gestureModel || gestureClasses.length === 0) return "?";
      
      let activations = features;
      
      // Feed-forward through all layers
      for (const layer of gestureModel.layers) {
          if (layer.type === "Dense") {
              const next_activations = new Array(layer.units).fill(0);
              
              // Matrix multiplication: Output = Input * Weights + Biases
              for (let i = 0; i < layer.units; i++) {
                  let sum = layer.biases[i];
                  for (let j = 0; j < activations.length; j++) {
                      sum += activations[j] * layer.weights[j][i];
                  }
                  
                  // Activation function
                  if (layer.activation === "relu") {
                      next_activations[i] = Math.max(0, sum);
                  } else if (layer.activation === "softmax") {
                      next_activations[i] = sum; // Softmax applied after loop
                  } else {
                      next_activations[i] = sum; // Linear fallback
                  }
              }
              
              if (layer.activation === "softmax") {
                  // Apply softmax to array
                  let maxVal = -Infinity;
                  for (let i = 0; i < next_activations.length; i++) {
                      if (next_activations[i] > maxVal) maxVal = next_activations[i];
                  }
                  
                  let sumExp = 0;
                  for (let i = 0; i < next_activations.length; i++) {
                      next_activations[i] = Math.exp(next_activations[i] - maxVal);
                      sumExp += next_activations[i];
                  }
                  
                  for (let i = 0; i < next_activations.length; i++) {
                      next_activations[i] /= sumExp;
                  }
              }
              
              activations = next_activations;
          }
      }
      
      // Find highest probability
      let maxProb = -1;
      let maxIdx = -1;
      for (let i = 0; i < activations.length; i++) {
          if (activations[i] > maxProb) {
              maxProb = activations[i];
              maxIdx = i;
          }
      }
      
      // Ensure prediction is fairly confident
      if (maxProb > 0.6) {
          return gestureClasses[maxIdx];
      } else {
          return "?";
      }
  }
  
  // Before we can use HandLandmarker class we must wait for it to finish loading
  const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numHands: 2
    });
    
    // Model loaded, ready to prompt for camera
    loadingIndicator.classList.remove("active");
    permissionRequest.classList.add("active");
    statusText.innerText = "Model loaded. Waiting for camera permissions.";
  };
  createHandLandmarker();
  
  const hasGetUserMedia = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };
  
  if (hasGetUserMedia()) {
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
    statusText.innerText = "Camera not supported by browser.";
  }
  
  function enableCam(event) {
    if (!handLandmarker) {
      console.log("Wait! objectDetector not loaded yet.");
      return;
    }
  
    if (webcamRunning === true) {
      webcamRunning = false;
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
      webcamRunning = true;
      enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
  
    // getUsermedia parameters.
    const constraints = {
      video: { width: 640, height: 480 }
    };
  
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
      
      // UI Updates
      permissionRequest.classList.remove("active");
      statusText.innerText = "Camera active. Tracking hands...";
      pulseDot.classList.add("active");
    });
  }
  
  async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = handLandmarker.detectForVideo(video, startTimeMs);
    }
  
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results && results.landmarks) {
      const drawingUtils = new DrawingUtils(canvasCtx);
      
      // We need 126 features total (63 for Left, 63 for Right). Default to 0 format.
      const featuresRaw = new Array(126).fill(0);
      let leftHandData = null;
      let rightHandData = null;

      // Extract left and right hand if present
      for (let h = 0; h < results.landmarks.length; h++) {
          const landmarks = results.landmarks[h];
          const handedness = results.handednesses[h][0].categoryName; // 'Left' or 'Right'
          
          if (handedness === 'Left') {
              leftHandData = landmarks;
          } else if (handedness === 'Right') {
              rightHandData = landmarks;
          }
      }

      // Populate features array
      if (leftHandData) {
          const wrist_x = leftHandData[0].x;
          const wrist_y = leftHandData[0].y;
          const wrist_z = leftHandData[0].z;
          for (let i = 0; i < 21; i++) {
              featuresRaw[i*3] = leftHandData[i].x - wrist_x;
              featuresRaw[i*3+1] = leftHandData[i].y - wrist_y;
              featuresRaw[i*3+2] = leftHandData[i].z - wrist_z;
          }
      }
      if (rightHandData) {
          const wrist_x = rightHandData[0].x;
          const wrist_y = rightHandData[0].y;
          const wrist_z = rightHandData[0].z;
          for (let i = 0; i < 21; i++) {
              featuresRaw[63 + i*3] = rightHandData[i].x - wrist_x;
              featuresRaw[63 + i*3+1] = rightHandData[i].y - wrist_y;
              featuresRaw[63 + i*3+2] = rightHandData[i].z - wrist_z;
          }
      }

      // Draw all hands
      for (const landmarks of results.landmarks) {
          drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: "#c084fc",
              lineWidth: 3
          });
          drawingUtils.drawLandmarks(landmarks, {
              color: "#38bdf8",
              lineWidth: 2,
              radius: 4
          });
      }

      // 2.5 IF RECORDING DATA
      if (isRecording) {
          recordedFrames++;
          
          if (recordingMode === 'csv') {
              let csvRow = `${currentLabel}`;
              for (let i = 0; i < featuresRaw.length; i++) {
                  csvRow += `,${featuresRaw[i]}`;
              }
              csvRow += "\n";
              collectedDataCsv += csvRow;
              collectionStatus.innerText = `Recording Dual-Hand CSV '${currentLabel}' (${recordedFrames}/${targetFrames})... Hold steady!`;
          } 
          else if (recordingMode === 'json') {
              if (recordedFrames % 5 === 0) {
                  // For simple JSON animation testing (maybe update to dual hand later if needed)
                  // using just the first detected hand for the animation backwards compability
                  const frameCoords = [];
                  for (let i = 0; i < 21; i++) {
                      frameCoords.push({
                          x: results.landmarks[0][i].x,
                          y: results.landmarks[0][i].y
                      });
                  }
                  collectedDataJson[currentLabel].push(frameCoords);
              }
              collectionStatus.innerText = `Recording JSON Sequence '${currentLabel}' (${recordedFrames}/${targetFrames})... Perform the gesture!`;
          }
          
          collectionStatus.style.color = "#10b981"; // Green
          
          if (recordedFrames >= targetFrames) {
              isRecording = false;
              labelInput.disabled = false;
              recordBtn.style.display = "inline-block";
              recordJsonBtn.style.display = "inline-block";
              cancelBtn.style.display = "none";
              downloadBtn.disabled = false;
              collectionStatus.style.color = "#38bdf8"; // Blue
              collectionStatus.innerText = `Finished recording '${currentLabel}'! Save the data when ready.`;
          }
      }

      // 3. Make Prediction
      if (gestureModel && gestureClasses.length > 0 && !isRecording) {
          const letter = predictNeuralNetwork(featuresRaw);
          predictionText.innerText = letter;
      } else if (isRecording) {
          predictionText.innerText = "REC";
      } else {
          predictionText.innerText = "?";
      }
    } else {
        // No hand detected
        predictionText.innerText = "...";
        if (isRecording) {
            collectionStatus.style.color = "#f97316"; // Orange
            collectionStatus.innerText = `Waiting for hand... (${recordedFrames}/${targetFrames})`;
        }
    }
    canvasCtx.restore();
  
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
      window.requestAnimationFrame(predictWebcam);
    }
  }
  
