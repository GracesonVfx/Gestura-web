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
  
  // Custom RF Model state
  let gestureModel = null;
  
  // Data Collection State
  let isRecording = false;
  let recordingMode = 'csv'; // 'csv' or 'json'
  let recordedFrames = 0;
  const targetFrames = 200;
  let currentLabel = "";
  let collectedDataCsv = "label,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6,x7,y7,z7,x8,y8,z8,x9,y9,z9,x10,y10,z10,x11,y11,z11,x12,y12,z12,x13,y13,z13,x14,y14,z14,x15,y15,z15,x16,y16,z16,x17,y17,z17,x18,y18,z18,x19,y19,z19,x20,y20,z20\n";
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
  
  // --- 1. Load the Custom JSON Model ---
  async function loadGestureModel() {
      try {
          const response = await fetch('/model.json');
          if (response.ok) {
              gestureModel = await response.json();
              console.log("Custom gesture model loaded!", gestureModel);
          } else {
              console.log("No custom model found at /model.json. Please train one first!");
          }
      } catch (e) {
          console.log("Error loading model.json. Skipping custom predictions.");
      }
  }
  loadGestureModel();
  
  // --- 2. Random Forest Inference Logic ---
  // A simple function to walk our custom JSON decision trees
  function predictRandomForest(features, model) {
      if (!model || !model.trees || model.trees.length === 0) return "?";
      
      const classCounts = {};
      model.classes.forEach(c => classCounts[c] = 0);
  
      // Pass features through each tree
      for (const tree of model.trees) {
          let node = tree;
          // Traverse until we hit a leaf
          while (node.type !== "leaf") {
              if (features[node.feature] <= node.threshold) {
                  node = node.left;
              } else {
                  node = node.right;
              }
          }
          classCounts[node.class]++;
      }
  
      // Find the class with the most votes
      let bestClass = "?";
      let maxVotes = -1;
      for (const [cls, votes] of Object.entries(classCounts)) {
          if (votes > maxVotes) {
              maxVotes = votes;
              bestClass = cls;
          }
      }
      return bestClass;
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
      
      for (const landmarks of results.landmarks) {
        // --- Custom Inference ---
        // 1. We must center coordinates on the wrist (landmark 0) to match our training data
        const wrist_x = landmarks[0].x;
        const wrist_y = landmarks[0].y;
        const wrist_z = landmarks[0].z;
  
        // 2. Extract and flatten all 21 joints exactly like Python did.
        const features = [];
        for (let i = 0; i < 21; i++) {
            features.push(landmarks[i].x - wrist_x);
            features.push(landmarks[i].y - wrist_y);
            features.push(landmarks[i].z - wrist_z);
        }
  
        // 2.5 IF RECORDING DATA
        if (isRecording) {
            recordedFrames++;
            
            if (recordingMode === 'csv') {
                let csvRow = `${currentLabel}`;
                for (let i = 0; i < features.length; i++) {
                    csvRow += `,${features[i]}`;
                }
                csvRow += "\n";
                collectedDataCsv += csvRow;
                collectionStatus.innerText = `Recording Static CSV '${currentLabel}' (${recordedFrames}/${targetFrames})... Hold steady!`;
            } 
            else if (recordingMode === 'json') {
                // For JSON, we only capture every 5th frame to make the animation file lightweight and smooth
                // (60fps / 5 = 12fps animation skeleton)
                if (recordedFrames % 5 === 0) {
                    // We save the raw 0-1 coordinates for the SVG animator
                    const frameCoords = [];
                    for (let i = 0; i < 21; i++) {
                        frameCoords.push({
                            x: landmarks[i].x,
                            y: landmarks[i].y
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
        if (gestureModel && !isRecording) {
            const letter = predictRandomForest(features, gestureModel);
            predictionText.innerText = letter; 
        } else if (isRecording) {
            predictionText.innerText = "REC";
        } else {
            predictionText.innerText = "?";
        }
  
        // --- Draw the hand ---
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#c084fc", // Purple from our gradient
          lineWidth: 3
        });
        drawingUtils.drawLandmarks(landmarks, {
          color: "#38bdf8", // Blue from our gradient
          lineWidth: 2,
          radius: 4
        });
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
  
