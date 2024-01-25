function log(message) {
  const log = document.getElementById("log");
  log.innerHTML = message;
}

// Image loading adapted from
// https://github.com/microsoft/onnxruntime-nextjs-template/blob/main/utils/imageHelper.ts
async function loadImage(url, size) {
  const buffer = await Jimp.read(url);
  return buffer.resize(size, size);
}

function convertToTensor(bitmap, width, height) {
  const data = new Float32Array(width * height * 3);
  let redOffset = 0;
  let greenOffset = width * height;
  let blueOffset = width * height * 2;
  for (let i = 0; i < width * height; i++) {
    // Index into original bitmap data
    const j = i * 4;
    // Normalize pixel values
    data[redOffset] = bitmap.data[j] / 255;
    data[greenOffset] = bitmap.data[j + 1] / 255;
    data[blueOffset] = bitmap.data[j + 2] / 255;
    // Increment offsets
    redOffset++;
    greenOffset++;
    blueOffset++;
  }
  return new ort.Tensor("float32", data, [1, 3, height, width]);
}

async function preload() {
  log("Loading model...");
  window.session = await ort.InferenceSession.create(
    "https://algorithmicgaze.s3.amazonaws.com/projects/2024-onnx-experiments/models/not-hotdog.onnx",
  );
  log("Model ready.");
}

async function run(url) {
  log(`Loading image...`);

  document.getElementById("current-image-wrapper").innerHTML =
    `<img src="${url}" id="current-image" class="w-48 h-48 object-cover rounded-lg shadow-lg">`;

  const imageBuffer = await loadImage(url, 384);
  const imageTensor = convertToTensor(imageBuffer.bitmap, 384, 384);

  const inputName = session.inputNames[0];
  const feeds = { [inputName]: imageTensor };

  log(`Running inference...`);
  const results = await window.session.run(feeds);
  console.log(results);

  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];
  console.log(outputTensor.data);
  const labels = ["hot_dog", "not_hot_dog"];
  // Return label with heightest probability
  const label =
    labels[outputTensor.data.indexOf(Math.max(...outputTensor.data))];
  console.log(label);
  log(`Label: ${label}<br> ${outputTensor.data}`);
}

function chooseImage(e) {
  run(e.target.src);
}

preload();

// Demo images
document.querySelectorAll(".demo-image").forEach((img) => {
  img.addEventListener("click", chooseImage);
});

// Upload zone
const uploadZoneEl = document.getElementById("upload-zone");

// Invisible file input element
const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.style.display = "none";
document.body.appendChild(fileInput);

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    processFile(file);
  }
});

function processFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    console.log(e.target.result);
    run(e.target.result);
  };
  reader.readAsDataURL(file);
}

uploadZoneEl.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadZoneEl.classList.remove("bg-blue-600");
  const file = e.dataTransfer.files[0];
  if (file) {
    processFile(file);
  }
});

uploadZoneEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadZoneEl.classList.add("bg-blue-600");
});

uploadZoneEl.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadZoneEl.classList.remove("bg-blue-600");
});

uploadZoneEl.addEventListener("click", () => {
  fileInput.click();
});
