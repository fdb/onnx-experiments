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

function convertTensorToImageData(tensor) {
  const size = 256;
  const rgba = new Uint8ClampedArray(size * size * 4);
  let redOffset = 0;
  let greenOffset = size * size;
  let blueOffset = size * size * 2;
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = tensor.data[redOffset] * 255;
    rgba[i + 1] = tensor.data[greenOffset] * 255;
    rgba[i + 2] = tensor.data[blueOffset] * 255;
    rgba[i + 3] = 255;
    // Increment offsets
    redOffset++;
    greenOffset++;
    blueOffset++;
  }
  return new ImageData(rgba, size, size);
}

async function preload() {
  log("Loading model (273MB)...");
  window.session = await ort.InferenceSession.create(
    "https://algorithmicgaze.s3.amazonaws.com/projects/2024-onnx-experiments/models/celeb-faces-vae-decoder.onnx",
  );
  log("Model ready.");
}

async function run() {
  log(`Create random tensor...`);
  const data = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    data[i] = Math.random() * 2 - 1;
  }
  const latentTensor = new ort.Tensor("float32", data, [1, 256]);

  const inputName = session.inputNames[0];
  const feeds = { [inputName]: latentTensor };

  log(`Running inference...`);
  const results = await window.session.run(feeds);

  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];

  log(`Converting to image...`);
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const imageData = convertTensorToImageData(outputTensor);
  ctx.putImageData(imageData, 0, 0);

  log(`Done.`);
}

function chooseImage(e) {
  run(e.target.src);
}

preload();

// Run button
const runButton = document.getElementById("run-button");
runButton.addEventListener("click", run);
