// predict.js
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

async function predictFire(lat, lon, year, doy) {
  // Hard-coded training min/max from an example dataset, or store them from training
  const latMin = 17.9,
    latMax = 70.3;
  const lonMin = -179,
    lonMax = -65.3;
  const yearMin = 1992,
    yearMax = 2020;

  function minMaxScale(val, min, max) {
    return max === min ? 0.0 : (val - min) / (max - min);
  }

  // Scale input
  const latScaled = minMaxScale(lat, latMin, latMax);
  const lonScaled = minMaxScale(lon, lonMin, lonMax);
  const yearScaled = minMaxScale(year, yearMin, yearMax);
  const doyScaled = doy / 366.0;

  // 1x4 input
  const inputTensor = tf.tensor2d(
    [[latScaled, lonScaled, yearScaled, doyScaled]],
    [1, 4]
  );

  // Load the saved model
  const model = await tf.loadLayersModel("file://./wildfire-model/model.json");

  // Model has 2 outputs => an array [probTensor, sizeTensor]
  const [probTensor, sizeTensor] = model.predict(inputTensor);

  // Convert from Tensor to JS values
  const probVal = (await probTensor.data())[0]; // Probability
  const sizeVal = (await sizeTensor.data())[0]; // Acres

  inputTensor.dispose();
  probTensor.dispose();
  sizeTensor.dispose();

  return { probability: probVal, size: sizeVal };
}

async function run() {
  const latTest = 29.650169372558594;
  const lonTest = -82.34162139892578;
  const yearTest = 2026; // example future date
  const doyTest = 200; // ~July 19

  const result = await predictFire(latTest, lonTest, yearTest, doyTest);
  console.log(`Probability of Fire: ${(result.probability * 100).toFixed(2)}%`);
  console.log(`Predicted Fire Size: ${result.size.toFixed(2)} acres`);
}

run();
