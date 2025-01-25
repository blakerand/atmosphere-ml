/**
 * predict.js
 *
 * Loads the trained pollution model from `./pollution-model`
 * and performs predictions given:
 *   - date (e.g., '2025-01-01')
 *   - longitude
 *   - latitude
 *
 * Usage example:
 *   node predict.js 2025-01-01 -112.046355 33.458652
 */

const tf = require("@tensorflow/tfjs-node");
const path = require("path");

// IMPORTANT: Must match training code
const globalMinYear = 1990; // set to your training min
const globalMaxYear = 2030; // set to your training max

/**
 * cyclical encoding
 */
function encodeCyclical(value, maxValue) {
  const sinVal = Math.sin((2 * Math.PI * value) / maxValue);
  const cosVal = Math.cos((2 * Math.PI * value) / maxValue);
  return [sinVal, cosVal];
}

/**
 * parseDateToFeatures matches the logic used during training:
 *   - yearScaled
 *   - cycMonthSin, cycMonthCos
 *   - cycDaySin, cycDayCos
 */
function parseDateToFeatures(dateString) {
  const date = new Date(dateString);
  const fullYear = date.getUTCFullYear();
  const yearScaled =
    (fullYear - globalMinYear) / (globalMaxYear - globalMinYear);

  // month in [1..12]
  const month = date.getUTCMonth() + 1;
  const [monthSin, monthCos] = encodeCyclical(month, 12);

  // day of year in [1..366]
  const startOfYear = new Date(fullYear, 0, 1);
  const dayIndex = Math.floor((date - startOfYear) / (1000 * 60 * 60 * 24)) + 1;
  const dayOfYear = Math.max(1, Math.min(dayIndex, 366));
  const [daySin, dayCos] = encodeCyclical(dayOfYear, 366);

  return [yearScaled, monthSin, monthCos, daySin, dayCos];
}

async function runPredict(dateStr, longitude, latitude) {
  // Load model from disk
  const modelPath = path.join(__dirname, "pollution-model", "model.json");
  const model = await tf.loadLayersModel(`file://${modelPath}`);

  // Prepare input using same transformations as training
  const [yearScaled, monthSin, monthCos, daySin, dayCos] =
    parseDateToFeatures(dateStr);

  // Features = [ yearScaled, cycMonthSin, cycMonthCos, cycDaySin, cycDayCos, longitude, latitude ]
  const inputTensor = tf.tensor2d([
    [yearScaled, monthSin, monthCos, daySin, dayCos, longitude, latitude],
  ]);

  // Perform prediction
  const prediction = model.predict(inputTensor);
  const result = prediction.dataSync(); // Float32Array of length 16

  // Dispose Tensors
  inputTensor.dispose();
  prediction.dispose();

  // Return or print results
  return result;
}

// If called directly from command line
if (require.main === module) {
  // e.g. node predict.js 2025-01-01 -112.046355 33.458652
  const dateStr = process.argv[2] || "2025-01-01";
  // const longitude = parseFloat(process.argv[3] || "-112.046355");
  // const latitude = parseFloat(process.argv[4] || "33.458652");

  const longitude = parseFloat(process.argv[3] || "-122.9382"); // Longview, WA
  const latitude = parseFloat(process.argv[4] || "46.1382");
  runPredict(dateStr, longitude, latitude).then((outputs) => {
    // The outputs array is in the order:
    // [
    //   0: O3 Mean
    //   1: O3 1st Max Value
    //   2: O3 1st Max Hour
    //   3: O3 AQI
    //   4: CO Mean
    //   5: CO 1st Max Value
    //   6: CO 1st Max Hour
    //   7: CO AQI
    //   8: SO2 Mean
    //   9: SO2 1st Max Value
    //   10: SO2 1st Max Hour
    //   11: SO2 AQI
    //   12: NO2 Mean
    //   13: NO2 1st Max Value
    //   14: NO2 1st Max Hour
    //   15: NO2 AQI
    // ]

    console.log(
      `\nPrediction for date=${dateStr}, lon=${longitude}, lat=${latitude}:`
    );
    console.log(`O3 Mean:             ${outputs[0].toFixed(5)}`);
    console.log(`O3 1st Max Value:    ${outputs[1].toFixed(5)}`);
    console.log(`O3 1st Max Hour:     ${outputs[2].toFixed(5)}`);
    console.log(`O3 AQI:              ${outputs[3].toFixed(5)}`);
    console.log(`CO Mean:             ${outputs[4].toFixed(5)}`);
    console.log(`CO 1st Max Value:    ${outputs[5].toFixed(5)}`);
    console.log(`CO 1st Max Hour:     ${outputs[6].toFixed(5)}`);
    console.log(`CO AQI:              ${outputs[7].toFixed(5)}`);
    console.log(`SO2 Mean:            ${outputs[8].toFixed(5)}`);
    console.log(`SO2 1st Max Value:   ${outputs[9].toFixed(5)}`);
    console.log(`SO2 1st Max Hour:    ${outputs[10].toFixed(5)}`);
    console.log(`SO2 AQI:             ${outputs[11].toFixed(5)}`);
    console.log(`NO2 Mean:            ${outputs[12].toFixed(5)}`);
    console.log(`NO2 1st Max Value:   ${outputs[13].toFixed(5)}`);
    console.log(`NO2 1st Max Hour:    ${outputs[14].toFixed(5)}`);
    console.log(`NO2 AQI:             ${outputs[15].toFixed(5)}`);

    // Example: Calculate overall US AQI from the 4 pollutant AQIs
    const overallAQI = Math.max(
      outputs[3],
      outputs[7],
      outputs[11],
      outputs[15]
    );
    console.log(`\nOverall US AQI:      ${overallAQI.toFixed(5)}`);
  });
}

module.exports = { runPredict };
