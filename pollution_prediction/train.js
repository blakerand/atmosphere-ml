/**
 * train.js
 *
 * Trains a pollution model on CSV data containing columns like:
 *   Date, O3 Mean, O3 1st Max Value, O3 1st Max Hour, O3 AQI,
 *   CO Mean, CO 1st Max Value, CO 1st Max Hour, CO AQI,
 *   SO2 Mean, SO2 1st Max Value, SO2 1st Max Hour, SO2 AQI,
 *   NO2 Mean, NO2 1st Max Value, NO2 1st Max Hour, NO2 AQI,
 *   Longitude_Geocoded, Latitude_Geocoded
 *
 * We parse each row, convert the Date into (scaledYear, cycMonthSin, cycMonthCos, cycDaySin, cycDayCos),
 * keep the Longitude/Latitude as numeric features, and use the 16 pollution metrics as targets.
 *
 * Saves the trained model to ./pollution-model
 */

const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { parse } = require("csv-parse");

/**
 * Utility: cyclical encoding for month/day-of-year.
 * By converting them to sine/cosine, the model sees them as cyclical features
 * (January is "close" to December, day 365 is close to day 1, etc.).
 */
function encodeCyclical(value, maxValue) {
  // e.g. for day-of-year in [1..365],
  // or month in [1..12], etc.
  const sinVal = Math.sin((2 * Math.PI * value) / maxValue);
  const cosVal = Math.cos((2 * Math.PI * value) / maxValue);
  return [sinVal, cosVal];
}

/**
 * Given a YYYY-MM-DD (or similar) string, parse out:
 *   - scaledYear (relative to minYear, maxYear found in data)
 *   - cyclical month
 *   - cyclical dayOfYear
 *
 * We'll fill in actual minYear / maxYear once we read the data.
 * For now, we define placeholders. We'll fix them after scanning the CSV.
 */
let globalMinYear = 1990;
let globalMaxYear = 2030;
function parseDateToFeatures(dateString) {
  const date = new Date(dateString);
  const fullYear = date.getUTCFullYear();
  const yearScaled =
    (fullYear - globalMinYear) / (globalMaxYear - globalMinYear);

  // month in [1..12]
  const month = date.getUTCMonth() + 1;
  const [monthSin, monthCos] = encodeCyclical(month, 12);

  // day of year in [1..365 or 366]
  const startOfYear = new Date(fullYear, 0, 1); // Jan 1
  const dayIndex = Math.floor((date - startOfYear) / (1000 * 60 * 60 * 24)) + 1;
  // clamp dayIndex to [1..366] just in case
  const dayOfYear = Math.max(1, Math.min(dayIndex, 366));
  const [daySin, dayCos] = encodeCyclical(dayOfYear, 366);

  return [yearScaled, monthSin, monthCos, daySin, dayCos];
}

/**
 * We read the CSV once to:
 *   1) find minYear and maxYear overall,
 *   2) build arrays of raw data so we can do a single pass,
 *   3) parse them into numeric features (with final known minYear/maxYear).
 */
async function loadDataset(csvPath) {
  return new Promise((resolve, reject) => {
    const allRows = [];
    const readStream = fs
      .createReadStream(csvPath)
      .pipe(parse({ columns: true, relax_quotes: true }));

    readStream
      .on("data", (row) => {
        if (!row.Date || !row.Latitude_Geocoded || !row.Longitude_Geocoded) {
          return; // skip incomplete
        }
        allRows.push(row);
      })
      .on("end", () => resolve(allRows))
      .on("error", (err) => reject(err));
  });
}

/**
 * Once we have all rows from CSV, we figure out the minYear and maxYear.
 */
function findMinMaxYear(rows) {
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < rows.length; i++) {
    const dateVal = rows[i].Date;
    const year = new Date(dateVal).getUTCFullYear();
    if (year < minY) minY = year;
    if (year > maxY) maxY = year;
  }
  return [minY, maxY];
}

/**
 * Convert each CSV row into an input feature array + target array.
 * input shape: [ yearScaled, cycMonthSin, cycMonthCos, cycDaySin, cycDayCos, longitude, latitude ]
 * target shape: 16 values:
 *   0: O3 Mean
 *   1: O3 1st Max Value
 *   2: O3 1st Max Hour
 *   3: O3 AQI
 *   4: CO Mean
 *   5: CO 1st Max Value
 *   6: CO 1st Max Hour
 *   7: CO AQI
 *   8: SO2 Mean
 *   9: SO2 1st Max Value
 *   10: SO2 1st Max Hour
 *   11: SO2 AQI
 *   12: NO2 Mean
 *   13: NO2 1st Max Value
 *   14: NO2 1st Max Hour
 *   15: NO2 AQI
 *
 * Note: Some rows might have missing columns or parse issues – skip or default them.
 */
function rowToFeatureTarget(row) {
  // parse input features
  const [yearScaled, monthSin, monthCos, daySin, dayCos] = parseDateToFeatures(
    row.Date
  );
  const longitude = parseFloat(row.Longitude_Geocoded);
  const latitude = parseFloat(row.Latitude_Geocoded);

  // parse target columns
  // Safely parse or skip
  function safeParse(str) {
    const val = parseFloat(str);
    return isNaN(val) ? 0.0 : val; // or skip the row entirely if you want
  }
  const O3_Mean = safeParse(row["O3 Mean"]);
  const O3_1st_Max_Value = safeParse(row["O3 1st Max Value"]);
  const O3_1st_Max_Hour = safeParse(row["O3 1st Max Hour"]);
  const O3_AQI = safeParse(row["O3 AQI"]);

  const CO_Mean = safeParse(row["CO Mean"]);
  const CO_1st_Max_Value = safeParse(row["CO 1st Max Value"]);
  const CO_1st_Max_Hour = safeParse(row["CO 1st Max Hour"]);
  const CO_AQI = safeParse(row["CO AQI"]);

  const SO2_Mean = safeParse(row["SO2 Mean"]);
  const SO2_1st_Max_Value = safeParse(row["SO2 1st Max Value"]);
  const SO2_1st_Max_Hour = safeParse(row["SO2 1st Max Hour"]);
  const SO2_AQI = safeParse(row["SO2 AQI"]);

  const NO2_Mean = safeParse(row["NO2 Mean"]);
  const NO2_1st_Max_Value = safeParse(row["NO2 1st Max Value"]);
  const NO2_1st_Max_Hour = safeParse(row["NO2 1st Max Hour"]);
  const NO2_AQI = safeParse(row["NO2 AQI"]);

  const inputFeatures = [
    yearScaled,
    monthSin,
    monthCos,
    daySin,
    dayCos,
    longitude,
    latitude,
  ];
  const targetValues = [
    O3_Mean,
    O3_1st_Max_Value,
    O3_1st_Max_Hour,
    O3_AQI,
    CO_Mean,
    CO_1st_Max_Value,
    CO_1st_Max_Hour,
    CO_AQI,
    SO2_Mean,
    SO2_1st_Max_Value,
    SO2_1st_Max_Hour,
    SO2_AQI,
    NO2_Mean,
    NO2_1st_Max_Value,
    NO2_1st_Max_Hour,
    NO2_AQI,
  ];

  return { inputFeatures, targetValues };
}

async function runTraining() {
  try {
    const csvPath = "./pollution_2000_2023_with_coords.csv"; // Change to your actual CSV file path
    console.log("Loading dataset from", csvPath);
    let rows = await loadDataset(csvPath);
    console.log("Loaded rows:", rows.length);

    // 1) Find minYear and maxYear from the entire dataset
    const [minY, maxY] = findMinMaxYear(rows);
    globalMinYear = minY;
    globalMaxYear = maxY;
    console.log(
      `globalMinYear=${globalMinYear}, globalMaxYear=${globalMaxYear}`
    );

    // 2) Convert each row to feature + target
    const allFeatures = [];
    const allTargets = [];
    for (const row of rows) {
      const { inputFeatures, targetValues } = rowToFeatureTarget(row);
      // We can add filtering if the row is incomplete; for now just push
      allFeatures.push(inputFeatures);
      allTargets.push(targetValues);
    }

    // Convert to Tensors
    const X = tf.tensor2d(allFeatures);
    const y = tf.tensor2d(allTargets);

    console.log("Feature shape:", X.shape, "Target shape:", y.shape);
    // E.g. shape [numSamples, 7] for X, [numSamples, 16] for y

    // 3) Split train/val
    const allIndices = Array.from(allFeatures.keys());
    const trainIndices = [];
    const valIndices = [];
    for (let i = 0; i < allIndices.length; i++) {
      if (Math.random() < 0.8) trainIndices.push(i);
      else valIndices.push(i);
    }
    const XTrain = tf.gather(X, trainIndices);
    const XVal = tf.gather(X, valIndices);
    const yTrain = tf.gather(y, trainIndices);
    const yVal = tf.gather(y, valIndices);

    // 4) Build model
    // More layers + more units to capture complexity
    const input = tf.input({ shape: [7] }); // [yearScaled, cycMonthSin, cycMonthCos, cycDaySin, cycDayCos, lon, lat]

    let xDense = tf.layers
      .dense({ units: 128, activation: "relu" })
      .apply(input);
    xDense = tf.layers.dense({ units: 128, activation: "relu" }).apply(xDense);
    xDense = tf.layers.dropout({ rate: 0.2 }).apply(xDense); // dropout for regularization
    xDense = tf.layers.dense({ units: 64, activation: "relu" }).apply(xDense);
    xDense = tf.layers.dropout({ rate: 0.2 }).apply(xDense);
    xDense = tf.layers.dense({ units: 32, activation: "relu" }).apply(xDense);

    // final output layer - we have 16 regression targets
    const output = tf.layers
      .dense({ units: 16, activation: "linear" })
      .apply(xDense);

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
      optimizer: tf.train.adam(),
      loss: "meanSquaredError", // or "huberLoss", etc.
      metrics: ["mae"],
    });

    model.summary();

    // 5) Train
    console.log("Starting training...");
    const earlyStop = tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience: 5,
    });

    await model.fit(XTrain, yTrain, {
      validationData: [XVal, yVal],
      epochs: 100,
      batchSize: 256,
      callbacks: [earlyStop],
    });

    // 6) Save model
    const savePath = "file://" + path.join(__dirname, "pollution-model");
    console.log("Saving model to", savePath);
    await model.save(savePath);
    console.log("Model saved successfully!");

    // Cleanup
    X.dispose();
    y.dispose();
    XTrain.dispose();
    XVal.dispose();
    yTrain.dispose();
    yVal.dispose();

    console.log("All done.");
  } catch (err) {
    console.error("Error in runTraining:", err);
  }
}

// Entry point
if (require.main === module) {
  runTraining();
}
