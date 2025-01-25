// train.js

const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const { parse } = require("csv-parse");

/**
 * Reads firedata.csv (the actual wildfires), parses columns, returns array of objects.
 */
async function loadPositiveData() {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream("firedata.csv") // Adjust filename/path if needed
      .pipe(parse({ columns: true, relax_quotes: true }))
      .on("data", (row) => {
        // We only keep rows that have the needed fields
        if (
          row.FIRE_SIZE &&
          row.LATITUDE &&
          row.LONGITUDE &&
          row.FIRE_YEAR &&
          row.DISCOVERY_DOY
        ) {
          rows.push({
            FIRE_SIZE: parseFloat(row.FIRE_SIZE),
            LATITUDE: parseFloat(row.LATITUDE),
            LONGITUDE: parseFloat(row.LONGITUDE),
            FIRE_YEAR: parseFloat(row.FIRE_YEAR),
            DISCOVERY_DOY: parseFloat(row.DISCOVERY_DOY),
          });
        }
      })
      .on("end", () => resolve(rows))
      .on("error", (err) => reject(err));
  });
}

/**
 * Finds minimum and maximum in an array, avoiding call-stack overflow with large arrays.
 */
function findMinMax(arr) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return { min, max };
}

/**
 * Helper: uniform in [minVal, maxVal].
 */
function randomInRange(minVal, maxVal) {
  return Math.random() * (maxVal - minVal) + minVal;
}

/**
 * Helper: clamp a value into [minVal, maxVal].
 */
function clamp(value, minVal, maxVal) {
  return Math.max(minVal, Math.min(value, maxVal));
}

/**
 * Build a Set of known fire combos (lat, lon, year, doy).
 * We'll store them as strings, e.g. "lat,lon,year,doy"
 * lat/lon are stored with a certain decimal precision to avoid floating mismatch.
 */
function buildFireSet(posData) {
  const fireSet = new Set();
  for (const row of posData) {
    const key = `${row.LATITUDE.toFixed(4)},${row.LONGITUDE.toFixed(4)},${
      row.FIRE_YEAR
    },${row.DISCOVERY_DOY}`;
    fireSet.add(key);
  }
  return fireSet;
}

/**
 * Generate "local" negative samples around each positive record.
 * For each real fire event, we pick a random lat/lon offset in a small neighborhood,
 * and possibly shift the day-of-year slightly, to ensure it's "close" but not the same.
 */
function createLocalNegativeSamples(posData, numLocalNeg, bounding, fireSet) {
  const negatives = [];
  let attempts = 0;
  const maxAttempts = numLocalNeg * 5;

  // Offsets for lat/lon/day-of-year
  const LAT_OFFSET_DEG = 0.5; // +/- half a degree
  const LON_OFFSET_DEG = 0.5;
  const DOY_OFFSET = 15; // +/- 15 days

  while (negatives.length < numLocalNeg && attempts < maxAttempts) {
    attempts++;
    // Pick a random positive row
    const idx = Math.floor(Math.random() * posData.length);
    const ref = posData[idx];
    const latBase = ref.LATITUDE;
    const lonBase = ref.LONGITUDE;
    const yearBase = ref.FIRE_YEAR;
    const doyBase = ref.DISCOVERY_DOY;

    // random offsets
    const latOffset = (Math.random() * 2 - 1) * LAT_OFFSET_DEG; // [-offset..offset]
    const lonOffset = (Math.random() * 2 - 1) * LON_OFFSET_DEG;
    const doyOffset = Math.floor((Math.random() * 2 - 1) * DOY_OFFSET);

    // new lat/lon within bounding
    const newLat = clamp(latBase + latOffset, bounding.latMin, bounding.latMax);
    const newLon = clamp(lonBase + lonOffset, bounding.lonMin, bounding.lonMax);

    // keep the same year (or optionally small +/- year offset)
    const newYear = yearBase;

    // new DOY in [1..366]
    let newDoy = doyBase + doyOffset;
    if (newDoy < 1) newDoy = 1;
    if (newDoy > 366) newDoy = 366;

    // Check collision
    const key = `${newLat.toFixed(4)},${newLon.toFixed(
      4
    )},${newYear},${newDoy}`;
    if (!fireSet.has(key)) {
      negatives.push({
        FIRE_SIZE: 0.0,
        LATITUDE: newLat,
        LONGITUDE: newLon,
        FIRE_YEAR: newYear,
        DISCOVERY_DOY: newDoy,
      });
      fireSet.add(key); // so we don't pick it again
    }
  }
  return negatives;
}

/**
 * Generate "global" random negative samples anywhere in the bounding box.
 */
function createGlobalNegativeSamples(numGlobalNeg, bounding, fireSet) {
  const negatives = [];
  let attempts = 0;
  const maxAttempts = numGlobalNeg * 10;

  while (negatives.length < numGlobalNeg && attempts < maxAttempts) {
    attempts++;
    const lat = randomInRange(bounding.latMin, bounding.latMax);
    const lon = randomInRange(bounding.lonMin, bounding.lonMax);
    const year = Math.floor(
      randomInRange(bounding.yearMin, bounding.yearMax + 1)
    );
    const doy = Math.floor(randomInRange(1, 367));

    const key = `${lat.toFixed(4)},${lon.toFixed(4)},${year},${doy}`;
    if (!fireSet.has(key)) {
      negatives.push({
        FIRE_SIZE: 0.0,
        LATITUDE: lat,
        LONGITUDE: lon,
        FIRE_YEAR: year,
        DISCOVERY_DOY: doy,
      });
      fireSet.add(key);
    }
  }
  return negatives;
}

/**
 * Generate negative samples specifically from areas that never had any fire:
 *   1) We discretize the bounding box into lat/lon "cells".
 *   2) Check how many fires occurred in each cell (based on the real data).
 *   3) If a cell had zero fires, sample random points (lat/lon) from that cell
 *      for random years/days, label them negative.
 */
function createNeverFireNegatives(
  posData,
  bounding,
  fireSet,
  numSamplesWanted
) {
  const negatives = [];

  // 1) Build a map from cell_id -> count_of_fires
  //    We'll define a cell size. For example 1.0 deg in lat & lon.
  //    Adjust cell size as needed. Smaller = more refined, but more cells to handle.
  const LAT_CELL_SIZE = 1.0;
  const LON_CELL_SIZE = 1.0;

  // Helper to get cell indices
  function latToCell(lat) {
    return Math.floor((lat - bounding.latMin) / LAT_CELL_SIZE);
  }
  function lonToCell(lon) {
    return Math.floor((lon - bounding.lonMin) / LON_CELL_SIZE);
  }

  // We'll track minCellLat / maxCellLat / minCellLon / maxCellLon too
  const latRange = bounding.latMax - bounding.latMin;
  const lonRange = bounding.lonMax - bounding.lonMin;
  const cellCountLat = Math.ceil(latRange / LAT_CELL_SIZE);
  const cellCountLon = Math.ceil(lonRange / LON_CELL_SIZE);

  // Fire map: cell_id -> count
  const fireCellMap = new Map();

  for (const row of posData) {
    const cLat = latToCell(row.LATITUDE);
    const cLon = lonToCell(row.LONGITUDE);
    const cellId = `${cLat},${cLon}`;
    fireCellMap.set(cellId, (fireCellMap.get(cellId) || 0) + 1);
  }

  // 2) Find all cell IDs that have zero fires
  const zeroFireCells = [];
  for (let cLat = 0; cLat < cellCountLat; cLat++) {
    for (let cLon = 0; cLon < cellCountLon; cLon++) {
      const cellId = `${cLat},${cLon}`;
      if (!fireCellMap.has(cellId)) {
        zeroFireCells.push({ cLat, cLon });
      }
    }
  }

  if (zeroFireCells.length === 0) {
    console.log("No truly never-fire cells found. Skipping this step.");
    return negatives;
  }

  console.log(
    `Found ${zeroFireCells.length} lat/lon cells that never had a fire. Sampling from them...`
  );

  // 3) Now we sample from these never-fire cells until we reach numSamplesWanted
  //    or exhaust attempts
  let attempts = 0;
  const maxAttempts = numSamplesWanted * 10;

  while (negatives.length < numSamplesWanted && attempts < maxAttempts) {
    attempts++;
    // pick a random never-fire cell
    const idx = Math.floor(Math.random() * zeroFireCells.length);
    const { cLat, cLon } = zeroFireCells[idx];

    // sample lat within that cell
    const latCellMin = bounding.latMin + cLat * LAT_CELL_SIZE;
    const latCellMax = latCellMin + LAT_CELL_SIZE;
    const lat = randomInRange(latCellMin, latCellMax);
    // clamp in case the bounding is partial
    const latClamped = clamp(lat, bounding.latMin, bounding.latMax);

    // sample lon within that cell
    const lonCellMin = bounding.lonMin + cLon * LON_CELL_SIZE;
    const lonCellMax = lonCellMin + LON_CELL_SIZE;
    const lon = randomInRange(lonCellMin, lonCellMax);
    const lonClamped = clamp(lon, bounding.lonMin, bounding.lonMax);

    // pick a random year in the bounding range
    const year = Math.floor(
      randomInRange(bounding.yearMin, bounding.yearMax + 1)
    );
    // pick a random DOY
    const doy = Math.floor(randomInRange(1, 367));

    // check collision with known fires
    const key = `${latClamped.toFixed(4)},${lonClamped.toFixed(
      4
    )},${year},${doy}`;
    if (!fireSet.has(key)) {
      negatives.push({
        FIRE_SIZE: 0.0,
        LATITUDE: latClamped,
        LONGITUDE: lonClamped,
        FIRE_YEAR: year,
        DISCOVERY_DOY: doy,
      });
      fireSet.add(key);
    }
  }

  return negatives;
}

async function runTraining() {
  console.log("Loading positive (actual wildfire) data from firedata.csv...");
  const positiveData = await loadPositiveData();
  console.log("Rows of actual fire data loaded:", positiveData.length);

  if (positiveData.length === 0) {
    console.error("No valid data rows found! Exiting.");
    return;
  }

  // 1) Figure out bounding box/time range from the actual fire data
  const latValues = positiveData.map((d) => d.LATITUDE);
  const lonValues = positiveData.map((d) => d.LONGITUDE);
  const yearValues = positiveData.map((d) => d.FIRE_YEAR);

  const { min: latMin, max: latMax } = findMinMax(latValues);
  const { min: lonMin, max: lonMax } = findMinMax(lonValues);
  const { min: yearMin, max: yearMax } = findMinMax(yearValues);

  const bounding = { latMin, latMax, lonMin, lonMax, yearMin, yearMax };

  // 2) Build a Set of known fire combos
  const fireSet = buildFireSet(positiveData);

  // 3) Decide how many negative samples from each method
  //    For example: 1:1 total ratio with positives. Then split among local, global, never-fire.

  const totalPos = positiveData.length;
  const totalNeg = totalPos; // 1:1 ratio overall

  // Let's say: 40% local, 30% global, 30% never-fire
  const localNegCount = Math.floor(totalNeg * 0.4);
  const globalNegCount = Math.floor(totalNeg * 0.3);
  const neverFireNegCount = totalNeg - localNegCount - globalNegCount; // remainder

  console.log(
    `Will generate ${localNegCount} local, ${globalNegCount} global, and ${neverFireNegCount} never-fire negatives.`
  );

  // 3a) Local negatives
  console.log("Creating local negative (no-fire) samples...");
  const localNegatives = createLocalNegativeSamples(
    positiveData,
    localNegCount,
    bounding,
    fireSet
  );
  console.log(`Local negatives: ${localNegatives.length}`);

  // 3b) Global random negatives
  console.log("Creating global negative (no-fire) samples...");
  const globalNegatives = createGlobalNegativeSamples(
    globalNegCount,
    bounding,
    fireSet
  );
  console.log(`Global negatives: ${globalNegatives.length}`);

  // 3c) Never-fire region negatives
  console.log("Creating never-fire region negative samples...");
  const neverFireNegatives = createNeverFireNegatives(
    positiveData,
    bounding,
    fireSet,
    neverFireNegCount
  );
  console.log(`Never-fire negatives: ${neverFireNegatives.length}`);

  // Combine them
  const negativeData = [
    ...localNegatives,
    ...globalNegatives,
    ...neverFireNegatives,
  ];
  console.log(`Total negative samples: ${negativeData.length}`);

  // 4) Combine positives + negatives into one dataset
  const fullData = positiveData.concat(negativeData);

  console.log("Total combined dataset size:", fullData.length);
  console.log(
    `Positives: ${positiveData.length}, Negatives: ${negativeData.length}`
  );

  // Cleanup arrays we no longer need
  positiveData.length = 0;
  negativeData.length = 0;

  // 5) Convert to Tensors: X = (lat, lon, year, doy), y = [class, size]
  function minMaxScale(value, min, max) {
    return max === min ? 0.0 : (value - min) / (max - min);
  }

  const XAll = [];
  const yClassAll = [];
  const ySizeAll = [];

  for (const row of fullData) {
    const latScaled = minMaxScale(row.LATITUDE, latMin, latMax);
    const lonScaled = minMaxScale(row.LONGITUDE, lonMin, lonMax);
    const yearScaled = minMaxScale(row.FIRE_YEAR, yearMin, yearMax);
    const doyScaled = row.DISCOVERY_DOY / 366.0; // 1..366 => ~0..1

    XAll.push([latScaled, lonScaled, yearScaled, doyScaled]);
    // classification: 1 if fire_size > 0
    yClassAll.push(row.FIRE_SIZE > 0 ? 1 : 0);
    // regression: actual fire size
    ySizeAll.push(row.FIRE_SIZE);
  }

  // Convert to Tensors
  const XTensor = tf.tensor2d(XAll, [XAll.length, 4]);
  const yClassTensor = tf.tensor2d(yClassAll, [yClassAll.length, 1]);
  const ySizeTensor = tf.tensor2d(ySizeAll, [ySizeAll.length, 1]);

  // 6) Split train/val
  const trainIndices = [];
  const valIndices = [];
  for (let i = 0; i < XAll.length; i++) {
    if (Math.random() < 0.8) trainIndices.push(i);
    else valIndices.push(i);
  }

  const XTrain = tf.gather(XTensor, trainIndices);
  const yClassTrain = tf.gather(yClassTensor, trainIndices);
  const ySizeTrain = tf.gather(ySizeTensor, trainIndices);

  const XVal = tf.gather(XTensor, valIndices);
  const yClassVal = tf.gather(yClassTensor, valIndices);
  const ySizeVal = tf.gather(ySizeTensor, valIndices);

  // 7) Build a two-headed model: classification + regression
  const input = tf.input({ shape: [4] });
  const dense1 = tf.layers
    .dense({ units: 64, activation: "relu" })
    .apply(input);
  const dense2 = tf.layers
    .dense({ units: 32, activation: "relu" })
    .apply(dense1);

  const classOut = tf.layers
    .dense({ units: 1, activation: "sigmoid", name: "classOut" })
    .apply(dense2);

  const sizeOut = tf.layers
    .dense({ units: 1, activation: "linear", name: "sizeOut" })
    .apply(dense2);

  const model = tf.model({ inputs: input, outputs: [classOut, sizeOut] });

  // 8) Compile with separate losses for each output
  model.compile({
    optimizer: "adam",
    loss: ["binaryCrossentropy", "meanSquaredError"],
    lossWeights: [1.0, 0.1], // reduce regression weight
    metrics: ["accuracy"], // applies to classification output
  });

  console.log("Starting training...");
  await model.fit(XTrain, [yClassTrain, ySizeTrain], {
    validationData: [XVal, [yClassVal, ySizeVal]],
    epochs: 50,
    batchSize: 1024,
  });

  console.log("Training complete. Saving model to ./wildfire-model ...");
  await model.save("file://./wildfire-model");
  console.log("Model saved successfully!");

  // Cleanup
  XTensor.dispose();
  yClassTensor.dispose();
  ySizeTensor.dispose();
  XTrain.dispose();
  XVal.dispose();
  yClassTrain.dispose();
  yClassVal.dispose();
  ySizeTrain.dispose();
  ySizeVal.dispose();

  console.log("All done.");
}

// Entry point
runTraining().catch((e) => console.error(e));
