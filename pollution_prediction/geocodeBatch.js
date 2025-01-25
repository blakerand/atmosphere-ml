/**
 * geocodeBatch.js
 *
 * This script:
 * 1. Reads a large CSV (pollution_2000_2023.csv).
 * 2. (Test mode) Processes only a small batch (so we don't do the entire file right now).
 * 3. Sends a POST to the Census Batch Geocoder (returntype=locations).
 * 4. Receives a CSV response containing matched lat/lons.
 * 5. Merges those lat/lons back into the original data.
 * 6. Appends them to a final CSV (pollution_2000_2023_with_coords.csv).
 */

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const csvParser = require("csv-parser");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

// -------------------------
// 1. CONFIGURABLE SETTINGS
// -------------------------

// Input CSV file:
const INPUT_CSV = path.join(__dirname, "pollution_2000_2023.csv");

// Output CSV file (the final):
const OUTPUT_CSV = path.join(__dirname, "pollution_2000_2023_with_coords.csv");

// The chunk (batch) size. For real usage, you'd likely set 9500 or so.
// For testing (as requested), we'll set a very small batch size (e.g. 2).
const BATCH_SIZE = 9500; // TEST BATCH ONLY

// Census Batch Geocoder endpoint (for "locations", i.e. lat/lon only)
const CENSUS_BATCH_URL =
  "https://geocoding.geo.census.gov/geocoder/locations/addressbatch";

// Benchmark to use (4 = "Public_AR_Current" as of this writing)
const CENSUS_BENCHMARK = "4"; // or 'Public_AR_Current'

// --------------
// 2. MAIN LOGIC
// --------------

async function main() {
  console.log("Reading CSV...");

  // We will store rows as objects with all original columns, plus:
  //  - __uniqueId (to correlate with Census response)
  //  - __street, __city, __state, __zip (for sending to Census)
  //  - __originalLongitude, __originalLatitude (parsed from the last column)
  const allRows = [];

  // Read CSV with csv-parser
  await new Promise((resolve, reject) => {
    fs.createReadStream(INPUT_CSV)
      .pipe(csvParser())
      .on("data", (row) => {
        // row is an object keyed by the CSV headers
        // Example headers from your sample:
        //  Index
        //  Date
        //  Address
        //  State
        //  County
        //  City
        //  O3 Mean
        //  ...
        //  NO2 AQI
        //  Longitude (which in your file actually is "Longitude,Latitude" combined)
        //  (Possibly a separate "Latitude" or none if your CSV lumps them together)

        // Let's create a unique ID. If your CSV has "Index", use that; otherwise create one.
        const uniqueId = row["Index"] || String(allRows.length + 1);

        // The Address, City, State, ZIP for the geocoder
        // (We do NOT change how we submit to the API.)
        const address = row["Address"] ? row["Address"].trim() : "";
        const city = row["City"] ? row["City"].trim() : "";
        const state = row["State"] ? row["State"].trim() : "";
        const zip = row["ZIP"] ? row["ZIP"].trim() : ""; // Only if your CSV has a ZIP col

        // -- IMPORTANT FIX --
        // If your CSV's final column lumps longitude & latitude in one string
        // e.g.  "-112.046355090882,33.458651868657"
        // we parse it now so it's not mistaken for an address.
        let originalLongitude = "";
        let originalLatitude = "";
        if (row["Longitude"]) {
          // 'Longitude' in your CSV actually might contain "lon,lat"
          const val = row["Longitude"].replace(/"/g, "").trim(); // remove any surrounding quotes
          if (val.includes(",")) {
            const [lon, lat] = val.split(",");
            originalLongitude = lon || "";
            originalLatitude = lat || "";
          } else {
            // If your CSV does indeed have separate columns named "Longitude" and "Latitude",
            // then adapt accordingly. But based on your sample, we do the split.
            originalLongitude = val;
          }
        }
        // If there's a separate "Latitude" column, you could read it as well:
        if (row["Latitude"]) {
          originalLatitude = row["Latitude"].replace(/"/g, "").trim();
        }

        // Store everything in allRows
        allRows.push({
          ...row,
          __uniqueId: uniqueId,
          __street: address,
          __city: city,
          __state: state,
          __zip: zip,
          __originalLongitude: originalLongitude,
          __originalLatitude: originalLatitude,
        });
      })
      .on("end", () => {
        console.log(`Finished reading CSV. Total rows read: ${allRows.length}`);
        resolve();
      })
      .on("error", reject);
  });

  // Process all rows, not just a test subset
  const totalChunks = Math.ceil(allRows.length / BATCH_SIZE);
  console.log(`Total rows to process: ${allRows.length}`);
  console.log(`We will process ${totalChunks} chunk(s).`);

  // Prepare our output CSV writer.
  // We want to keep original headers plus new columns: __longitude, __latitude
  const firstRow = allRows[0];
  const originalHeaders = Object.keys(firstRow)
    // Filter out any of our internal fields (starts with "__")
    .filter((k) => !k.startsWith("__"));

  const csvWriter = createCsvWriter({
    path: OUTPUT_CSV,
    header: [
      // Original columns in the same order
      ...originalHeaders.map((h) => ({ id: h, title: h })),
      // Two new columns from the geocode results
      { id: "__longitude", title: "Longitude_Geocoded" },
      { id: "__latitude", title: "Latitude_Geocoded" },
    ],
    append: false, // overwrite if it exists
  });

  const finalResults = [];

  for (let i = 0; i < totalChunks; i++) {
    const start = i * BATCH_SIZE;
    const end = Math.min(start + BATCH_SIZE, allRows.length);
    const chunkRows = allRows.slice(start, end);

    console.log(
      `\nProcessing chunk #${i + 1} (rows ${start} to ${end - 1})...`
    );

    // Build the CSV in memory for Census batch
    const batchCsv = buildBatchCsv(chunkRows);

    let batchResultCsv;
    try {
      // Send to the Census batch geocoder
      batchResultCsv = await sendBatchToCensus(batchCsv);
    } catch (err) {
      console.error(`Error in chunk #${i + 1}`, err);
      throw err;
    }

    // Parse the Census CSV response into a map: { uniqueId: { longitude, latitude } }
    const resultsMap = parseCensusResponseCsv(batchResultCsv);

    // Attach the returned lat/lon to each row
    for (const row of chunkRows) {
      const match = resultsMap[row.__uniqueId];
      if (match) {
        row.__longitude = match.longitude;
        row.__latitude = match.latitude;
      } else {
        row.__longitude = "";
        row.__latitude = "";
      }
      finalResults.push(row);
    }
  }

  // Write everything to the final CSV
  console.log("\nWriting final CSV (test batch only) with lat/lon included...");
  await csvWriter.writeRecords(finalResults);

  console.log(`Done! Wrote: ${OUTPUT_CSV}`);
}

// --------------
// 3. HELPER FUNCS
// --------------

/**
 * Build the CSV string (in memory) for the batch geocoder.
 * Format: uniqueId,StreetAddress,City,State,Zip
 */
function buildBatchCsv(rows) {
  const lines = [];
  for (const r of rows) {
    // The Census requires columns in the order: uniqueId,Street,City,State,ZIP
    const lineArr = [
      r.__uniqueId,
      r.__street,
      r.__city,
      r.__state,
      r.__zip || "",
    ];

    // Minimal CSV escaping: if a field contains a comma or quote, wrap in quotes and escape quotes
    const quoted = lineArr.map((field) => {
      if (!field) return "";
      const needsQuote = field.includes(",") || field.includes('"');
      let out = field;
      if (needsQuote) {
        out = out.replace(/"/g, '""'); // escape existing quotes
        out = `"${out}"`;
      }
      return out;
    });

    lines.push(quoted.join(","));
  }
  return lines.join("\n");
}

/**
 * Send one batch CSV to the Census batch geocoder (returntype=locations).
 * Returns the raw CSV string from the geocoder.
 */
async function sendBatchToCensus(batchCsvString) {
  const form = new FormData();
  form.append("addressFile", batchCsvString, {
    filename: "batch.csv",
    contentType: "text/csv",
  });
  form.append("benchmark", CENSUS_BENCHMARK);

  const response = await axios.post(CENSUS_BATCH_URL, form, {
    headers: form.getHeaders(),
    responseType: "text",
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  });
  return response.data; // The raw CSV string
}

/**
 * Parse the Census batch geocoder's CSV response into a lookup object:
 *   resultsMap[uniqueId] = { longitude, latitude }
 *
 * Sample response row:
 *   "1234","1645 E ROOSEVELT ST, PHOENIX, AZ, 85006","1645 E ROOSEVELT ST, PHOENIX, AZ, 85006","Exact","-112.043703","33.458472","1111","L"
 */
function parseCensusResponseCsv(csvText) {
  const resultsMap = {};
  const lines = csvText.split(/\r?\n/).filter(Boolean);

  for (const line of lines) {
    const parts = splitCsvLine(line);
    if (!parts[0]) continue;

    const uid = parts[0].replace(/^"|"$/g, ""); // uniqueId
    const test = parts[4] ? parts[4].replace(/^"|"$/g, "") : "";
    const coordinates = parts[5] ? parts[5].replace(/^"|"$/g, "") : "";

    const longitude = coordinates.split(",")[0];
    const latitude = coordinates.split(",")[1];

    resultsMap[uid] = { longitude: longitude, latitude: latitude };
  }

  return resultsMap;
}

/**
 * A minimal CSV split that accounts for double quotes as a field enclosure.
 */
function splitCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      // Check if next char is also a quote => escaped quote
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (c === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += c;
    }
  }
  // push the last field
  result.push(current);

  return result;
}

// Run it
main().catch((err) => {
  console.error("Fatal error in script:", err);
  process.exit(1);
});
