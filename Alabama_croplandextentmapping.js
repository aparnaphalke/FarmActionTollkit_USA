/***************************************************************************************
Farm Action Toolkit (FAcT) — USA (Alabama ASD) — Crop/Non-crop Mapping (HLSS30)

Purpose
- Produce a crop/non-crop map for ONE Alabama ASD (Agricultural Statistics District) for ONE
  label year using HLS S30 (Sentinel-2-like surface reflectance).
- Predictors are biweekly NDVI + NDWI (+ optional NDBI) composites for Jan–Dec.
- Gap-filling is used to make predictors wall-to-wall within the ASD.

Key idea
- Labels come from USDA CDL (cropland band), with a user-defined list of “non-crop” CDL classes.
- NLCD is used to FORCE non-crop in forests, urban areas, water, and wetlands
  (both in labels and in the final prediction).

Data
- Predictors: NASA/HLS/HLSS30/v002 (S30), bands B2,B3,B4,B8,B11 + Fmask
- Labels: USDA/NASS/CDL/<Y>
- Exclusion mask: USGS/NLCD_RELEASES/2021_REL/NLCD/<NLCD_YEAR> landcover

Outputs
- Map layers for QA (labels, masks, coverage, prediction, difference map)
- Exports to Google Drive (Tasks tab):
  1) pred_clean raster (crop=1, noncrop=0) for the ASD
  2) train/test sample tables (CSV)
  3) testPred table (CSV)
  4) metrics table (CSV)

How to run (Google Earth Engine Code Editor)
1) Open https://code.earthengine.google.com/
2) Paste this script, set SELECT_STASD and Y
3) Click Run
4) Inspect layers + Console prints
5) Run Exports from the Tasks tab

Notes
- Fmask in HLS is a BITMASK (values like 128, 132, 160 are combinations of bits).
  Do NOT use f.eq(0). Decode bits with bitwiseAnd.
- Gap filling:
  - Primary composite uses images inside the target biweekly window
  - Fallback composite uses images within ±FILL_DAYS of the window start
  - Filled = primary.unmask(fallback).unmask(0)

***************************************************************************************/

// =====================================================================
// SETTINGS
// =====================================================================

// ASD selection
var SELECT_STASD = 110;      // 110,120,130,140,150,160
var Y = 2024;                // label year (CDL)
var STEP_DAYS = 14;          // biweekly windows

// Sampling / model
var SCALE = 30;
var POINTS_PER_CLASS = 1200;
var TRAIN_FRACTION = 0.7;
var N_TREES = 250;
var TILE_SCALE = 4;

// Gap-filling window (days around each window start)
var FILL_DAYS = 30;

// Masks / options
var NLCD_YEAR = 2021;        // latest official NLCD in this GEE release path
var USE_NDBI = true;         // reduce built-up confusion

// HLSS30 Fmask controls (bitmask)
var MASK_CIRRUS   = true;    // Bit 0
var MASK_CLOUD    = true;    // Bit 1
var MASK_ADJACENT = true;    // Bit 2 (adjacent to cloud/shadow) — set false if too aggressive
var MASK_SHADOW   = true;    // Bit 3
var KEEP_SNOW     = true;    // Bit 4 (snow/ice) — keep by default to avoid winter=all masked
var KEEP_WATER    = true;    // Bit 5 (water) — keep by default


// =====================================================================
// 1) ASD SELECT
// =====================================================================

var asdAll = ee.FeatureCollection('projects/ee-aparnapm16/assets/Alabama_ASD_2012_20m')
  .filter(ee.Filter.eq('STATE', '01'));

var asdOne = asdAll.filter(ee.Filter.eq('STASD_N', SELECT_STASD));
var asdGeom = asdOne.geometry();

print('Selected STASD_N:', SELECT_STASD);
print('Selected ASD (first feature):', asdOne.first());

Map.setOptions('HYBRID');
Map.centerObject(asdGeom, 9);
Map.addLayer(
  asdOne.style({color:'00ffff', width:4, fillColor:'00000000'}),
  {},
  'Selected ASD',
  true
);


// =====================================================================
// 2) LABELS (CDL year Y) + NLCD FORCING
// =====================================================================

var cdl = ee.Image('USDA/NASS/CDL/' + Y).clip(asdGeom);
var cropland = cdl.select('cropland');

// CDL classes you want to treat as NONCROP for this use-case.
// (Includes grass/pasture etc. and horticulture/tree crops.)
var nonCropValues = ee.List([
  81, 82, 83, 87, 88, 92,
  111, 112,
  121, 122, 123, 124,
  63, 141, 142, 143,
  152, 131,
  190, 195,

  // horticulture / tree crops -> treat as NONCROP for your use-case
  66, 67, 68, 69, 70, 71,
  72, 74, 75, 76, 77, 204
]);

var nonCropMask = cropland.remap(
  nonCropValues,
  ee.List.repeat(1, nonCropValues.length()),
  0
).rename('nonCrop').toByte();

// CDL-only label (no NLCD cleaning)
var cropMask1 = nonCropMask.not().rename('label').toByte();
Map.addLayer(
  cropMask1,
  {min:0, max:1, palette:['ff0000','00ff00']},
  'CropMask1 (CDL-only)',
  false
);

// NLCD exclusion mask (force noncrop in forests/urban/water/wetlands)
var nlcd = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/' + NLCD_YEAR)
  .select('landcover')
  .clip(asdGeom);

var excludeClasses = ee.List([
  // Forest
  41, 42, 43,
  // Urban
  21, 22, 23, 24,
  // Water / Wetlands
  11, 90, 95
]);

// IMPORTANT: unmask(0) so NLCD no-data doesn't break the exclude logic
var nlcdExcludeMask = nlcd.remap(
  excludeClasses,
  ee.List.repeat(1, excludeClasses.length()),
  0
).rename('exclude')
 .unmask(0)
 .toByte();

Map.addLayer(
  nlcdExcludeMask,
  {min:0, max:1, palette:['000000','ff00ff']},
  'NLCD Exclude (1=forest/urban/water)',
  false
);

// Optional explicit forest mask (also unmasked)
var nlcdForest = nlcd.remap([41,42,43], [1,1,1], 0)
  .rename('forest')
  .unmask(0)
  .toByte();

Map.addLayer(
  nlcdForest,
  {min:0, max:1, palette:['000000','00ff00']},
  'NLCD Forest (green)',
  false
);

// Force NLCD excluded pixels to NONCROP (0) instead of masking them out
var cropMask = cropMask1.where(nlcdExcludeMask.eq(1), 0).rename('label').toByte();

Map.addLayer(
  cropMask,
  {min:0, max:1, palette:['ff0000','00ff00']},
  'CropMask (CDL + NLCD forced noncrop)',
  true
);

// Label sanity checks
print('Crop pixel sum (label=1) CDL-only:',
  cropMask1.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: asdGeom,
    scale: SCALE,
    maxPixels: 1e13
  })
);

print('Crop pixel sum (label=1) CDL+NLCD:',
  cropMask.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: asdGeom,
    scale: SCALE,
    maxPixels: 1e13
  })
);


// =====================================================================
// 3) DATE HELPERS
// =====================================================================

function seasonStart(year){
  year = ee.Number(year);
  return ee.Date.fromYMD(year, 1, 1);
}
function seasonEnd(year){
  year = ee.Number(year);
  return ee.Date.fromYMD(year, 12, 31).advance(1, 'day');
}
function ymd(d){ return ee.Date(d).format('YYYY_MM_dd'); }


// =====================================================================
// 4) HLSS30 HELPERS (S30 only)
// =====================================================================
//
// Fmask bitmask (HLS doc / your screenshot):
// Bit 0: Cirrus
// Bit 1: Cloud
// Bit 2: Adjacent to cloud/shadow
// Bit 3: Cloud shadow
// Bit 4: Snow/ice
// Bit 5: Water
// Bits 6-7: Aerosol level (ignored here)
//
// IMPORTANT: Fmask values like 128,130,132... are combinations of bits.
// Do NOT test equality to 0. Use bitwise decoding.
//

function maskHLSS30(img){
  var f = img.select('Fmask').toUint16();

  var cirrus = f.bitwiseAnd(1 << 0).neq(0);
  var cloud  = f.bitwiseAnd(1 << 1).neq(0);
  var adj    = f.bitwiseAnd(1 << 2).neq(0);
  var shadow = f.bitwiseAnd(1 << 3).neq(0);
  var snow   = f.bitwiseAnd(1 << 4).neq(0);
  var water  = f.bitwiseAnd(1 << 5).neq(0);

  var good = ee.Image(1);

  if (MASK_CIRRUS)   good = good.and(cirrus.not());
  if (MASK_CLOUD)    good = good.and(cloud.not());
  if (MASK_ADJACENT) good = good.and(adj.not());
  if (MASK_SHADOW)   good = good.and(shadow.not());

  // If we do NOT want to keep snow/water, mask them too.
  if (!KEEP_SNOW)  good = good.and(snow.not());
  if (!KEEP_WATER) good = good.and(water.not());

  return img.updateMask(good);
}

function addIndices(img){
  var ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndwi = img.normalizedDifference(['B3','B8']).rename('NDWI');

  if (USE_NDBI) {
    var ndbi = img.normalizedDifference(['B11','B8']).rename('NDBI');
    return img.addBands([ndvi, ndwi, ndbi]);
  } else {
    return img.addBands([ndvi, ndwi]);
  }
}

// NOTE: Keeping the function name s2SeasonCollection() to minimize downstream changes.
// It now returns HLSS30 images (S30 only).
function s2SeasonCollection(year, geom){
  var start = seasonStart(year);
  var end   = seasonEnd(year);

  var base = ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(geom)
    .filterDate(start, end)
    .map(maskHLSS30);

  // Keep only needed reflectance bands + Fmask
  var keep = USE_NDBI
    ? ['B2','B3','B4','B8','B11','Fmask']
    : ['B2','B3','B4','B8','Fmask'];

  return base
    .select(keep)
    .map(addIndices);
}

function zeroIdx(){
  return USE_NDBI
    ? ee.Image.constant([0, 0, 0]).rename(['NDVI','NDWI','NDBI']).toFloat()
    : ee.Image.constant([0, 0]).rename(['NDVI','NDWI']).toFloat();
}

// Simple visible brightness proxy (lower is better)
function addCloudScore(img) {
  var vis = img.select(['B2','B3','B4']).reduce(ee.Reducer.mean()).rename('CLOUDSCORE');
  return img.addBands(vis);
}

// Choose the "best" pixel through time using qualityMosaic on a score band.
// Here: QS = -CLOUDSCORE (prefers darker/less haze in visible bands).
function bestComposite(col, geom) {
  var scored = col.map(addCloudScore);

  var withQS = scored.map(function(i){
    return i.addBands(i.select('CLOUDSCORE').multiply(-1).rename('QS'));
  });

  var comp = withQS.qualityMosaic('QS').clip(geom);

  // Recompute indices on the selected composite
  var ndvi = comp.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndwi = comp.normalizedDifference(['B3','B8']).rename('NDWI');

  if (USE_NDBI) {
    var ndbi = comp.normalizedDifference(['B11','B8']).rename('NDBI');
    return comp.addBands([ndvi, ndwi, ndbi], null, true)
      .select(['NDVI','NDWI','NDBI'])
      .toFloat();
  } else {
    return comp.addBands([ndvi, ndwi], null, true)
      .select(['NDVI','NDWI'])
      .toFloat();
  }
}


// =====================================================================
// 5) GAP-FILLED BIWEEKLY STACK (Jan–Dec, one year)
// =====================================================================
//
// GAP FILLING (wall-to-wall predictors):
// 1) PRIMARY: best composite within the target biweekly window
// 2) FALLBACK: best composite within ±FILL_DAYS of the window start
// 3) FILLED: primary → fallback → zeros (only if both missing)
//

function biweeklyStackOneYearFilled(year, geom){
  year = ee.Number(year);
  var start = seasonStart(year);
  var end   = seasonEnd(year);

  var col = s2SeasonCollection(year, geom);

  var nSteps = end.difference(start, 'day').divide(STEP_DAYS).ceil();
  var ks = ee.List.sequence(0, nSteps.subtract(1));

  var imgs = ee.ImageCollection(ks.map(function(k){
    k = ee.Number(k);
    var winStart = start.advance(k.multiply(STEP_DAYS), 'day');
    var winEnd   = winStart.advance(STEP_DAYS, 'day');

    // PRIMARY window composite
    var wcol = col.filterDate(winStart, winEnd);
    var primary = ee.Image(ee.Algorithms.If(
      wcol.size().gt(0),
      bestComposite(wcol, geom),
      zeroIdx().clip(geom)
    ));

    // FALLBACK window composite (±FILL_DAYS around window start)
    var fStart = winStart.advance(-FILL_DAYS, 'day');
    var fEnd   = winStart.advance( FILL_DAYS, 'day');
    var fcol   = col.filterDate(fStart, fEnd);
    var fallback = ee.Image(ee.Algorithms.If(
      fcol.size().gt(0),
      bestComposite(fcol, geom),
      zeroIdx().clip(geom)
    ));

    // Fill missing pixels
    var filled = primary.unmask(fallback).unmask(zeroIdx().clip(geom));

    // Stamp band names so each window is unique
    var stamp = ymd(winStart);
    return filled.rename(filled.bandNames().map(function(b){
      return ee.String(b).cat('_Y').cat(year.format()).cat('_BW_').cat(stamp);
    }));
  }));

  var stack = imgs.toBands().toFloat();

  // Clean band names produced by toBands() (removes leading numeric prefixes)
  var clean = stack.bandNames().map(function(b){
    return ee.String(b).replace('^[0-9]+_', '');
  });

  return stack.rename(clean);
}

var predictors = biweeklyStackOneYearFilled(Y, asdGeom);
print('Predictor band count:', predictors.bandNames().size());

// Coverage check (green=has data somewhere after filling)
var cov = predictors.reduce(ee.Reducer.firstNonNull()).mask().rename('cov');
Map.addLayer(cov, {min:0, max:1, palette:['red','green']}, 'Predictor coverage (green=filled)', false);


// =====================================================================
// 6) DISPLAY SAMPLE COMPOSITE (FIRST BIWEEK, FILLED)
// =====================================================================

var startY = seasonStart(Y);
var winStart0 = startY;
var winEnd0 = winStart0.advance(STEP_DAYS, 'day');

var col0 = s2SeasonCollection(Y, asdGeom).filterDate(winStart0, winEnd0);
print('HLSS30 first biweek count (masked):', col0.size());

// Primary0
var primary0 = ee.Image(ee.Algorithms.If(
  col0.size().gt(0),
  bestComposite(col0, asdGeom),
  zeroIdx().clip(asdGeom)
));

// Fallback0
var fStart0 = winStart0.advance(-FILL_DAYS, 'day');
var fEnd0   = winStart0.advance( FILL_DAYS, 'day');
var fcol0   = s2SeasonCollection(Y, asdGeom).filterDate(fStart0, fEnd0);

var fallback0 = ee.Image(ee.Algorithms.If(
  fcol0.size().gt(0),
  bestComposite(fcol0, asdGeom),
  zeroIdx().clip(asdGeom)
));

// Filled0
var filled0 = primary0.unmask(fallback0).unmask(zeroIdx().clip(asdGeom));

Map.addLayer(
  filled0.select('NDVI'),
  {min:0, max:0.8, palette:['brown','yellow','green']},
  'Sample NDVI (first biweek, FILLED)',
  true
);

Map.addLayer(
  filled0.select('NDWI'),
  {min:-0.5, max:0.5, palette:['brown','white','blue']},
  'Sample NDWI (first biweek, FILLED)',
  false
);

if (USE_NDBI) {
  Map.addLayer(
    filled0.select('NDBI'),
    {min:-0.5, max:0.5, palette:['blue','white','brown']},
    'Sample NDBI (first biweek, FILLED)',
    false
  );
}


// =====================================================================
// 7) BALANCED STRATIFIED SAMPLES
// =====================================================================

var trainingImage = predictors.addBands(cropMask);

var samples = trainingImage.stratifiedSample({
  numPoints: POINTS_PER_CLASS,
  classBand: 'label',
  region: asdGeom,
  scale: SCALE,
  seed: 42,
  geometries: true,
  classValues: [0, 1],
  classPoints: [POINTS_PER_CLASS, POINTS_PER_CLASS],
  dropNulls: true,
  tileScale: TILE_SCALE
});

var split = samples.randomColumn('rand', 42);
var train = split.filter(ee.Filter.lt('rand', TRAIN_FRACTION));
var test  = split.filter(ee.Filter.gte('rand', TRAIN_FRACTION));

print('Train size:', train.size());
print('Test size:', test.size());
print('Train crop:', train.filter(ee.Filter.eq('label', 1)).size());
print('Train noncrop:', train.filter(ee.Filter.eq('label', 0)).size());


// =====================================================================
// 8) TRAIN RF + PREDICT (+ NLCD FORCED NONCROP)
// =====================================================================

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  minLeafPopulation: 2,
  seed: 42
}).train({
  features: train,
  classProperty: 'label',
  inputProperties: predictors.bandNames()
});

var pred = predictors.classify(rf).rename('pred').toByte();
Map.addLayer(pred, {min:0, max:1, palette:['ff0000','00ff00']}, 'RF Prediction (raw)', false);

// HARD ENFORCE: excluded pixels must be noncrop in final output
var pred_clean = pred.where(nlcdExcludeMask.eq(1), 0).rename('pred_clean').toByte();

// Extra bulletproof: force forest explicitly too
pred_clean = pred_clean.where(nlcdForest.eq(1), 0).rename('pred_clean').toByte();

Map.addLayer(pred_clean, {min:0, max:1, palette:['ff0000','00ff00']}, 'RF Prediction CLEAN', true);


// =====================================================================
// 9) DIAGNOSTICS (OPTIONAL)
// =====================================================================

var fp = cropMask.eq(0).and(pred_clean.eq(1)).selfMask();     // ref=0, pred=1
var forestFP = fp.and(nlcdForest.eq(1)).selfMask();          // only forest commission

Map.addLayer(fp, {palette:['ff0000']}, 'Commission FP (ref=0 pred=1)', false);
Map.addLayer(forestFP, {palette:['ff00ff']}, 'Forest Commission (FP ∩ forest)', false);


// =====================================================================
// 10) DIFFERENCE MAP (ref=cropMask1, cls=pred_clean)
// =====================================================================

var ref = cropMask1.rename('ref');
var cls = pred_clean.rename('cls');

var diffVis = ee.Image().expression(
  "(ref == 0 && cls == 0) ? 0" +     // true noncrop
  ": (ref == 1 && cls == 1) ? 2" +   // true crop
  ": (ref == 0 && cls == 1) ? 1" +   // false crop
  ": -1",                            // missed crop
  { ref: ref, cls: cls }
).rename('diff');

Map.addLayer(
  diffVis.clip(asdOne),
  {
    min: -1,
    max: 2,
    palette: [
      '0000ff',  // -1 missed crop (blue)
      '000000',  //  0 true noncrop (black)
      'ff0000',  //  1 false crop (red)
      '00ff00'   //  2 true crop (green)
    ]
  },
  'Difference Map (CDL-only label vs Classified)',
  true
);


// =====================================================================
// 11) ACCURACY (HOLDOUT SAMPLES)
// =====================================================================

var testPred = test.classify(rf);
var cm = testPred.errorMatrix('label', 'classification');

print('Confusion Matrix (holdout):', cm);
print('Overall Accuracy:', cm.accuracy());
print("User's Accuracy:", cm.consumersAccuracy());
print("Producer's Accuracy:", cm.producersAccuracy());
print('Kappa:', cm.kappa());


// Map-vs-Map accuracy vs CDL-only (no NLCD) reference
var evalSamples = cropMask1.rename('ref_cdl_only')
  .addBands(pred_clean.rename('cls'))
  .sample({
    region: asdGeom,
    scale: SCALE,
    numPixels: 100000,
    seed: 202,
    tileScale: TILE_SCALE,
    geometries: false
  });

var cm_map = evalSamples.errorMatrix('ref_cdl_only', 'cls');
print('Confusion Matrix (CDL-only ref -> pred_clean):', cm_map);
print('OA (map-vs-map):', cm_map.accuracy());
print('Kappa (map-vs-map):', cm_map.kappa());


// =====================================================================
// 12) EXPORTS
// =====================================================================

var tag = 'STASD_' + SELECT_STASD + '_Y' + Y + '_BWidx_Filled_NLCDforced_HLSS30';

// Prediction raster
Export.image.toDrive({
  image: pred_clean,
  description: 'AL_ASD_RF_CropNonCrop_' + tag,
  fileNamePrefix: 'AL_ASD_RF_CropNonCrop_' + tag,
  region: asdGeom,
  scale: SCALE,
  maxPixels: 1e13
});

// Training / testing samples
Export.table.toDrive({
  collection: train,
  description: 'AL_ASD_Train_' + tag,
  fileNamePrefix: 'AL_ASD_Train_' + tag,
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: test,
  description: 'AL_ASD_Test_' + tag,
  fileNamePrefix: 'AL_ASD_Test_' + tag,
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: testPred,
  description: 'AL_ASD_TestPred_' + tag,
  fileNamePrefix: 'AL_ASD_TestPred_' + tag,
  fileFormat: 'CSV'
});

// Metrics table (single row)
var metrics = ee.Feature(null, {
  stasd_n: SELECT_STASD,
  label_year_cdl: Y,
  step_days: STEP_DAYS,
  fill_days: FILL_DAYS,
  use_ndbi: USE_NDBI,
  points_per_class: POINTS_PER_CLASS,
  train_fraction: TRAIN_FRACTION,
  n_trees: N_TREES,
  overall_accuracy_holdout: cm.accuracy(),
  kappa_holdout: cm.kappa()
});

Export.table.toDrive({
  collection: ee.FeatureCollection([metrics]),
  description: 'AL_ASD_Metrics_' + tag,
  fileNamePrefix: 'AL_ASD_Metrics_' + tag,
  fileFormat: 'CSV'
});
