# Land Cover Classification with RandomForest (Python)

## 🧠 Overview

This project performs land cover classification using a multiband raster (e.g., Sentinel-2) and labeled points in a shapefile. It uses a `RandomForestClassifier` with preprocessing, cross-validation, and applies the model to the full image (and optionally to a batch of rasters). The output includes classified raster, preview image, QML style, and a GeoJSON legend.

---

## 📦 Requirements

* Python >= 3.8
* rasterio
* geopandas
* numpy
* matplotlib
* seaborn
* tqdm
* scikit-learn
* scipy
* joblib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How it works

1. **Read training shapefile and multiband raster**
2. **Extract features (spectral bands) from raster at labeled points**
3. **Preprocess features with standard scaling**
4. **Train and evaluate a Random Forest model** (with 5-fold stratified cross-validation)
5. **Apply the trained model to classify the full raster image**
6. **Apply median filtering to reduce "salt-and-pepper" effect**
7. **Export results**:

   * Classified raster (`.tif`)
   * Preview image (`.png`)
   * QGIS QML style file (`.qml`)
   * GeoJSON legend (`.geojson`)
   * Saved model and scaler (`.joblib`)

---

## 🗃️ Inputs

* `labels.shp` — Shapefile with labeled point geometries and a `label` column.
* `*.vrt` or `*.tif` — Multiband raster(s) for classification.

---

## 🧪 Outputs

* `classified_*.tif` — Classified raster.
* `classified_RF_preview.png` — PNG visualization.
* `classified_RF.qml` — QGIS styling file.
* `classified_RF_legend.geojson` — Vector legend.
* `trained_model_RF.joblib` / `scaler_RF.joblib` — Trained model and scaler.

---

## 📁 Batch Processing

To classify multiple rasters, simply add their paths to the `image_paths` list.

```python
image_paths = [
    "raster1.vrt",
    "raster2.vrt",
    "raster3.vrt"
]
```

Each will be classified and exported with its own output files.

---

## 📌 Notes

* A 3x3 median filter is applied to reduce classification noise.
* The QML file enables direct style loading in QGIS.
* The `label` values in the shapefile must correspond to integer class IDs.

---

## 📜 License

Feel free to use and adapt.

---


## Contributions
Contributions to this repository are welcome. If you find any bugs or have suggestions for improvements, feel free to submit an issue or pull request.

land-cover-classification

## 👤 Author

Developed by Junior Muyumba as part of land cover mapping pipeline automation.
Feel free to contribute or fork!
