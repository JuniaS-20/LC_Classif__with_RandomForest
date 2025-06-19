import rasterio
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
import json
import os
from rasterio.enums import ColorInterp
from scipy.ndimage import median_filter
import joblib

# === 1. Reading data ===
image_paths = ['S2A_MSIL2A_20230706T080611_N0509_R078_T35LNH_20230706T120603.vrt','S2A_MSIL2A_20230127T081211_N0509_R078_T35LNG_20230127T120321.vrt']  # Ajouter d'autres rasters ici si besoin
shapefile_path = 'labels.shp'

gdf = gpd.read_file(shapefile_path)

# === 2. Extraction of features and labels ===
with rasterio.open(image_paths[0]) as src_ref:
    bands = src_ref.read()
    transform = src_ref.transform
    crs_raster = src_ref.crs
    profile = src_ref.profile

if gdf.crs != crs_raster: # === Reproject the shapefile to the same CRS as the raster ===
    gdf = gdf.to_crs(crs_raster)

features, labels = [], []
for idx, row in gdf.iterrows():
    x, y = row.geometry.x, row.geometry.y
    label = row['label']
    row_idx, col_idx = ~transform * (x, y)
    row_idx, col_idx = int(row_idx), int(col_idx)
    if 0 <= row_idx < bands.shape[1] and 0 <= col_idx < bands.shape[2]:
        pixel = bands[:, row_idx, col_idx]
        features.append(pixel)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

# === 3. pre-processing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. cross-validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
print("Validation croisee - Accuracy moyenne :", np.mean(cv_scores))

# === 5. final training ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "trained_model_RF.joblib")
joblib.dump(scaler, "scaler_RF.joblib")
print("→ Modele et scaler sauvegardes.")

y_pred = model.predict(X_test)
print("Accuracy test:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# === 6. Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matrice de Confusion")
plt.xlabel("Prevu")
plt.ylabel("Reel")
plt.tight_layout()
plt.savefig("confusion_matrix_RF.png")
plt.close()

# === 7. Batch classification ===
colormap = {
    0: (255, 255, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 165, 0)
}

for image_path in image_paths:
    with rasterio.open(image_path) as src:
        bands = src.read()
        h, w = bands.shape[1], bands.shape[2]
        transform = src.transform
        crs = src.crs

    full_image_2D = bands.reshape(bands.shape[0], -1).T
    full_image_scaled = scaler.transform(full_image_2D)

    prediction_flat = []
    batch_size = 100000
    for i in tqdm(range(0, full_image_scaled.shape[0], batch_size), desc=f"Classification {os.path.basename(image_path)}"):
        batch = full_image_scaled[i:i+batch_size]
        prediction_flat.extend(model.predict(batch))

    prediction_2D = np.array(prediction_flat).reshape(h, w).astype(rasterio.uint8)
    prediction_filtered = median_filter(prediction_2D, size=3)

    # Export raster
    profile_out = {
        'driver': 'GTiff',
        'height': prediction_filtered.shape[0],
        'width': prediction_filtered.shape[1],
        'count': 1,
        'dtype': rasterio.uint8,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"classified_{basename}_RF.tif"

    with rasterio.open(out_path, 'w', **profile_out) as dst:
        dst.write(prediction_filtered, 1)
        dst.write_colormap(1, colormap)
        dst.set_band_description(1, "Classification")
        dst.colorinterp = [ColorInterp.palette]
    print(f"→ Raster classifie exporte : {out_path}")

    # PNG preview
    cmap = mcolors.ListedColormap([np.array(colormap[i])/255 for i in sorted(colormap.keys())])
    bounds = [i - 0.5 for i in range(1, len(colormap) + 2)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(prediction_filtered, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title("Classification RandomForest")
    preview_path = f"preview_{basename}.png"
    plt.savefig(preview_path, bbox_inches='tight')
    plt.close()
    print(f"→ Apercu PNG exporte : {preview_path}")

# === 8. QML and Legend ===
qml_content = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories=\"AllStyleCategories\" version=\"3.16\">
  <renderer-v2 type=\"paletted\" forceraster=\"0\" enableorderby=\"0\">
    <rastershader>
      <colorrampshader colorRampType=\"EXACT\" clip=\"0\">
"""
for class_id, color in colormap.items():
    qml_content += f'        <item alpha="255" value="{class_id}" label="Classe {class_id}" color="{color[0]},{color[1]},{color[2]},255"/>\n'
qml_content += """      </colorrampshader>
    </rastershader>
    <classificationMinMaxOrigin>MinMaxFullExtent</classificationMinMaxOrigin>
  </renderer-v2>
  <layerTransparency/>
  <customproperties/>
</qgis>
"""
with open("classified_RF.qml", "w") as f:
    f.write(qml_content)
print("→ Fichier QML exporte : classified_RF.qml")

legend = []
for class_id, color in colormap.items():
    legend.append({
        "type": "Feature",
        "properties": {
            "class_id": class_id,
            "label": f"Classe {class_id}",
            "color": f"rgb({color[0]},{color[1]},{color[2]})"
        },
        "geometry": None
    })
legend_geojson = {
    "type": "FeatureCollection",
    "features": legend
}
with open("classified_RF_legend.geojson", "w") as f:
    json.dump(legend_geojson, f, indent=2)
print("→ Legende vectorielle exportee : classified_RF_legend.geojson")
