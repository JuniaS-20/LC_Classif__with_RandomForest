import rasterio
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from rasterio.enums import ColorInterp

# === 1. Lecture des données ===
image_path = 'S2A_MSIL2A_20230706T080611_N0509_R078_T35LNH_20230706T120603.vrt'
shapefile_path = 'labels.shp'

gdf = gpd.read_file(shapefile_path)

with rasterio.open(image_path) as src:
    bands = src.read()
    transform = src.transform
    crs_raster = src.crs
    profile = src.profile

# Reprojection des points si nécessaire
if gdf.crs != crs_raster:
    gdf = gdf.to_crs(crs_raster)

# === 2. Extraction des features et labels ===
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

# === 3. Entraînement et test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# === 4. Export matrice confusion en image ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matrice de Confusion - RandomForest")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrix_RF.png")
plt.close()

# === 5. Appliquer à toute l’image ===
h, w = bands.shape[1], bands.shape[2]
full_image_2D = bands.reshape(bands.shape[0], -1).T  # (n_pixels, n_bands)

# Utilisation de tqdm pour la barre de progression
prediction_flat = []
batch_size = 100000
for i in tqdm(range(0, full_image_2D.shape[0], batch_size), desc="Classification de l'image"):
    batch = full_image_2D[i:i+batch_size]
    prediction_flat.extend(model.predict(batch))
prediction_flat = np.array(prediction_flat)

prediction_2D = prediction_flat.reshape(h, w).astype(rasterio.uint8)

# === 6. Export du raster classifié avec colormap ===
# profile.update({
#     'count': 1,
#     'dtype': rasterio.uint8,
#     'compress': 'lzw'
# })

# Nouveau profil propre pour l'export || appel à  ColorInterp
profile_out = {
    'driver': 'GTiff',
    'height': prediction_2D.shape[0],
    'width': prediction_2D.shape[1],
    'count': 1,
    'dtype': rasterio.uint8,
    'crs': src.crs,
    'transform': src.transform,
    'compress': 'lzw'
}


out_path = "classified_RF.tif"
with rasterio.open(out_path, 'w', **profile_out) as dst:
    dst.write(prediction_2D, 1)
    # Définition de la colormap
    colormap = {
        0: (255, 255, 255), # Inconu - Clair
        1: (0, 255, 0),    # Végétation - Vert
        2: (0, 0, 255),    # Eau - Bleu
        3: (255, 165, 0)   # Sol - Orange
    }
    dst.write_colormap(1, colormap)
    dst.set_band_description(1, "Classification")
    dst.colorinterp = [ColorInterp.palette]

print(f"→ Raster classifié exporté : {out_path}")

# === 7. Aperçu visuel coloré (PNG) ===
import matplotlib.colors as mcolors

# Création de la colormap pour l'aperçu
cmap = mcolors.ListedColormap([np.array(colormap[i])/255 for i in sorted(colormap.keys())])
bounds = [i - 0.5 for i in range(1, len(colormap) + 2)]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 10))
plt.imshow(prediction_2D, cmap=cmap, norm=norm)
plt.axis('off')
plt.title("Classification RandomForest")
plt.savefig("classified_RF_preview.png", bbox_inches='tight')
plt.close()
print("→ Aperçu PNG exporté : classified_RF_preview.png")

# === 8. Génération du fichier QML pour QGIS ===
qml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.16">
  <renderer-v2 type="paletted" forceraster="0" enableorderby="0">
    <rastershader>
      <colorrampshader colorRampType="EXACT" clip="0">
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
print("→ Fichier QML exporté : classified_RF.qml")

# === 9. Création de la légende vectorielle (GeoJSON) ===
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
print("→ Légende vectorielle exportée : classified_RF_legend.geojson")
