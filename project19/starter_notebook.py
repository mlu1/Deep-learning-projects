# Barbados Plot Automation - Complete Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import os
import glob
import warnings
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# Optional: raster <-> polygon via scikit-image
try:
    from skimage.draw import polygon as draw_polygon
    from skimage import measure
except ImportError:
    raise ImportError("scikit-image is required: pip install scikit-image")

# --- Step 1: Load and preprocess metadata ---
train_csv = "Train.csv"
df = pd.read_csv(train_csv)
df['geometry'] = df['geometry'].apply(wkt.loads)

# Convert 3D polygons to 2D coords
def to_2d(geom):
    if isinstance(geom, Polygon):
        return [(x, y) for x, y, *_ in geom.exterior.coords]
    if isinstance(geom, MultiPolygon):
        pts = []
        for p in geom.geoms:
            pts.extend([(x, y) for x, y, *_ in p.exterior.coords])
        return pts
    return []

df['polygon_coords'] = df['geometry'].apply(to_2d)

# Load blind-test IDs
test_ids = pd.read_csv("Test.csv")
test_set = set(test_ids['ID'].astype(str))

# Exclude test IDs from training pool
train_val = df[~df['ID'].astype(str).isin(test_set)].reset_index(drop=True)
train_df, val_df = train_test_split(train_val, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_ids)}")

# --- Step 2: Constants ---
BOUNDS = (40600, 42600, 66500, 71000)   # Update to match full extent
IMG_SIZE = (512, 512)
BATCH = 8
IMAGE_DIR = "data/"  # Adjust as needed

# --- Step 3: Utilities ---
def rasterize_polygon(coords, size=IMG_SIZE, bounds=BOUNDS):
    mask = np.zeros(size, np.uint8)
    if not coords: return mask
    minx, maxx, miny, maxy = bounds
    xs = [(x-minx)/(maxx-minx)*(size[1]-1) for x,y in coords]
    ys = [(1-(y-miny)/(maxy-miny))*(size[0]-1) for x,y in coords]
    rr, cc = draw_polygon(ys, xs, mask.shape)
    mask[rr, cc] = 1
    return mask


def load_img(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img, np.float32)/255.0

# Resolve image path for a given ID
def resolve_img(pid):
    for pattern in [f"{pid}.jpg", f"anonymised_{pid}.jpg"]:
        p = os.path.join(IMAGE_DIR, pattern)
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(IMAGE_DIR, f"*{pid}*.jpg"))
    return hits[0] if hits else None

# --- Step 4: tf.data pipelines ---
def dataset_generator(df_split):
    for _, row in df_split.iterrows():
        img_path = resolve_img(row['ID'])
        if not img_path:
            warnings.warn(f"Missing image for {row['ID']}")
            continue
        img = load_img(img_path)
        mask = rasterize_polygon(row['polygon_coords'])[..., None]
        yield img, mask


def make_ds(df_split):
    ds = tf.data.Dataset.from_generator(
        lambda: dataset_generator(df_split),
        output_signature=(
            tf.TensorSpec((*IMG_SIZE,3), tf.float32),
            tf.TensorSpec((*IMG_SIZE,1), tf.float32)
        ))
    return ds.shuffle(500).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df)
val_ds   = make_ds(val_df)

# --- Step 5: U-Net model ---
def build_unet():
    inp = layers.Input((*IMG_SIZE,3))
    def conv_block(x,f): return layers.Conv2D(f,3,padding='same',activation='relu')(
                                  layers.Conv2D(f,3,padding='same',activation='relu')(x))
    def enc(x,f): c=conv_block(x,f); return c, layers.MaxPooling2D()(c)
    def dec(x,s,f): x=layers.Conv2DTranspose(f,2,2,padding='same')(x); x=layers.Concatenate()([x,s]); return conv_block(x,f)
    c1,p1=enc(inp,64); c2,p2=enc(p1,128); c3,p3=enc(p2,256); c4,p4=enc(p3,512)
    bn=conv_block(p4,1024)
    u1=dec(bn,c4,512); u2=dec(u1,c3,256); u3=dec(u2,c2,128); u4=dec(u3,c1,64)
    out=layers.Conv2D(1,1,activation='sigmoid')(u4)
    return models.Model(inp,out)

model = build_unet()
model.compile('adam','binary_crossentropy',['accuracy'])
model.summary()

# --- Step 6: Training ---
callbacks_list = [
    callbacks.ModelCheckpoint('best.h5', save_best_only=True),
    callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
]
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks_list)
model.load_weights('best.h5')

# --- Step 7: Sanity check prediction ---
for imgs, masks in val_ds.take(1):
    preds = model.predict(imgs)
    binp = (preds>0.3).astype(np.uint8)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(imgs[0]); plt.title('Input')
    plt.subplot(1,3,2); plt.imshow(masks[0,...,0],cmap='gray'); plt.title('GT')
    plt.subplot(1,3,3); plt.imshow(binp[0,...,0],cmap='gray'); plt.title('Pred')
    plt.show()
    break

# --- Step 8: Inference utils ---
def predict_mask(pid):
    p=resolve_img(pid); m=model.predict(load_img(p)[None])[0,...,0]; return (m>0.3).astype(np.uint8)

def mask_to_polygons(mask):
    conts=measure.find_contours(mask,0.5)
    out=[]
    h,w=mask.shape
    minx,maxx,miny,maxy=BOUNDS
    for cnt in conts:
        pts=[(c/(w-1)*(maxx-minx)+minx,(1-r/(h-1))*(maxy-miny)+miny) for r,c in cnt]
        if len(pts)>2: out.append(pts)
    return out

# --- Step 9: Submission ---
def build_test_submission(test_ids_df, output='submission.csv'):
    rows=[]
    for pid in test_ids_df['ID']:
        meta=df[df['ID'].astype(str)==pid].iloc[0] if pid in df['ID'].astype(str).values else None
        target=meta['Surveyed For'] if meta is not None else ''
        cert=meta['Certified date'] if meta is not None else ''
        area=meta['Total Area'] if meta is not None else ''
        unit=meta['Unit of Measurement'] if meta is not None else ''
        parish=meta['Parish'] if meta is not None else ''
        lt=meta['LT Num'] if meta is not None else ''
        geom=mask_to_polygons(predict_mask(pid))
        rows.append({
            'ID':pid,'TargetSurvey':target,'Certified date':cert,
            'Total Area':area,'Unit of Measurement':unit,'Parish':parish,
            'LT Num':lt,'geometry':geom[0] if geom else []
        })
    pd.DataFrame(rows)[['ID','TargetSurvey','Certified date','Total Area',
                        'Unit of Measurement','Parish','LT Num','geometry']].to_csv(output,index=False)
    print(f'Saved → {output} with {len(rows)} rows')

# Execute submission build
build_test_submission(test_ids)

