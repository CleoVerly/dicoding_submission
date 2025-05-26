# Image Classification with Custom CNN

Proyek ini melakukan klasifikasi gambar menggunakan model CNN yang dibangun dengan TensorFlow. Model dilatih menggunakan data gambar dari dataset Intel Image Classification, lalu disimpan dalam format SavedModel, TFLite, dan TFJS.

## 📂 Struktur Folder

```
├── saved_model_format/    # Model dalam format SavedModel
├── tflite/
│   └── model.tflite       # Model TFLite
├── tfjs/
│   ├── model.json         # Model TFJS
│   └── group1-shard*      # Bobot model untuk TFJS
├── README.md              # Dokumentasi proyek ini
├── Notebook.ipynb         # Notebook utama (training + evaluasi)
```

## Menjalankan Proyek

### 1. Instalasi Dependensi

```bash
pip install tensorflow matplotlib seaborn pandas
```

### 2. Jalankan Notebook

Buka `Notebook.ipynb` dan jalankan semua sel. Proyek mencakup:
- Preprocessing dan augmentasi data
- Training CNN model tanpa transfer learning
- Penyimpanan model ke SavedModel, TFLite, dan TFJS
- Inference dari gambar validasi

## Inference Contoh

```python
from keras.layers import TFSMLayer

inference_layer = TFSMLayer("saved_model_format", call_endpoint="serving_default")

for images, labels in val_ds.take(1):
    test_img = images[0]
    input_img = tf.expand_dims(test_img, axis=0)
    raw_output = inference_layer(input_img)
    pred = list(raw_output.values())[0].numpy()

predicted_index = np.argmax(pred)
predicted_class = class_names[predicted_index]
confidence = np.max(pred)

plt.imshow(test_img.numpy() / 255.0)
plt.title(f"Prediksi: {predicted_class} ({confidence:.2f})")
```

Contoh output:
```
Prediksi: forest (confidence: 0.89)
Label Asli: forest
```

## 📦 Format Model

| Format     | Platform                          |
|------------|-----------------------------------|
| SavedModel | TensorFlow Serving / Python       |
| TFLite     | Android / Embedded Devices        |
| TFJS       | Web / JavaScript Application      |

---

## 👤 Author

- Nama: Muhammad Agusriansyah
- Email: misteragus46@gmail.com / mc224d5y1338@student.devacademy.id
