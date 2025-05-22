# Laporan Proyek Machine Learning - Josua Sianturi

## **Domain Proyek**

Pasar mobil bekas di Indonesia mengalami pertumbuhan pesat seiring meningkatnya permintaan masyarakat terhadap kendaraan terjangkau. Namun, proses penentuan harga mobil bekas masih sangat subjektif dan rentan terhadap kesalahan estimasi.

Masalah ini penting untuk diselesaikan karena dapat menyebabkan kerugian baik bagi penjual maupun pembeli. Dengan menerapkan machine learning, perusahaan dapat membangun sistem prediksi harga berbasis data historis, sehingga harga mobil bisa ditentukan secara objektif dan efisien.

Menurut riset OLX Autos dan Katadata Insight Center (2023), lebih dari 60% konsumen di Indonesia mempertimbangkan harga sebagai faktor utama dalam memilih mobil bekas. Oleh karena itu, sistem prediktif berbasis data sangat relevan dalam meningkatkan kepercayaan dan efisiensi pasar mobil bekas.

Penelitian oleh Pandey, Rastogi, dan Singh (2020) juga menunjukkan bahwa algoritma **Random Forest** sangat efektif dalam memprediksi harga jual mobil, mengingat kemampuannya menangani variabel non-linear dan kompleksitas fitur secara bersamaan.

**Referensi:**

- OLX Autos & Katadata Insight Center. (2023). _Tren Pasar Mobil Bekas di Indonesia_.
- Wulandari, Rahajeng. (2025). Mobil Bekas Masih Akan Jadi Pilihan Masyarakat di 2025. Katadata.co.id. [Tautan](https://otomotif.katadata.co.id/mobil/mobil-bekas-masih-akan-jadi-pilihan-masyarakat-di-2025-14306)
- Pandey, Abhishek; Rastogi, Vanshika; Singh, Sanika. (2020). _Car’s Selling Price Prediction using Random Forest Machine Learning Algorithm_. 5th Int. Conference on Next Generation Computing Technologies (NGCT-2019). SSRN: [https://ssrn.com/abstract=3702236](https://ssrn.com/abstract=3702236) atau [DOI](http://dx.doi.org/10.2139/ssrn.3702236)

## Business Understanding

### Problem Statements

- Bagaimana cara mengidentifikasi fitur kendaraan bekas yang paling berpengaruh terhadap harga jual?
- Bagaimana cara memprediksi harga wajar sebuah mobil bekas berdasarkan spesifikasi tertentu?
- Bagaimana perusahaan dapat menghindari penetapan harga yang terlalu rendah atau terlalu tinggi secara sistematis?

### Goals

- Mengidentifikasi fitur-fitur utama yang memengaruhi harga kendaraan.
- Mengembangkan model prediktif yang mampu memperkirakan harga mobil bekas dengan akurat.
- Menyediakan sistem penilaian harga otomatis untuk mendukung pengambilan keputusan oleh tim penjualan.

### Solution Statements

- Menggunakan beberapa model regresi seperti K-Nearest Neighbors, Random Forest, dan XGBoost untuk membandingkan performa.
- Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan akurasi prediksi.
- Evaluasi performa model menggunakan metrik RMSE dan R² Score.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah _Vehicle Dataset from CarDekho_ yang tersedia secara publik di Kaggle:  
🔗 [https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

Dataset ini berisi 4.340 data mobil bekas dengan berbagai fitur yang mencerminkan spesifikasi mobil dan informasi penjualan. Meski berasal dari India, banyak fitur dalam dataset ini yang tetap relevan dengan konteks pasar mobil bekas di Indonesia.

#### **Variabel-variabel pada dataset:**

- **name**: Nama lengkap kendaraan (merk dan model)
- **year**: Tahun produksi kendaraan
- **selling_price**: Harga jual mobil bekas (dalam INR)
- **km_driven**: Total jarak tempuh mobil dalam kilometer
- **fuel**: Jenis bahan bakar (Petrol, Diesel, CNG, LPG, Electric)
- **seller_type**: Jenis penjual (Dealer, Individual, Trustmark Dealer)
- **transmission**: Jenis transmisi (Manual atau Automatic)
- **owner**: Status kepemilikan (First Owner, Second Owner, Third Owner, Fourth & Above Owner, Test Drive Car)

#### Statistik Deskriptif

| Fitur          | Mean      | Std Dev    | Min   | 25%      | 50%     | 75%     | Max      |
|----------------|-----------|------------|--------|----------|---------|---------|----------|
| year           | 2013.09   | 4.22       | 1992   | 2011     | 2014    | 2016    | 2020     |
| selling_price  | 504127.30 | 578548.70  | 20000  | 208749.80| 350000  | 600000  | 8900000  |
| km_driven      | 66215.78  | 46644.10   | 1      | 35000    | 60000   | 90000   | 806599   |

- Tidak ditemukan missing value pada data.
- Outlier terdeteksi pada `selling_price` dan `km_driven` (nilai ekstrim seperti harga jual hingga ₹8.900.000 dan jarak tempuh lebih dari 800.000 km).
- Fitur-fitur memiliki skala yang sangat berbeda, sehingga diperlukan normalisasi atau standardisasi sebelum digunakan dalam pelatihan model machine learning.

## Data Preparation

### Konversi Tahun Produksi menjadi Usia Kendaraan

Tahun produksi kendaraan (`year`) diubah menjadi usia kendaraan (`age`) untuk memberikan informasi yang lebih relevan bagi model. Usia kendaraan dihitung berdasarkan tahun saat ini.

```python
from datetime import datetime
df.rename(columns={'year': 'age'}, inplace=True)
current_year = datetime.now().year
df['age'] = current_year - df['age']
```
### Encoding Fitur Kategorikal

Beberapa fitur kategorikal diubah ke format numerik agar bisa digunakan dalam model machine learning.

- `name`: Diubah menggunakan **Label Encoding**.
- `owner`: Diubah menggunakan **Ordinal Encoding** berdasarkan tingkat kepemilikan.
- `fuel`, `seller_type`, dan `transmission`: Diubah menggunakan **One-Hot Encoding**  
  (_drop first category untuk menghindari dummy variable trap_).

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])

owner_order = [['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']]
oe = OrdinalEncoder(categories=owner_order)
df['owner'] = oe.fit_transform(df[['owner']])

categorical_cols = ['fuel', 'seller_type', 'transmission']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Konversi seluruh kolom menjadi tipe numerik
df = df.astype(int)

```
### Normalisasi Fitur

Proses normalisasi dilakukan untuk menyetarakan skala dari setiap fitur numerik. Hal ini penting terutama untuk model seperti K-Nearest Neighbors yang sensitif terhadap perbedaan skala.

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('selling_price', axis=1)
y = df['selling_price'].values.reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

```
### Pembagian Data Latih dan Data Uji

Dataset dibagi menjadi data latih dan data uji dengan rasio **80:20** menggunakan fungsi `train_test_split`.  
Pembagian ini bertujuan untuk mengevaluasi performa model secara obyektif terhadap data yang belum pernah dilihat sebelumnya.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=44, shuffle=True
)
```
## Modeling
Pada tahap ini, dilakukan pengembangan dan evaluasi model machine learning guna memprediksi harga jual kendaraan bekas berdasarkan fitur-fitur spesifik. Tiga algoritma regresi dipertimbangkan, yaitu K-Nearest Neighbors (KNN), Random Forest Regressor, dan Extreme Gradient Boosting (XGBoost).

#### 1. **K-Nearest Neighbors (KNN)**
   Model KNN digunakan sebagai baseline karena kesederhanaannya dan tidak memerlukan asumsi distribusi data. Model ini diinisialisasi dengan parameter `n_neighbors=5`. Kelebihan KNN adalah interpretabilitasnya yang tinggi, namun kelemahannya adalah sensitivitas terhadap skala dan dimensi data, serta kurang optimal pada data besar atau kompleks.

#### 2. **Random Forest Regressor**
   Random Forest adalah ensemble method berbasis bagging yang membangun banyak pohon keputusan dan menggabungkan hasilnya. Parameter awal adalah `n_estimators=100` dan `max_depth=10`. Model ini cenderung robust terhadap overfitting dan dapat menangani data dengan fitur non-linear, namun kurang efisien dalam interpretasi dan tuning.

#### 3. **XGBoost Regressor**
   XGBoost adalah model boosting yang mengoptimalkan loss function melalui pendekatan gradient boosting. Model diinisialisasi dengan `n_estimators=100`, `learning_rate=0.1`, dan `max_depth=3`. Kelebihan utamanya adalah akurasi tinggi dan efisiensi komputasi, meskipun tuning-nya relatif kompleks.


## Evaluation

Dalam proyek ini, model dikembangkan untuk memprediksi harga jual kendaraan bekas, sehingga permasalahan termasuk dalam **regresi**. Oleh karena itu, metrik evaluasi yang digunakan harus mencerminkan **akurasi kuantitatif prediksi** secara kontinu, yaitu:

### 1. Root Mean Squared Error (RMSE)

RMSE mengukur seberapa jauh nilai prediksi model berbeda dari nilai aktual dalam satuan yang sama dengan target (dalam kasus ini, INR). RMSE sangat sensitif terhadap outlier karena mengkuadratkan selisih sebelum menghitung akar rata-rata.

   **Formula:**

   $$
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
   $$

Semakin kecil nilai RMSE, semakin akurat model dalam melakukan prediksi.

### 2. R² Score (Koefisien Determinasi)

R² Score mengukur proporsi variabilitas target yang dapat dijelaskan oleh fitur-fitur input dalam model. Nilainya berada dalam rentang [0, 1], di mana nilai mendekati 1 menunjukkan model menjelaskan hampir seluruh variasi target.
Nilai R² mendekati 0 menunjukkan model kurang mampu menjelaskan variasi dalam data.

## Hasil Evaluasi Model

Berdasarkan evaluasi menggunakan metrik RMSE dan R², diperoleh hasil sebagai berikut:

| **Model**              | **RMSE (Train)** | **RMSE (Test)** | **R² Score (Test)** |
|------------------------|------------------|------------------|---------------------|
| **XGBoost (Tuned)**    | 127,467          | **227,316**      | **0.8430**          |
| Random Forest (Tuned)  | 118,731          | 238,972          | 0.8265              |
| Random Forest          | 155,157          | 244,496          | 0.8184              |
| XGBoost                | 246,931          | 255,900          | 0.8011              |
| K-Nearest Neighbors    | 282,798          | 264,143          | 0.7880              |

Model **XGBoost (Tuned)** menunjukkan performa terbaik secara keseluruhan, dengan **RMSE terkecil** di data uji dan **R² Score tertinggi**, yaitu 0.843. Artinya, model ini mampu menjelaskan sekitar 84% dari variasi harga kendaraan bekas secara akurat.

Sebaliknya, model **K-Nearest Neighbors (KNN)** memiliki performa paling rendah, dengan **R² sebesar 0.788** dan **RMSE tertinggi**. Hal ini mengindikasikan bahwa KNN kurang cocok untuk menangani data yang kompleks dan berdimensi tinggi seperti kasus ini.


