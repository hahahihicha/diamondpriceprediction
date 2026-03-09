# ================================
# 1. IMPORT LIBRARY
# ================================
import pandas as pd
import numpy as np

# library untuk machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# model yang digunakan
from xgboost import XGBRegressor

# library untuk menyimpan model
import joblib


# ================================
# 2. LOAD DATASET
# ================================
# membaca dataset diamond
df = pd.read_csv("diamonds.csv")


# ================================
# 3. DATA PREPROCESSING
# ================================

# mengubah data kategori menjadi angka
df['cut'] = df['cut'].map({'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4})
df['color'] = df['color'].map({'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6})
df['clarity'] = df['clarity'].map({'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7})


# ================================
# 4. MEMISAHKAN FITUR DAN TARGET
# ================================
# fitur yang digunakan
X = df[['carat','cut','color','clarity','depth','table','x','y','z']]

# target yang diprediksi
y = df['price']


# ================================
# 5. SPLIT DATA TRAINING DAN TEST
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 6. SCALING DATA
# ================================
# scaler digunakan untuk menyamakan skala data
scaler = StandardScaler()

# fit scaler pada data training
X_train = scaler.fit_transform(X_train)

# transform data testing
X_test = scaler.transform(X_test)


# ================================
# 7. TRAIN MODEL
# ================================
# menggunakan model XGBoost
model = XGBRegressor()

# melatih model
model.fit(X_train, y_train)


# ================================
# 8. EVALUASI MODEL
# ================================
y_pred = model.predict(X_test)

print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))


# ================================
# 9. MENYIMPAN MODEL
# ================================
# model dan scaler disimpan agar bisa dipakai di streamlit

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model berhasil disimpan!")