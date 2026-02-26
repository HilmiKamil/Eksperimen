import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("heart_disease_raw/raw/heart_disease_uci.csv")

# ===============================
# HANDLE KATEGORIKAL
# ===============================
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ===============================
# PISAH FITUR & TARGET
# ===============================
X = df.drop("num", axis=1)
y = df["num"]

# Jadikan klasifikasi biner
y = (y > 0).astype(int)

# ===============================
# IMPUTASI MISSING VALUE (WAJIB)
# ===============================
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# SIMPAN OUTPUT
# ===============================
os.makedirs("preprocessing/heart_disease_preprocessing", exist_ok=True)

np.save("preprocessing/heart_disease_preprocessing/X_train.npy", X_train)
np.save("preprocessing/heart_disease_preprocessing/y_train.npy", y_train)
np.save("preprocessing/heart_disease_preprocessing/X_test.npy", X_test)
np.save("preprocessing/heart_disease_preprocessing/y_test.npy", y_test)

print("Preprocessing selesai & file disimpan")