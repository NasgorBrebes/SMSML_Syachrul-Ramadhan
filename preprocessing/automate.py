import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data():
    # 1. Atur Path Folder (Supaya jalan di Laptop maupun GitHub)
    # BASE_DIR adalah folder utama project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path Input dan Output
    DATA_PATH = os.path.join(BASE_DIR, 'dataset_raw', 'HR-Employee-Attrition.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset_raw')
    
    print(f"🔄 Memulai proses data dari: {DATA_PATH}")
    
    # Cek apakah file ada
    if not os.path.exists(DATA_PATH):
        print("❌ Error: File dataset tidak ditemukan!")
        return

    # 2. Load Data
    df = pd.read_csv(DATA_PATH)

    # 3. Data Cleaning (Sesuai hasil eksperimen Notebook)
    # Drop kolom variansi 0 dan ID
    cols_to_drop = ['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber']
    # Hanya drop kolom yang benar-benar ada di df
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    df_clean = df.drop(columns=cols_to_drop)

    # 4. Encoding (Ubah Teks -> Angka)
    for col in df_clean.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
    
    # 5. Split Fitur & Target
    target_col = 'Attrition'
    if target_col not in df_clean.columns:
        print("❌ Error: Kolom target 'Attrition' hilang.")
        return

    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]

    # 6. Scaling (Standarisasi Angka)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # 7. Split Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 8. Simpan File Bersih
    print("💾 Menyimpan data hasil processing...")
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)
    
    print("✅ AUTOMATION SUKSES! 4 File (X_train, y_train, dst) telah dibuat.")

if __name__ == "__main__":
    preprocess_data()