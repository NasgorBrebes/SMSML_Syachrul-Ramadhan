import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- KONFIGURASI ---
# Link MLflow DagsHub milikmu
MLFLOW_TRACKING_URI = "https://dagshub.com/NasgorBrebes/Proyek-Akhir-MLOps.mlflow"

def train_and_tune():
    # 1. Load Data (Mengambil data bersih dari folder dataset_raw)
    # Kita naik satu level (..) lalu masuk ke dataset_raw
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset_raw')
    
    print("🔄 Loading data dari folder dataset_raw...")
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()
    except FileNotFoundError:
        print("❌ Error: File X_train/y_train tidak ditemukan. Jalankan automate Kriteria 1 dulu!")
        return

    # 2. Setup MLflow ke DagsHub
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Experiment_Interview_Model")

    # 3. Hyperparameter Tuning (Manual Loop)
    # Kita coba kombinasi jumlah pohon (n_estimators) dan kedalaman (max_depth)
    n_estimators_list = [50, 100]
    max_depth_list = [10, 20]
    
    run_count = 0
    print(f"🚀 Mulai Training ke DagsHub: {MLFLOW_TRACKING_URI}")
    
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            run_count += 1
            run_name = f"Run_{run_count}_Est{n_est}_Depth{depth}"
            
            print(f"   ▶ Training {run_name}...")
            
            with mlflow.start_run(run_name=run_name):
                # A. Train Model
                model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
                model.fit(X_train, y_train)
                
                # B. Predict
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                # C. LOGGING (Syarat Advance: Manual Logging)
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", acc)
                print(f"      --> Accuracy: {acc:.4f}")
                
                # Log Model (Simpan file model .pkl ke cloud)
                mlflow.sklearn.log_model(model, "model")
                
                # D. LOG ARTEFAK TAMBAHAN (Syarat Advance: Min 2 Artefak)
                
                # Artefak 1: Gambar Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix - Acc: {acc:.2f}")
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                cm_path = "confusion_matrix.png"
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path) # Upload gambar
                plt.close()
                
                # Artefak 2: File Text Classification Report
                report = classification_report(y_test, y_pred)
                report_path = "classification_report.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path) # Upload text
                
                # Hapus file sementara di laptop agar folder bersih
                if os.path.exists(cm_path): os.remove(cm_path)
                if os.path.exists(report_path): os.remove(report_path)

    print("\n✅ Tuning Selesai! Cek website DagsHub kamu sekarang.")

if __name__ == "__main__":
    train_and_tune()