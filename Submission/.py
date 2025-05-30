# =============================================================================
# Impor Pustaka
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# =============================================================================
# 1. Pemuatan Data (Data Loading)
# =============================================================================
print("Memuat dataset...")
df = pd.read_csv('heart.csv')
print("Dataset berhasil dimuat.")
print("Informasi Dataset Awal:")
df.info()
print("\nContoh Data:")
print(df.head())
print("\nStatistik Deskriptif:")
print(df.describe())
print("\nJumlah Nilai Unik per Kolom:")
print(df.nunique())
print("\nDistribusi Kelas Target (HeartDisease):")
print(df['HeartDisease'].value_counts())

# =============================================================================
# 2. Pra-pemrosesan Data (Data Preprocessing)
# =============================================================================
print("\nMemulai pra-pemrosesan data...")

# Memisahkan fitur (X) dan target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
print("Fitur dan target telah dipisahkan.")

# One-hot encode fitur kategorikal
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=np.number).columns

print(f"Kolom Kategorikal: {list(categorical_cols)}")
print(f"Kolom Numerikal: {list(numerical_cols)}")

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("One-hot encoding selesai.")
final_feature_names = X.columns.tolist()

# Scaling fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling fitur selesai.")

# Encoding label target (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoding label target selesai.")
report_target_names = ['Tidak Sakit Jantung', 'Sakit Jantung']

# Pembagian data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Data dibagi menjadi data latih ({X_train.shape[0]} sampel) dan data uji ({X_test.shape[0]} sampel).")
print("Pra-pemrosesan data selesai.")

# =============================================================================
# 3. Pemodelan (Modeling) - Tanpa Tuning
# =============================================================================
print("\nMemulai pemodelan (tanpa tuning)...")

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results_initial = {}

for name, model in models.items():
    print(f"\nMelatih model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results_initial[name] = acc
    print(f"Akurasi {name}: {acc:.4f}")
    print(f"Laporan Klasifikasi {name}:\n", classification_report(y_test, y_pred, target_names=report_target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_target_names, yticklabels=report_target_names)
    plt.title(f"Confusion Matrix - {name} (Initial)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"confusion_matrix_{name.lower().replace(' ', '_')}_initial.png")
    plt.close()

    # Precision, Recall, F1-score per Kelas
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_encoded))
    x_metrics = np.arange(len(report_target_names))

    plt.figure(figsize=(10, 5))
    plt.bar(x_metrics - 0.25, precision, width=0.25, label='Precision')
    plt.bar(x_metrics, recall, width=0.25, label='Recall')
    plt.bar(x_metrics + 0.25, f1, width=0.25, label='F1-score')
    plt.xlabel("Kelas")
    plt.ylabel("Skor")
    plt.title(f"Precision, Recall, F1-score per Kelas - {name} (Initial)")
    plt.xticks(x_metrics, report_target_names, rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"metrics_per_class_{name.lower().replace(' ', '_')}_initial.png")
    plt.close()


# Feature Importance (hanya untuk model yang mendukung)
models_with_importance_initial = {
    "Random Forest": models["Random Forest"],
    "Decision Tree": models["Decision Tree"]
}
for model_name_imp, model_obj_imp in models_with_importance_initial.items():
    if hasattr(model_obj_imp, 'feature_importances_'):
        importances = model_obj_imp.feature_importances_
        if len(final_feature_names) == len(importances):
            feat_imp_df = pd.DataFrame({
                'Fitur': final_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, max(6, len(final_feature_names) * 0.3))) # Adjust height
            sns.barplot(x='Importance', y='Fitur', data=feat_imp_df, palette='coolwarm')
            plt.title(f"Feature Importance - {model_name_imp} (Initial)")
            plt.xlabel("Importance Score")
            plt.ylabel("Fitur")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"feature_importance_{model_name_imp.lower().replace(' ', '_')}_initial.png")
            plt.close()

print("Pemodelan (tanpa tuning) selesai.")

# =============================================================================
# 4. Pemodelan (Modeling) - Dengan Hyperparameter Tuning (GridSearchCV)
# =============================================================================
print("\nMemulai pemodelan (dengan hyperparameter tuning)...")

param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
}

best_estimators = {}
results_tuned = {}

for name, model_instance in models.items(): # Menggunakan instance model awal
    print(f"\nMelakukan GridSearchCV untuk: {name}")
    grid_search = GridSearchCV(estimator=model_instance,
                               param_grid=param_grids[name],
                               cv=3,
                               n_jobs=-1,
                               verbose=1,
                               scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search.best_estimator_
    print(f"Parameter terbaik untuk {name}: {grid_search.best_params_}")
    print(f"Skor validasi silang terbaik untuk {name}: {grid_search.best_score_:.4f}")

    # Evaluasi model terbaik hasil tuning
    y_pred_tuned = best_estimators[name].predict(X_test)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    results_tuned[name] = acc_tuned
    print(f"Akurasi Test Set {name} (Tuned): {acc_tuned:.4f}")
    print(f"Laporan Klasifikasi {name} (Tuned):\n", classification_report(y_test, y_pred_tuned, target_names=report_target_names))

    # Confusion Matrix (Tuned)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=report_target_names, yticklabels=report_target_names)
    plt.title(f"Confusion Matrix - {name} (Tuned)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"confusion_matrix_{name.lower().replace(' ', '_')}_tuned.png")
    plt.close()

# Feature Importance (hanya untuk model yang mendukung setelah tuning)
models_with_importance_tuned = {
    "Random Forest": best_estimators["Random Forest"],
    "Decision Tree": best_estimators["Decision Tree"]
}
for model_name_imp_tuned, model_obj_imp_tuned in models_with_importance_tuned.items():
    if hasattr(model_obj_imp_tuned, 'feature_importances_'):
        importances_tuned = model_obj_imp_tuned.feature_importances_
        if len(final_feature_names) == len(importances_tuned):
            feat_imp_df_tuned = pd.DataFrame({
                'Fitur': final_feature_names,
                'Importance': importances_tuned
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, max(6, len(final_feature_names) * 0.3))) # Adjust height
            sns.barplot(x='Importance', y='Fitur', data=feat_imp_df_tuned, palette='coolwarm')
            plt.title(f"Feature Importance - {model_name_imp_tuned} (Tuned)")
            plt.xlabel("Importance Score")
            plt.ylabel("Fitur")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"feature_importance_{model_name_imp_tuned.lower().replace(' ', '_')}_tuned.png")
            plt.close()

print("Pemodelan (dengan hyperparameter tuning) selesai.")

# =============================================================================
# 5. Perbandingan Akurasi Model
# =============================================================================
initial_accuracies = [results_initial[name] for name in models.keys()]
tuned_accuracies = [results_tuned[name] for name in models.keys() if name in results_tuned]
model_names_plot = list(models.keys())

x_plot = np.arange(len(model_names_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x_plot - width/2, initial_accuracies, width, label='Akurasi Awal')
rects2 = ax.bar(x_plot + width/2, tuned_accuracies, width, label='Akurasi Tuned')

ax.set_ylabel('Akurasi')
ax.set_title('Perbandingan Akurasi Model: Awal vs Tuned')
ax.set_xticks(x_plot)
ax.set_xticklabels(model_names_plot)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')
fig.tight_layout()
# plt.show()
plt.savefig("accuracy_comparison.png")
plt.close()

print("\nAnalisis selesai. Plot telah disimpan.")

# =============================================================================
# Akhir Skrip
# =============================================================================