import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns

plt.rcParams.update({'font.size': 14})

# ========================
# Klasy: ParticleType 0 = Electron, 1 = Muon, 2 = Pion
# ========================
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Electron", "Muon", "Pion"]
CLASS_COLORS = ["blue", "red", "green"]
CLASS_HATCH = [None, "//", "\\\\"]
TARGET = "ParticleType"

# ========================
# Importing data
# ========================
def load(path):
    with uproot.open(path) as f:
        return f["MLDataTree"].arrays(library="pd")

df_Electron = load("../Classifiers/MLData3TypeMCElectronFull.root")
df_Muon = load("../Classifiers/MLData3TypeMCMuonFull.root")
df_Electron_Background = load("../Classifiers/MLData3TypeMCElectronBackground.root")
df_Muon_Background = load("../Classifiers/MLData3TypeMCMuonBackground.root")
df_Pion = load("../Classifiers/MLData3TypeMCPionFull.root")  
df = pd.concat(
    [df_Electron, df_Muon, df_Electron_Background, df_Muon_Background, df_Pion],
    ignore_index=True
)

print("Liczba czastek w klasach:")
print(df[TARGET].value_counts().sort_index())

eta_values = {
    0: 'FullRange',
    1: '<1.0',
    2: '1.0-1.5',
    3: '1.5-2.5'
}

features_list = ['Cal_FVariable', 'Cal_EMprop', 'Cal_Lambda', 'Cal_Lambda2', 'Cal_Radius']

# ========================
# Splitting dataset to train and test
# ========================
df_train, df_test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET]
)

X_train = df_train[features_list]
y_train = df_train[TARGET].astype(int)

# ========================
# Training
# ========================
print("Training model on data...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=2
)
rf.fit(X_train, y_train)

# ========================
# Exporting model
# ========================
initial_type = [('float_input', FloatTensorType([None, len(features_list)]))]
onnx_model = convert_sklearn(
    rf,
    initial_types=initial_type,
    target_opset=17,
    options={type(rf): {"zipmap": False}}
)
onnx_model.ir_version = 9
# Wyjscia ONNX: [0] label (N,), [1] probabilities (N,3) w kolejnosci Electron, Muon, Pion
with open("../ONNX/RFMuonElectronPion.onnx", "wb") as fo:
    fo.write(onnx_model.SerializeToString())

# ========================
# Generating outputs
# ========================
reports = {}
with PdfPages("Plots/Classifier/MCRF_3Class.pdf") as pdf:

    for eta in range(4):
        eta_name = eta_values[eta]
        print(f"\n=== Testing for EtaRange = {eta_name} ===")

        if eta == 0:
            df_eta = df_test.copy()
        else:
            df_eta = df_test[df_test['EtaRange'] == eta].copy()

        if len(df_eta) == 0:
            continue

        print(f"Number of Leptons/Hadrons: {len(df_eta)}")

        X_test = df_eta[features_list]
        y_test = df_eta[TARGET].astype(int).values

        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)   # shape (N, 3)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Test:    {acc:.4f}")

        reports[f"EtaRange = {eta_name}"] = classification_report(
            y_test, y_pred, labels=CLASS_LABELS,
            target_names=CLASS_NAMES, zero_division=0
        )

        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        fig.suptitle(
            f'EtaRange = {eta_name} (TEST DATA) - Accuracy: {acc:.3f}',
            fontsize=16, fontweight='bold'
        )

        # 1. Confusion Matrix (3x3)
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2%', ax=axes[0, 0], cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_title('Confusion Matrix')

        # 2. ROC One-vs-Rest
        y_bin = label_binarize(y_test, classes=CLASS_LABELS)
        for k, name in enumerate(CLASS_NAMES):
            if y_bin[:, k].sum() == 0 or y_bin[:, k].sum() == len(y_bin):
                continue  # brak pozytywow lub negatywow w tym przedziale eta
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_proba[:, k])
            axes[0, 1].plot(fpr, tpr, lw=2, color=CLASS_COLORS[k],
                            label=f'{name} vs rest (AUC = {auc(fpr, tpr):.4f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves (One-vs-Rest)')
        axes[0, 1].legend(loc='lower right')

        # 3. Feature Importance
        importances = rf.feature_importances_
        axes[0, 2].bar(features_list, importances)
        axes[0, 2].set_title('Feature Importance')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4-6. Classifier response: P(klasa k) dla kazdej prawdziwej klasy
        for k, name in enumerate(CLASS_NAMES):
            ax = axes[1, k]
            for j, true_name in enumerate(CLASS_NAMES):
                mask = y_test == CLASS_LABELS[j]
                if mask.sum() == 0:
                    continue
                ax.hist(y_proba[mask, k], bins=25, range=(0, 1), density=True,
                        alpha=0.5, label=f'True {true_name}',
                        color=CLASS_COLORS[j], hatch=CLASS_HATCH[j])
            ax.set_xlabel(f'P({name})')
            ax.set_title(f'Classifier Response: P({name})')
            ax.legend(loc='upper center')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # Correlation matrix
    fig2 = plt.figure(figsize=(10, 8))
    sns.heatmap(df[features_list + [TARGET]].corr(), annot=True, cmap='coolwarm')
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)

    # Classification reports summary
    fig = plt.figure(figsize=(11, 14))
    fig.text(
        0.35, 0.98,
        "Classification Reports Summary (3 classes)\nTrained:  MCData.root | Test: MCData.root",
        ha='center', fontsize=18, fontweight='bold'
    )
    y_pos = 0.90
    for title, report in reports.items():
        fig.text(0.05, y_pos, f"\n{title}", fontsize=12, fontweight='bold', family='monospace')
        y_pos -= 0.03
        for line in report.split('\n'):
            if line.strip():
                fig.text(0.05, y_pos, line, fontsize=9, family='monospace')
                y_pos -= 0.015
        y_pos -= 0.03

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("\n" + "=" * 70)
print(f"End. Testing data: {len(df_test)} particles.")
print("PDF: Plots/Classifier/MCRF_3Class.pdf")
print("=" * 70)

# ========================================================
# EXPORT SEPARATE PNG FILES (FullRange)
# ========================================================
print("\nGenerating separate PNG files for FullRange...")

X_full_test = df_test[features_list]
y_full_test = df_test[TARGET].astype(int).values
y_full_proba = rf.predict_proba(X_full_test)
y_full_bin = label_binarize(y_full_test, classes=CLASS_LABELS)

# 1. ROC curves (One-vs-Rest)
plt.figure(figsize=(8, 6))
for k, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_full_bin[:, k], y_full_proba[:, k])
    plt.plot(fpr, tpr, lw=3, color=CLASS_COLORS[k],
             label=f'{name} vs rest (AUC = {auc(fpr, tpr):.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("Plots/Classifier/ROC_FullRange_3Class.png", dpi=300)
plt.close()

# 2. Response plots - jeden plik na klase
for k, name in enumerate(CLASS_NAMES):
    plt.figure(figsize=(8, 6))
    for j, true_name in enumerate(CLASS_NAMES):
        mask = y_full_test == CLASS_LABELS[j]
        plt.hist(y_full_proba[mask, k], bins=50, range=(0, 1), density=True,
                 alpha=0.5, label=f'True {true_name}',
                 color=CLASS_COLORS[j], hatch=CLASS_HATCH[j])
    plt.xlabel(f'Classifier Score (Probability of being {name})')
    plt.ylabel('Normalized counts')
    plt.title(f'Classifier Response: P({name})')
    plt.legend(loc='upper center')
    plt.grid(alpha=0.3)
    plt.savefig(f"Plots/Classifier/Response_{name}_FullRange.png", dpi=300)
    plt.close()

# 3. Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_full_test, rf.predict(X_full_test), labels=CLASS_LABELS, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (FullRange)')
plt.tight_layout()
plt.savefig("Plots/Classifier/ConfusionMatrix_FullRange_3Class.png", dpi=300)
plt.close()

# 4. Feature importance
plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances - Random Forest (3 classes)')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [features_list[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig("Plots/Classifier/Features_Importance_3Class.png", dpi=300)
plt.close()

print("PNG files have been saved in the Plots/Classifier/ folder.")