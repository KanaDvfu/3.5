import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os

# Создаем папки для сохранения результатов
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# ————————————————————————————————————————————————————————————————
# Загружаем данные
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
data = pd.read_csv("data/banknote.csv", names=columns)

print("Размер данных:", data.shape)
print(data.head())
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Тест
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# SVM без масштабирования
model_no_scaling = SVC(kernel='rbf', C=1, gamma='scale')
model_no_scaling.fit(X_train, y_train)

y_pred_no_scaling = model_no_scaling.predict(X_test)

acc_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print("Accuracy without scaling:", acc_no_scaling)
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# SVM со стандартизацией
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = SVC(kernel='rbf', C=1, gamma='scale')
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

acc_scaled = accuracy_score(y_test, y_pred_scaled)
print("Accuracy with scaling:", acc_scaled)
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Матрицы ошибок
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"outputs/figures/{filename}")
    plt.close()

save_confusion_matrix(y_test, y_pred_no_scaling,
                      "Confusion Matrix (No Scaling)",
                      "banknote_no_scaling_cm.png")

save_confusion_matrix(y_test, y_pred_scaled,
                      "Confusion Matrix (With Scaling)",
                      "banknote_scaled_cm.png")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Сохраняем таблицу результатов
results = pd.DataFrame({
    "Model": ["SVM without scaling", "SVM with scaling"],
    "Accuracy": [acc_no_scaling, acc_scaled]
})

results.to_csv("outputs/tables/banknote_basic_results.csv", index=False)
print(results, "\n")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Сравнение разных ядер
kernels = ['linear', 'rbf', 'poly']

results_kernel = []

for kernel in kernels:
    model = SVC(kernel=kernel, C=1, gamma='scale')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    results_kernel.append((kernel, acc))
    print(f"Kernel: {kernel}, Accuracy: {acc}")

results_kernel_df = pd.DataFrame(results_kernel, columns=["Kernel", "Accuracy"])
results_kernel_df.to_csv("outputs/tables/banknote_kernel_comparison.csv", index=False)

print(results_kernel_df, "\n")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Поиск оптимальных параметров
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

acc_best = accuracy_score(y_test, y_pred_best)
print("Test accuracy with best parameters:", acc_best)

cv_results = pd.DataFrame(grid.cv_results_)
cv_results.to_csv("outputs/tables/banknote_gridsearch_results.csv", index=False)
# ————————————————————————————————————————————————————————————————