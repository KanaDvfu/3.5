import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------
# Helpers: saving tables/figures
# ------------------------------

def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_series_counts(s: pd.Series, path: str, col_name: str = "count") -> pd.DataFrame:
    df = s.value_counts(dropna=False).rename(col_name).reset_index().rename(columns={"index": s.name or "value"})
    save_df(df, path)
    return df


def save_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", filename), dpi=200)
    plt.close()


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", filename), dpi=200)
    plt.close()

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures/adult", exist_ok=True)
os.makedirs("outputs/tables/adult", exist_ok=True)

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race",
    "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

# ————————————————————————————————————————————————————————————————
# Загрузка и объединение данных
# Загружаем данные
train_data = pd.read_csv("data/adult.data", names=columns, sep=", ", engine="python")
test_data = pd.read_csv("data/adult.test", names=columns, sep=", ", engine="python", skiprows=1)

# Убираем точку в конце income в test
test_data["income"] = test_data["income"].str.replace(".", "", regex=False)

# Объединяем
data = pd.concat([train_data, test_data], ignore_index=True)

print("Размер данных:", data.shape)
print(data.head())

# Summary before cleaning
summary_before = pd.DataFrame([
    {"stage": "raw_combined", "rows": int(data.shape[0]), "cols": int(data.shape[1])}
])
save_df(summary_before, "outputs/tables/adult/adult_summary_stages.csv")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Очистка данных
# Заменяем ? на NaN
data.replace("?", np.nan, inplace=True)

# Удаляем строки с пропусками
data.dropna(inplace=True)

print("Размер после очистки:", data.shape)

# Append summary after cleaning
summary_after_clean = pd.DataFrame([
    {"stage": "after_dropna", "rows": int(data.shape[0]), "cols": int(data.shape[1])}
])
summary_stages = pd.concat([summary_before, summary_after_clean], ignore_index=True)
save_df(summary_stages, "outputs/tables/adult/adult_summary_stages.csv")

# Class balance after cleaning
save_series_counts(data["income"], "outputs/tables/adult/adult_income_distribution_after_clean.csv")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Кодирование категориальных признаков
# Разделяем X и y
X = data.drop("income", axis=1)
y = data["income"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("Размер после кодирования:", X.shape)

summary_after_ohe = pd.DataFrame([
    {"stage": "after_one_hot", "rows": int(X.shape[0]), "cols": int(X.shape[1])}
])
summary_stages = pd.concat([summary_stages, summary_after_ohe], ignore_index=True)
save_df(summary_stages, "outputs/tables/adult/adult_summary_stages.csv")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Масштабирование
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Базовая модель (Linear SVM)
linear_model = LinearSVC(max_iter=5000)
linear_model.fit(X_train_scaled, y_train)

y_pred_linear = linear_model.predict(X_test_scaled)

acc_linear = accuracy_score(y_test, y_pred_linear)
print("Linear SVM Accuracy:", acc_linear, "\n")

save_confusion_matrix(y_test, y_pred_linear, "AdultIncome: LinearSVC (baseline)", "adult_linear_baseline_cm.png")

# Classification report (baseline)
report_linear = pd.DataFrame(classification_report(y_test, y_pred_linear, output_dict=True)).T.reset_index().rename(columns={"index": "label"})
save_df(report_linear, "outputs/tables/adult/adult_linear_baseline_classification_report.csv")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Подбор C для LinearSVC
from sklearn.model_selection import GridSearchCV

param_grid_linear = {
    'C': [0.01, 0.1, 1, 10, 100]
}

grid_linear = GridSearchCV(
    LinearSVC(max_iter=5000),
    param_grid_linear,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_linear.fit(X_train_scaled, y_train)

print("Best C (Linear):", grid_linear.best_params_)
print("Best CV score (Linear):", grid_linear.best_score_)

best_linear = grid_linear.best_estimator_
y_pred_best_linear = best_linear.predict(X_test_scaled)

acc_best_linear = accuracy_score(y_test, y_pred_best_linear)
print("Test accuracy with best Linear C:", acc_best_linear, "\n")

save_confusion_matrix(y_test, y_pred_best_linear, "AdultIncome: LinearSVC (best C)", "adult_linear_bestC_cm.png")

report_best_linear = pd.DataFrame(classification_report(y_test, y_pred_best_linear, output_dict=True)).T.reset_index().rename(columns={"index": "label"})
save_df(report_best_linear, "outputs/tables/adult/adult_linear_bestC_classification_report.csv")

# Save GridSearchCV results table
cv_results_linear = pd.DataFrame(grid_linear.cv_results_)
save_df(cv_results_linear, "outputs/tables/adult/adult_linear_gridsearch_cv_results.csv")
# ————————————————————————————————————————————————————————————————

# ————————————————————————————————————————————————————————————————
# Берем подвыборку для RBF (ускорение)
sample_data = data.sample(n=10000, random_state=42)

X_sample = sample_data.drop("income", axis=1)
y_sample = sample_data["income"]

X_sample = pd.get_dummies(X_sample, drop_first=True)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
)

scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train_s)
X_test_s = scaler_s.transform(X_test_s)

rbf_model = SVC(kernel='rbf', C=1, gamma='scale')
rbf_model.fit(X_train_s, y_train_s)

y_pred_rbf = rbf_model.predict(X_test_s)

acc_rbf = accuracy_score(y_test_s, y_pred_rbf)
print("RBF SVM Accuracy (sample 10k):", acc_rbf)

save_confusion_matrix(y_test_s, y_pred_rbf, "AdultIncome: SVC RBF (sample 10k)", "adult_rbf_sample10k_cm.png")

report_rbf = pd.DataFrame(classification_report(y_test_s, y_pred_rbf, output_dict=True)).T.reset_index().rename(columns={"index": "label"})
save_df(report_rbf, "outputs/tables/adult/adult_rbf_sample10k_classification_report.csv")

# Полный запуск RBF на всем наборе данных
# Может выполняться долго, поэтому оборачиваем в try/except
acc_rbf_full = None
rbf_full_status = "not_run"

try:
    rbf_full_model = SVC(kernel='rbf', C=1, gamma='scale')
    rbf_full_model.fit(X_train_scaled, y_train)

    y_pred_rbf_full = rbf_full_model.predict(X_test_scaled)
    acc_rbf_full = accuracy_score(y_test, y_pred_rbf_full)
    rbf_full_status = "ok"

    print("RBF SVM Accuracy (full dataset):", acc_rbf_full)

    save_confusion_matrix(
        y_test,
        y_pred_rbf_full,
        "AdultIncome: SVC RBF (full dataset)",
        "adult_rbf_full_cm.png"
    )

    report_rbf_full = (
        pd.DataFrame(classification_report(y_test, y_pred_rbf_full, output_dict=True))
        .T.reset_index()
        .rename(columns={"index": "label"})
    )
    save_df(report_rbf_full, "outputs/tables/adult/adult_rbf_full_classification_report.csv")

except Exception as e:
    rbf_full_status = f"failed: {type(e).__name__}: {e}"
    print("RBF full dataset run failed:", e)

# Comparison table for the report
comparison_rows = [
    {"model": "LinearSVC_baseline", "dataset": "full", "accuracy": float(acc_linear)},
    {"model": f"LinearSVC_bestC_{grid_linear.best_params_.get('C')}", "dataset": "full", "accuracy": float(acc_best_linear)},
    {"model": "SVC_RBF", "dataset": "sample_10k", "accuracy": float(acc_rbf)},
]

if acc_rbf_full is not None:
    comparison_rows.append(
        {"model": "SVC_RBF", "dataset": "full", "accuracy": float(acc_rbf_full)}
    )

comparison = pd.DataFrame(comparison_rows)

save_df(comparison, "outputs/tables/adult/adult_model_comparison.csv")

rbf_full_status_df = pd.DataFrame([
    {"status": rbf_full_status, "accuracy": acc_rbf_full}
])
save_df(rbf_full_status_df, "outputs/tables/adult/adult_rbf_full_status.csv")

save_barplot(comparison, "model", "accuracy", "AdultIncome: Accuracy comparison", "adult_accuracy_comparison.png")
# ————————————————————————————————————————————————————————————————