#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DS785 Capstone – Predicting Risk of Heart Disease for Early Detection: Machine Learning Approach (Code)

#Author: Larry Vue  

#This notebook contains the code used for my DS785 capstone project on
#predicting risk of heart disease using the UCI Cleveland dataset
#(with supporting EDA on a larger Kaggle cardio dataset).

#The notebook is organized by project stage:

#1. Setup and helper functions  
#2. Data collection and basic cleaning (Cleveland + Kaggle)  
#3. Cleveland EDA and feature engineering  
#4. Kaggle EDA and feature engineering  
#5. Model development and evaluation – final calibrated logistic model (Cleveland)  
#6. Reduced external validation using Kaggle (reduced features)  
#7. Other machine learning analysis  
#   - 7.1 Multi-model ROC comparison (Logistic / KNN / RF / XGBoost)  
#   - 7.2 PCA + K-Means + KNN visualizations    
#   - 7.3 False Negative vs false pasetive bar chart
#   - 7.4 XGBoost comparison

#Run cells from top to bottom.


# In[3]:


# 1. Setup and helper functions
#    - Imports
#    - Paths and random seed
#    - A few small helper utilities


# In[2]:


# 1. Setup: imports, paths, helper functions

#  Basic Python / scientific stack 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

#  scikit-learn core tools 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#  Metrics and curve utilities 
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve  # <-- calibration_curve comes from here

# -Models / algorithms 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Global config: random seed, file paths, and plotting style

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Edit these paths for your machine if needed
PATH_CLEVELAND = r"C:\Users\larry\Desktop\DS785 Capstone\processed.cleveland.data"
PATH_KAGGLE   = r"C:\Users\larry\Desktop\DS785 Capstone\cardio_train.csv"

FIG_DIR_CLEV   = r"C:\Users\larry\Desktop\DS785 Capstone\figs"
FIG_DIR_KAGGLE = r"C:\Users\larry\Desktop\DS785 Capstone\figs_kaggle"
FIG_DIR_MODEL  = r"C:\Users\larry\Desktop\DS785 Capstone\figs_p4_simple"

for d in [FIG_DIR_CLEV, FIG_DIR_KAGGLE, FIG_DIR_MODEL]:
    os.makedirs(d, exist_ok=True)

# Simple default matplotlib style (keep it basic)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True

# 
# Small helper functions used later
# 

def show_shape_and_head(df, name, n=5):
    """Print shape and show the first few rows of a DataFrame."""
    print(f"{name} shape:", df.shape)
    display(df.head(n))


def pick_threshold_by_recall(y_true, y_prob, target_recall=0.95):
    """
    Simple helper: scan thresholds from 0.99 down to 0.01
    and return the first threshold where Recall >= target_recall.
    """
    for thr in np.linspace(0.99, 0.01, 99):
        y_pred = (y_prob >= thr).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= target_recall:
            return thr
    # fallback if we never hit the target
    return 0.50


def print_basic_metrics(y_true, y_prob, threshold=0.5, label="Model"):
    """
    Print a small set of metrics at a chosen threshold.
    This is handy for quick checks while modeling.
    """
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    acc = (y_pred == y_true).mean()

    print(f"\n[{label}] @ threshold = {threshold:.2f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  PR-AUC:    {ap:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print(f"  Accuracy:  {acc:.3f}")


# In[3]:


## 2. Data collection and basic cleaning

#This section:

#     - Loads the Cleveland and Kaggle datasets.
#     - Performs light cleaning (types, missing values, basic filters)


# In[4]:


#  2.1 Load and clean the UCI Cleveland dataset 

# Column names from the UCI Cleveland heart disease documentation
CLEVELAND_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

def load_clean_cleveland(path):
    """
    Load the UCI Cleveland dataset and apply very basic cleaning.

    Steps:
      1. Read the CSV with the right column names.
      2. Turn "?" into missing values (NaN).
      3. Convert numeric-looking columns to numbers.
      4. Fill the small number of missing values in 'ca' and 'thal' with the mode.
      5. Create a binary target:
         - target = 0  → num == 0  (no disease)
         - target = 1  → num  > 0  (disease present)
      6. Drop the original 'num' column.
    """
    df = pd.read_csv(path, header=None, names=CLEVELAND_COLS)

    # "?" means missing in this file
    df = df.replace("?", np.nan)

    # Convert columns that should be numeric
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal", "num"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # For the categorical columns, keep them as integers
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Fill the small number of missing ca/thal with the mode
    for c in ["ca", "thal"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

    # Make a binary target from 'num' (0..4)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df


# load Cleveland
df_clev = load_clean_cleveland(PATH_CLEVELAND)

print("Cleveland shape:", df_clev.shape)
print("Cleveland target balance:")
print(df_clev["target"].value_counts(normalize=True).round(3))
print("\nFirst 5 rows of Cleveland:")
display(df_clev.head())


# For modeling use these features:
num_cols_clev = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols_clev = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

X_clev = df_clev[num_cols_clev + cat_cols_clev].copy()
y_clev = df_clev["target"].copy()

print("\nX_clev shape:", X_clev.shape)
print("y_clev shape:", y_clev.shape)


#  2.2 Load and basic-clean the Kaggle cardio dataset 


def load_basic_kaggle(path):
    """
    Load the Kaggle cardio dataset and do basic cleaning.

    Steps:
      1. Read CSV (note: it uses ';' as the separator).
      2. Convert age from days to whole years.
      3. Drop rows with clearly impossible height/weight/BP values.
      4. Create a simple BMI feature.
      5. Drop the 'id' column (not useful for modeling).
    """
    # Kaggle uses semicolons as separators
    df = pd.read_csv(path, sep=";")

    print("Original Kaggle shape:", df.shape)

    # age is in days → convert to years (rounded)
    df["age"] = (df["age"] / 365).round().astype(int)

    # Remove rows with unrealistic values
    df = df[
        (df["height"].between(100, 250)) &   # cm
        (df["weight"].between(30, 200)) &    # kg
        (df["ap_hi"].between(80, 250)) &     # systolic BP
        (df["ap_lo"].between(50, 200))       # diastolic BP
    ].copy()

    print("Kaggle shape after basic cleaning:", df.shape)

    # Simple BMI feature
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2

    # Drop 'id'
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


df_kaggle = load_basic_kaggle(PATH_KAGGLE)

print("\nKaggle target balance (cardio):")
print(df_kaggle["cardio"].value_counts(normalize=True).round(3))
print("\nFirst 5 rows of Kaggle (after cleaning):")
display(df_kaggle.head())

# Features/target from Kaggle
X_kaggle = df_kaggle.drop(columns=["cardio"]).copy()
y_kaggle = df_kaggle["cardio"].copy()

print("\nX_kaggle shape:", X_kaggle.shape)
print("y_kaggle shape:", y_kaggle.shape)

print("\nData loading and basic cleaning complete – ready for EDA sections.")


# In[5]:


# 3. Cleveland EDA and feature engineering
#    - Basic profiling
#    - Simple visualizations
#    - Group comparisons (target 0 vs 1)
#    - Feature engineering for modeling


# In[6]:


# 3.0 Make sure Cleveland data is loaded

# Quick check fo df_clev from Section 2

print("Cleveland dataset (df_clev) ready for EDA.")
print("Shape:", df_clev.shape)
print("Target balance:")
print(df_clev["target"].value_counts(normalize=True).round(3))

 
# 3.1 Descriptive statistics
 
desc_clev = df_clev.describe(include="all").T
print("\nDescriptive stats (first few rows):")
display(desc_clev.head())

desc_path = os.path.join(FIG_DIR_CLEV, "cle_descriptive_stats.csv")
desc_clev.to_csv(desc_path)
print(f"Saved descriptive stats to: {desc_path}")


# 3.2 Simple distributions: histograms and boxplots

numeric_clev = ["age", "trestbps", "chol", "thalach", "oldpeak"]

for col in numeric_clev:
    # Histogram
    plt.figure()
    df_clev[col].hist(bins=30)
    plt.title(f"Cleveland – Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    out_hist = os.path.join(FIG_DIR_CLEV, f"cle_{col}_hist.png")
    plt.savefig(out_hist)
    plt.close()
    
    # Boxplot
    plt.figure()
    df_clev[[col]].boxplot()
    plt.title(f"Cleveland – Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    out_box = os.path.join(FIG_DIR_CLEV, f"cle_{col}_box.png")
    plt.savefig(out_box)
    plt.close()

print("Saved basic histograms and boxplots to:", FIG_DIR_CLEV)


# 3.3 Correlation heatmap (numeric features only)

num_cols_all = df_clev.select_dtypes(include=[np.number])                       .drop(columns=["target"]).columns.tolist()

corr_clev = df_clev[num_cols_all].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_clev, interpolation="nearest")
plt.xticks(range(len(num_cols_all)), num_cols_all, rotation=90)
plt.yticks(range(len(num_cols_all)), num_cols_all)
plt.title("Cleveland – Correlation Heatmap (numeric)")
plt.colorbar()
plt.tight_layout()
corr_path = os.path.join(FIG_DIR_CLEV, "cle_corr_heatmap.png")
plt.savefig(corr_path)
plt.close()

print("Saved correlation heatmap to:", corr_path)

# (Optional) save the raw correlation matrix
corr_csv_path = os.path.join(FIG_DIR_CLEV, "cle_correlation_matrix.csv")
corr_clev.to_csv(corr_csv_path)
print("Saved correlation matrix to:", corr_csv_path)


# 3.4 Group comparisons: target = 0 vs 1 for key numerics
#       - If both groups look roughly normal: Welch t-test
#       - Otherwise: Mann–Whitney U test


def compare_groups_numeric(df, feature, target_col="target"):
    """
    Compare the distribution of one numeric feature between
    target=0 and target=1.
    Returns a small dictionary with test name, p-value, and means.
    """
    g0 = df.loc[df[target_col] == 0, feature].dropna()
    g1 = df.loc[df[target_col] == 1, feature].dropna()
    
    # Simple normality check using Shapiro (only if at least 3 values)
    p0 = stats.shapiro(g0).pvalue if len(g0) >= 3 else 0.0
    p1 = stats.shapiro(g1).pvalue if len(g1) >= 3 else 0.0
    
    if p0 > 0.05 and p1 > 0.05:
        # Use Welch t-test (does not assume equal variances)
        stat, p = stats.ttest_ind(g0, g1, equal_var=False)
        test_name = "t-test"
    else:
        # Use Mann–Whitney U test (non-parametric)
        stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        test_name = "Mann-Whitney U"
    
    return {
        "feature": feature,
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_target0": float(g0.mean()),
        "mean_target1": float(g1.mean())
    }

rows = [compare_groups_numeric(df_clev, col, target_col="target")
        for col in numeric_clev]

group_compare_df = pd.DataFrame(rows).sort_values("p_value")
print("\nGroup comparisons (target 0 vs 1) for key numerics:")
display(group_compare_df)

group_csv = os.path.join(FIG_DIR_CLEV, "cle_group_comparisons_numeric.csv")
group_compare_df.to_csv(group_csv, index=False)
print("Saved group comparison table to:", group_csv)

# 
# 3.5 Light outlier handling + feature engineering
#     - Winsorize a few skewed variables using IQR
#     - Create interpretable engineered features:
#         * chol_per_age
#         * thalach_per_age
#         * oldpeak_x_exang
#         * age_band (categorical bands)


def winsorize_iqr(series, whisker=1.5):
    """
    Simple IQR-based winsorization.
    Values below Q1 - 1.5*IQR are set to that lower bound.
    Values above Q3 + 1.5*IQR are set to that upper bound.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    return series.clip(lower=low, upper=high)

# Make a copy so we keep original df_clev unchanged
df_clev_fe = df_clev.copy()

# Winsorize the same numeric columns we plotted
for col in numeric_clev:
    df_clev_fe[col] = winsorize_iqr(df_clev_fe[col])

# Feature 1: cholesterol per ag
df_clev_fe["chol_per_age"] = df_clev_fe["chol"] / df_clev_fe["age"].replace(0, np.nan)
df_clev_fe["chol_per_age"] = df_clev_fe["chol_per_age"].fillna(
    df_clev_fe["chol_per_age"].median()
)

# Feature 2: thalach (peak heart rate) per age
df_clev_fe["thalach_per_age"] = df_clev_fe["thalach"] / df_clev_fe["age"].replace(0, np.nan)
df_clev_fe["thalach_per_age"] = df_clev_fe["thalach_per_age"].fillna(
    df_clev_fe["thalach_per_age"].median()
)

# Feature 3: interaction oldpeak × exang
df_clev_fe["oldpeak_x_exang"] = df_clev_fe["oldpeak"] * df_clev_fe["exang"]

# Feature 4: age bands (categorical risk groups) 
age_bins = [0, 39, 49, 59, 69, 150]
age_labels = ["<=39", "40-49", "50-59", "60-69", "70+"]

df_clev_fe["age_band"] = pd.cut(
    df_clev_fe["age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

# One-hot encode age_band 
df_clev_fe = pd.get_dummies(df_clev_fe, columns=["age_band"], drop_first=False)

print("\nEngineered columns added:")
new_cols = [c for c in df_clev_fe.columns if c not in df_clev.columns]
print(new_cols)


# 3.6 Save engineered Cleveland dataset for modeling
#
clev_fe_path = os.path.join(FIG_DIR_CLEV, "cleveland_featured_for_modeling.csv")
df_clev_fe.to_csv(clev_fe_path, index=False)

print(f"\nSaved engineered Cleveland dataset for modeling to:\n  {clev_fe_path}")
print("Final shape with engineered features:", df_clev_fe.shape)


# In[7]:


# Light winsorization + engineered features
# (Here we assume df_clev_fe already exists with your engineered features.)

# If you don't already have a plain "age_band" column, create it from age:
bins   = [0, 39, 49, 59, 69, 150]
labels = ["<=39", "40-49", "50-59", "60-69", "70+"]

df_clev_fe["age_band"] = pd.cut(
    df_clev_fe["age"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# --- Disease rate by age_band ---
rate = df_clev_fe.groupby("age_band")["target"].mean().reindex(labels)

plt.figure(figsize=(6, 4))
plt.bar(rate.index.astype(str), rate.values)

for i, v in enumerate(rate.values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.ylim(0, 1)
plt.ylabel("Disease rate")
plt.title("Target rate by age band")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR_CLEV, "cle_ageband_rate.png"), dpi=150)
plt.close()

# --- thalach_per_age by target ---
plt.figure(figsize=(6, 4))
data0 = df_clev_fe.loc[df_clev_fe["target"] == 0, "thalach_per_age"]
data1 = df_clev_fe.loc[df_clev_fe["target"] == 1, "thalach_per_age"]

plt.boxplot([data0, data1], labels=["target=0", "target=1"])
plt.ylabel("thalach_per_age")
plt.title("thalach_per_age by target")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR_CLEV, "cle_thalach_per_age_box.png"), dpi=150)
plt.close()

print("Cleveland EDA + feature engineering complete.")


# In[8]:


# 4. Kaggle EDA and feature engineering
#    Dataset: cardio_train.csv (Kaggle "Cardio" dataset)


# In[9]:



# 4.1 Load Kaggle dataset and basic profiling

df_kaggle = pd.read_csv(PATH_KAGGLE, sep=";")  # Kaggle uses ";" as separator

print("Original Kaggle shape:", df_kaggle.shape)
print(df_kaggle.head())

# Target column in this dataset is called "cardio" (0 = no disease, 1 = disease)
print("\nTarget distribution (cardio):")
print(df_kaggle["cardio"].value_counts(normalize=True).round(3))

print("\nMissing values per column:")
print(df_kaggle.isna().sum())

# Descriptive stats for the raw Kaggle data
desc_kaggle_raw = df_kaggle.describe().T
print("\nDescriptive stats (first few rows):")
print(desc_kaggle_raw.head())

desc_raw_path = os.path.join(FIG_DIR_KAGGLE, "kg_descriptive_stats_raw.csv")
desc_kaggle_raw.to_csv(desc_raw_path)
print(f"\nSaved raw descriptive stats to: {desc_raw_path}")


# 4.2 Simple cleaning steps

# age is stored in days → convert to years
df_kaggle["age"] = (df_kaggle["age"] / 365).round().astype(int)

# Remove obviously impossible values to keep the data realistic
df_kaggle = df_kaggle[
    (df_kaggle["height"].between(100, 250)) &   # height in cm
    (df_kaggle["weight"].between(30, 200)) &    # weight in kg
    (df_kaggle["ap_hi"].between(80, 250)) &     # systolic BP
    (df_kaggle["ap_lo"].between(50, 200))       # diastolic BP
].copy()

# Drop ID column
if "id" in df_kaggle.columns:
    df_kaggle = df_kaggle.drop(columns=["id"])

print("\nShape after basic cleaning:", df_kaggle.shape)


# 4.3 Simple histograms and boxplots for key numeric features
numeric_kaggle = ["age", "height", "weight", "ap_hi", "ap_lo"]

for col in numeric_kaggle:
    # Histogram
    plt.figure()
    df_kaggle[col].hist(bins=40)
    plt.title(f"Kaggle Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR_KAGGLE, f"kg_{col}_hist.png"), dpi=150)
    plt.close()

    # Boxplot
    plt.figure()
    df_kaggle[[col]].boxplot()
    plt.title(f"Kaggle Boxplot: {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR_KAGGLE, f"kg_{col}_box.png"), dpi=150)
    plt.close()

print("\nSaved basic histograms and boxplots to:", FIG_DIR_KAGGLE)


# 4.4 Simple group comparisons (cardio = 0 vs 1) for a few numerics
def compare_groups_kaggle(feature):
    """
    Compare cardio=0 vs cardio=1 for a single numeric feature.
    Uses a simple normality check on a sample, then either
    Welch t-test or Mann-Whitney U test.
    """
    g0 = df_kaggle.loc[df_kaggle["cardio"] == 0, feature].dropna()
    g1 = df_kaggle.loc[df_kaggle["cardio"] == 1, feature].dropna()

    # for speed, down-sample if there are many rows
    g0s = g0.sample(min(5000, len(g0)), random_state=0)
    g1s = g1.sample(min(5000, len(g1)), random_state=0)

    p0 = stats.shapiro(g0s).pvalue if len(g0s) >= 3 else 0.0
    p1 = stats.shapiro(g1s).pvalue if len(g1s) >= 3 else 0.0

    if p0 > 0.05 and p1 > 0.05:
        test_name = "t-test"
        stat, p_val = stats.ttest_ind(g0s, g1s, equal_var=False)
    else:
        test_name = "Mann-Whitney U"
        stat, p_val = stats.mannwhitneyu(g0s, g1s, alternative="two-sided")

    return {
        "feature": feature,
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "mean_0": float(g0.mean()),
        "mean_1": float(g1.mean())
    }

features_to_check = ["age", "ap_hi", "ap_lo", "weight"]
rows_kaggle = [compare_groups_kaggle(f) for f in features_to_check]
group_kaggle_df = pd.DataFrame(rows_kaggle).sort_values("p_value")

print("\nGroup comparisons (cardio 0 vs 1) for key numerics:")
print(group_kaggle_df)

group_cmp_path = os.path.join(FIG_DIR_KAGGLE, "kg_group_comparisons_numeric.csv")
group_kaggle_df.to_csv(group_cmp_path, index=False)
print(f"\nSaved group comparison table to: {group_cmp_path}")


# 4.5 Interpretable feature engineering

# BMI (Body Mass Index) from height and weight
height_m = df_kaggle["height"] / 100.0
df_kaggle["bmi"] = df_kaggle["weight"] / (height_m ** 2)

# Pulse pressure = systolic - diastolic
df_kaggle["pulse_pressure"] = df_kaggle["ap_hi"] - df_kaggle["ap_lo"]

# Approximate mean arterial pressure
df_kaggle["map_est"] = df_kaggle["ap_lo"] + df_kaggle["pulse_pressure"] / 3.0

# Age bands (coarse risk groups)
age_bins   = [0, 29, 39, 49, 59, 69, 120]
age_labels = ["<=29", "30-39", "40-49", "50-59", "60-69", "70+"]

df_kaggle["age_band"] = pd.cut(
    df_kaggle["age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

print("\nEngineered columns added:")
print([c for c in ["bmi", "pulse_pressure", "map_est", "age_band"] if c in df_kaggle.columns])

# For later external validation / modeling
df_kaggle_fe = df_kaggle.copy()

fe_path = os.path.join(FIG_DIR_KAGGLE, "kg_dataset_featured.csv")
df_kaggle_fe.to_csv(fe_path, index=False)
print(f"\nSaved engineered Kaggle dataset to:\n  {fe_path}")
print("Final Kaggle shape with engineered features:", df_kaggle_fe.shape)


# 4.6 Simple correlation heatmap (numerics only)

num_cols_kaggle = df_kaggle_fe.select_dtypes(include=[np.number]).columns.tolist()

num_cols_no_target = [c for c in num_cols_kaggle if c != "cardio"]

corr_kaggle = df_kaggle_fe[num_cols_no_target].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_kaggle, interpolation="nearest")
plt.xticks(range(len(num_cols_no_target)), num_cols_no_target, rotation=90)
plt.yticks(range(len(num_cols_no_target)), num_cols_no_target)
plt.title("Kaggle: Correlation Heatmap (numeric features)")
plt.colorbar()
plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR_KAGGLE, "kg_corr_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()

corr_csv_path = os.path.join(FIG_DIR_KAGGLE, "kg_correlation_matrix.csv")
corr_kaggle.to_csv(corr_csv_path)

print("\nSaved Kaggle correlation heatmap to:", heatmap_path)
print("Saved Kaggle correlation matrix to:", corr_csv_path)
print("\nKaggle EDA + feature engineering complete.")


# In[10]:



# 5. Model development and evaluation – final logistic model (Cleveland)

# This section:
# - Loads the engineered Cleveland dataset from Section 3
# - Builds a preprocessing + logistic regression pipeline
# - Tunes C (regularization strength) with cross-validation
# - Calibrates probabilities (isotonic)
# - Evaluates on a held-out test set
# - Saves key plots and tables for the paper / slides


# In[11]:


# 5. Model development and evaluation – final logistic (Cleveland)

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


# 5.1 Load featured Cleveland dataset and set up X / y


# This CSV was created in Section 3
cle_model_path = os.path.join(FIG_DIR_CLEV, "cleveland_featured_for_modeling.csv")
df_clev_model = pd.read_csv(cle_model_path)

print("Loaded Cleveland modeling dataset from:")
print(" ", cle_model_path)
print("Shape:", df_clev_model.shape)

# Target column
y = df_clev_model["target"]

# Feature columns = everything except target
feature_cols = [c for c in df_clev_model.columns if c != "target"]
X = df_clev_model[feature_cols]

# Categorical columns = original Cleveland categoricals
# (age band dummies are already numeric)
cat_cols_model = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Numeric columns = everything else
num_cols_model = [c for c in X.columns if c not in cat_cols_model]

print("\nNumber of features:", len(feature_cols))
print("Numeric columns:", num_cols_model)
print("Categorical columns:", cat_cols_model)

# Train/test split (25% test, stratified by target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)
print("Train target balance:\n", y_train.value_counts(normalize=True).round(3))



# 5.2 Preprocessing and logistic pipeline


# Numeric: impute missing values with median, then standardize
numeric_steps = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: impute missing with most frequent, then one-hot encode
categorical_steps = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer to apply the right steps to each column type
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_steps, num_cols_model),
        ("cat", categorical_steps, cat_cols_model),
    ]
)

# Logistic regression model (class_weight="balanced" for small dataset)
logistic = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

# Full pipeline: preprocess -> logistic
logistic_pipe = Pipeline(steps=[
    ("pre", preprocess),
    ("clf", logistic)
])


# 5.3 Hyperparameter tuning (C) with cross-validation


# We tune only C (strength of regularization) to keep it simple
log_param_grid = {
    "clf__C": [0.1, 1.0, 3.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log_grid = GridSearchCV(
    estimator=logistic_pipe,
    param_grid=log_param_grid,
    scoring="roc_auc",     # good for ranking patients by risk
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("\nTuning logistic regression (scoring = ROC-AUC)...")
log_grid.fit(X_train, y_train)

print("Best CV ROC-AUC:", round(log_grid.best_score_, 3))
print("Best hyperparameters:", log_grid.best_params_)

best_logistic_pipe = log_grid.best_estimator_


# 5.4 Probability calibration (isotonic)


# Calibrate predicted probabilities to observed risk.
# Isotonic regression with 5-fold CV on the training set.
calibrated_logistic = CalibratedClassifierCV(
    base_estimator=best_logistic_pipe,
    method="isotonic",
    cv=5
)

print("\nFitting calibrated logistic model (isotonic)...")
calibrated_logistic.fit(X_train, y_train)

# Predicted probabilities on the test set (for class 1 = disease)
y_prob_test = calibrated_logistic.predict_proba(X_test)[:, 1]


# 5.5 Helper: evaluate at a given threshold


def eval_at_threshold(y_true, y_prob, threshold=0.50):
    """
    Turn probabilities into 0/1 predictions using the given threshold,
    then compute simple metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": (y_pred == y_true).mean()
    }
    return metrics, y_pred


# Evaluate at three thresholds: 0.50, 0.45, 0.40
metrics_050, y_pred_050 = eval_at_threshold(y_test, y_prob_test, threshold=0.50)
metrics_045, y_pred_045 = eval_at_threshold(y_test, y_prob_test, threshold=0.45)
metrics_040, y_pred_040 = eval_at_threshold(y_test, y_prob_test, threshold=0.40)

# Put metrics into a small table and save
metrics_df = pd.DataFrame([metrics_050, metrics_045, metrics_040])
metrics_path = os.path.join(FIG_DIR_CLEV, "cle_logistic_calibrated_test_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\n=== Test metrics for calibrated logistic (Cleveland) ===")
print(metrics_df)
print("\nSaved metrics table to:", metrics_path)


# 5.6 Confusion matrix plot (at threshold 0.45)


def plot_confusion_matrix_simple(y_true, y_pred, title, filename):
    """
    Simple 2x2 confusion matrix with counts in each cell.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (No disease)", "1 (Disease)"])
    ax.set_yticklabels(["0 (No disease)", "1 (Disease)"])

    # Add counts to each cell
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12, color="black")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR_CLEV, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix to:", out_path)

# Confusion matrix at 0.45 
plot_confusion_matrix_simple(
    y_test,
    y_pred_045,
    title="Confusion Matrix – Calibrated Logistic (threshold = 0.45)",
    filename="cm_logistic_calibrated_0p45.png"
)


# 5.7 ROC and Precision-Recall curves (test set)


plt.figure(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_test, name="Calibrated Logistic")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve – Calibrated Logistic (Cleveland test set)")
plt.tight_layout()
roc_path = os.path.join(FIG_DIR_CLEV, "roc_calibrated_logistic_cleveland.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print("Saved ROC curve to:", roc_path)

plt.figure(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_prob_test, name="Calibrated Logistic")
plt.title("Precision–Recall Curve – Calibrated Logistic (Cleveland test set)")
plt.tight_layout()
pr_path = os.path.join(FIG_DIR_CLEV, "pr_calibrated_logistic_cleveland.png")
plt.savefig(pr_path, dpi=150)
plt.close()
print("Saved PR curve to:", pr_path)


# 5.8 Calibration plot (test set)


# Build calibration curve on the test set
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob_test,
    n_bins=10,
    strategy="quantile"   # roughly equal-sized bins
)

plt.figure(figsize=(5, 5))
plt.plot(prob_pred, prob_true, marker="o", label="Calibrated logistic")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed event rate")
plt.title("Calibration Plot – Calibrated Logistic (Cleveland test)")
plt.legend()
plt.tight_layout()
cal_path = os.path.join(FIG_DIR_CLEV, "calibration_calibrated_logistic_cleveland.png")
plt.savefig(cal_path, dpi=150)
plt.close()
print("Saved calibration plot to:", cal_path)


# 5.9 Feature importance: logistic coefficients


def get_feature_names_from_preprocessor(fitted_preprocessor, num_cols, cat_cols):
    """
    Build a list of final feature names after preprocessing:
    numeric column names + one-hot expanded categorical names.
    """
    # Numeric features keep their original names
    num_feats = list(num_cols)

    # Get OneHotEncoder inside the "cat" pipeline
    cat_pipeline = fitted_preprocessor.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["onehot"]
    cat_feats = list(ohe.get_feature_names_out(cat_cols))

    return np.array(num_feats + cat_feats)

# Use the fitted preprocessor from the best pipeline
fitted_pre = best_logistic_pipe.named_steps["pre"]
feature_names = get_feature_names_from_preprocessor(
    fitted_pre,
    num_cols_model,
    cat_cols_model
)

# Extract logistic coefficients from the uncalibrated best pipeline
lr_clf = best_logistic_pipe.named_steps["clf"]
coef_series = pd.Series(lr_clf.coef_.ravel(), index=feature_names)

# Top 10 features by absolute coefficient 
top10 = coef_series.abs().sort_values(ascending=False).head(10)
top10_signed = coef_series.loc[top10.index]

# Save CSV for appendix and plotting
coef_csv_path = os.path.join(FIG_DIR_CLEV, "logistic_coefficients_top10.csv")
top10_signed.to_csv(coef_csv_path, header=["coefficient"])
print("\nSaved top-10 logistic coefficients to:", coef_csv_path)

# Simple bar plot for top-10 coefficients
plt.figure(figsize=(7, 3.5))
top10_signed.sort_values(key=np.abs).plot(kind="barh")
plt.title("Top 10 Logistic Coefficients (Cleveland)")
plt.xlabel("Coefficient (log-odds)")
plt.tight_layout()
coef_plot_path = os.path.join(FIG_DIR_CLEV, "logistic_coefficients_top10.png")
plt.savefig(coef_plot_path, dpi=150)
plt.close()
print("Saved top-10 coefficients plot to:", coef_plot_path)

print("\n=== Section 5 complete: Calibrated logistic model for Cleveland is trained and evaluated. ===")


# In[12]:



# 5.10 Threshold workload & cost per 1000 patients


# Simple "what if" workload/cost table for a screening program.
# Assumptions:
# - Screening 1000 patients
# - Each flagged patient (model-positive) costs $200 in follow-up



patients_per_1000 = 1000
cost_per_flagged  = 200   # dollars per flagged patient (placeholder)

thresholds = [0.50, 0.45, 0.40]

rows = []
for thr in thresholds:
    y_pred_thr = (y_prob_test >= thr).astype(int)
    flagged_rate = y_pred_thr.mean()
    
    # Round once, then reuse for cost
    flagged_per_1000 = int(round(flagged_rate * patients_per_1000))
    est_cost = flagged_per_1000 * cost_per_flagged

    rows.append({
        "Threshold": thr,
        "Flagged %": round(flagged_rate * 100, 1),
        "Flagged / 1000": flagged_per_1000,
        "Est. Cost / 1000": est_cost,
    })

work_df = pd.DataFrame(rows)
print("\n=== Workload & cost per 1000 patients (Cleveland test set) ===")
print(work_df)


# Save as CSV for appendix / paper
work_csv_path = os.path.join(FIG_DIR_CLEV, "cle_threshold_workload_cost.csv")
work_df.to_csv(work_csv_path, index=False)
print("Saved workload/cost table to:", work_csv_path)

# Small table-style figure for slides
fig, ax = plt.subplots(figsize=(5.0, 1.5))
ax.axis("off")

table = ax.table(
    cellText=work_df.values,
    colLabels=work_df.columns,
    cellLoc="center",
    loc="center"
)

# Slight vertical stretch so text is readable
table.scale(1.0, 1.3)

plt.tight_layout()
work_fig_path = os.path.join(FIG_DIR_CLEV, "cle_threshold_workload_cost.png")
plt.savefig(work_fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved workload/cost figure to:", work_fig_path)


# In[13]:


# 6. Reduced external validation – Cleveland vs Kaggle
#    (using overlapping / "comparable" features)


# Train a simpler logistic model on Cleveland using only
# Features that we can also build from the Kaggle cardio dataset.
# Apply that model to Kaggle as a rough external check.


# In[14]:



# 6.1 Build reduced feature set for Cleveland


# Reuse the Cleveland modeling data from Section 5
# (df_clev_model was loaded from cleveland_featured_for_modeling.csv)
reduced_features = ["age", "sex", "trestbps", "chol", "fbs"]

X_clev_red = df_clev_model[reduced_features].copy()
y_clev_red = df_clev_model["target"].astype(int)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clev_red,
    y_clev_red,
    test_size=0.25,
    stratify=y_clev_red,
    random_state=RANDOM_STATE
)

print("Reduced Cleveland – train shape:", Xc_train.shape,
      "test shape:", Xc_test.shape)

# 6.2 Load Kaggle cardio dataset and map to same features

df_kag = pd.read_csv(PATH_KAGGLE, sep=";")
print("Raw Kaggle cardio shape:", df_kag.shape)
print("Kaggle columns:", list(df_kag.columns))

# Basic cleaning (keep it simple)
df_kag = df_kag.drop_duplicates().copy()

# Convert age from days to years
df_kag["age_years"] = df_kag["age"] / 365.25

# Sex: Kaggle gender is usually 1 = female, 2 = male
df_kag["sex"] = (df_kag["gender"] == 2).astype(int)

# Resting systolic blood pressure ~ ap_hi
# Clip to a reasonable range to avoid extreme outliers
df_kag["trestbps"] = df_kag["ap_hi"].clip(80, 250)

# Rough cholesterol mapping: ordinal -> mg/dL-ish
chol_map = {
    1: 200,  # normal
    2: 240,  # above normal
    3: 280   # well above normal
}
df_kag["chol"] = df_kag["cholesterol"].map(chol_map)

# Fasting blood sugar–like flag from glucose category
df_kag["fbs"] = (df_kag["gluc"] > 1).astype(int)

# Target (1 = cardiovascular event)
df_kag["target"] = df_kag["cardio"].astype(int)

# Keep only the reduced features + target, rename age_years -> age
df_kag_red = (
    df_kag[["age_years", "sex", "trestbps", "chol", "fbs", "target"]]
    .rename(columns={"age_years": "age"})
    .dropna()
)

X_kag_red = df_kag_red[reduced_features].copy()
y_kag_red = df_kag_red["target"].astype(int)

print("Reduced Kaggle shape (after mapping & dropna):", X_kag_red.shape)



# 6.3 Train + calibrate reduced logistic model (Cleveland only)


# All reduced features are numeric, so we just use numeric preprocessing
num_cols_red = reduced_features

numeric_steps_red = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocess_red = ColumnTransformer(
    transformers=[
        ("num", numeric_steps_red, num_cols_red)
    ],
    remainder="drop"
)

logistic_red = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipe_red = Pipeline(steps=[
    ("pre", preprocess_red),
    ("clf", logistic_red)
])

param_grid_red = {
    "clf__C": [0.1, 1.0, 3.0]
}

cv_red = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gs_red = GridSearchCV(
    estimator=pipe_red,
    param_grid=param_grid_red,
    scoring="roc_auc",
    cv=cv_red,
    n_jobs=-1,
    refit=True
)

print("\nTuning reduced logistic model (Cleveland only, ROC-AUC)...")
gs_red.fit(Xc_train, yc_train)

print("Best CV ROC-AUC (reduced):", round(gs_red.best_score_, 3))
print("Best C (reduced):", gs_red.best_params_["clf__C"])

best_pipe_red = gs_red.best_estimator_

# Calibrate on Cleveland training data
calibrated_red = CalibratedClassifierCV(
    base_estimator=best_pipe_red,
    method="isotonic",
    cv=5
)

print("\nFitting calibrated reduced model (isotonic, Cleveland)...")
calibrated_red.fit(Xc_train, yc_train)

# Predicted probabilities
y_prob_clev = calibrated_red.predict_proba(Xc_test)[:, 1]
y_prob_kag  = calibrated_red.predict_proba(X_kag_red)[:, 1]


# 6.4 Compare performance: Cleveland test vs Kaggle external


# We reuse eval_at_threshold() from Section 5.
thresholds_ext = [0.50, 0.45, 0.40]

rows = []
for thr in thresholds_ext:
    m_clev, _ = eval_at_threshold(yc_test, y_prob_clev, threshold=thr)
    m_kag,  _ = eval_at_threshold(y_kag_red, y_prob_kag,  threshold=thr)

    m_clev["Dataset"] = f"Cleveland test (thr={thr:.2f})"
    m_kag["Dataset"]  = f"Kaggle external (thr={thr:.2f})"

    rows.extend([m_clev, m_kag])

metrics_ext_df = pd.DataFrame(rows)

# Put Dataset first for easier reading
cols_order = ["Dataset", "Threshold", "ROC_AUC", "PR_AUC",
              "Recall", "Precision", "F1", "Accuracy"]
metrics_ext_df = metrics_ext_df[cols_order]

print("\n=== Reduced external validation: Cleveland vs Kaggle ===")
print(metrics_ext_df)

# Save to CSV for paper / appendix
ext_metrics_path = os.path.join(FIG_DIR_CLEV, "cle_kaggle_reduced_external_metrics.csv")
metrics_ext_df.to_csv(ext_metrics_path, index=False)
print("\nSaved external validation metrics to:", ext_metrics_path)


# In[15]:



# 6.X Calibration plot: internal (Cleveland) vs external (Kaggle)

from sklearn.calibration import calibration_curve

# Calibration curve for Cleveland test (reduced model)
prob_true_int, prob_pred_int = calibration_curve(
    yc_test,           # true labels
    y_prob_clev,       # predicted probs on Cleveland test
    n_bins=10,
    strategy="quantile"
)

# Calibration curve for Kaggle external (reduced model)
prob_true_ext, prob_pred_ext = calibration_curve(
    y_kag_red,         # true labels
    y_prob_kag,        # predicted probs on Kaggle reduced
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred_int, prob_true_int, marker="o", label="Internal (Cleveland, reduced)")
plt.plot(prob_pred_ext, prob_true_ext, marker="s", label="External (Kaggle, reduced)")
plt.plot([0, 1], [0, 1], "g--", label="Perfect calibration")

plt.xlabel("Predicted probability")
plt.ylabel("Observed event rate")
plt.title("Calibration: Internal vs External (Reduced)")
plt.legend()
plt.tight_layout()

cal_reduced_path = os.path.join(FIG_DIR_CLEV, "calibration_internal_external_reduced.png")
plt.savefig(cal_reduced_path, dpi=150)
plt.close()

print("Saved internal vs external calibration plot to:", cal_reduced_path)


# In[16]:


# 7. Other machine learning analysis
# 7.1 Multi-model ROC comparison (Logistic / KNN / RF)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# 7.1.1 Define features and target for this section
#       reuse the same Cleveland modeling dataset as Section 5


# X_full = all features from df_clev_model (already engineered in Section 3)
X_full = df_clev_model[feature_cols].copy()

# y_full = target (0 = no disease, 1 = disease)
y_full = df_clev_model["target"].astype(int)

print("X_full shape:", X_full.shape)
print("y_full value counts:")
print(y_full.value_counts())


# 7.1.2 Train/test split for model comparison

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_full,
    y_full,
    test_size=0.25,
    stratify=y_full,
    random_state=RANDOM_STATE
)

print("\nMulti-model train shape:", X_train_m.shape,
      "test shape:", X_test_m.shape)

# 7.1.3 Shared preprocessing: one-hot encode categoricals,
#       standardize numerics
#       (reuse cat_cols_model, num_cols_model from Section 5)


preproc_m = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols_model),
        ("num", StandardScaler(), num_cols_model),
    ],
    remainder="drop",
)

# 7.1.4 Define models


# (a) Logistic regression (baseline)
log_untuned = Pipeline([
    ("pre", preproc_m),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
])

# (b) Logistic regression with regularization (C=2.0)
log_tuned = Pipeline([
    ("pre", preproc_m),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear", C=2.0))
])

# (c) K-Nearest Neighbors (KNN)
knn_tuned = Pipeline([
    ("pre", preproc_m),
    ("clf", KNeighborsClassifier(n_neighbors=7))
])

# (d) Random Forest (shallow to avoid too much overfitting)
rf_tuned = Pipeline([
    ("pre", preproc_m),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=RANDOM_STATE
    ))
])


# 7.1.5 Fit all models and compute ROC-AUC on the same test set


models = {
    "Logistic_Untuned": log_untuned,
    "Logistic_Tuned":   log_tuned,
    "KNN_Tuned":        knn_tuned,
    "RF_Tuned":         rf_tuned,
}

auc_rows = []

print("\nFitting models and computing ROC-AUC on multi-model test set...")
for name, model in models.items():
    model.fit(X_train_m, y_train_m)
    y_proba = model.predict_proba(X_test_m)[:, 1]
    auc = roc_auc_score(y_test_m, y_proba)
    auc_rows.append({"Model": name, "ROC-AUC": auc})
    print(f"  {name:16s} | Test ROC-AUC = {auc:.3f}")

auc_df = (
    pd.DataFrame(auc_rows)
    .sort_values("ROC-AUC", ascending=False)
    .reset_index(drop=True)
)

print("\nMulti-model ROC-AUC comparison (higher is better):")
display(auc_df.style.format({"ROC-AUC": "{:.3f}"}))


# 7.1.6 Plot ROC curves for all models on the same figure


plt.figure(figsize=(8, 6))

for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test_m,
        y_test_m,
        name=name,
        ax=plt.gca()
    )

# Diagonal reference line (no-skill classifier)
plt.plot([0, 1], [0, 1], "--", color="purple")

# Update legend labels to include AUC values
handles, labels = plt.gca().get_legend_handles_labels()
label_to_auc = {row["Model"]: row["ROC-AUC"] for _, row in auc_df.iterrows()}

new_labels = []
for label in labels:
    auc = label_to_auc.get(label, float("nan"))
    new_labels.append(f"{label} (AUC = {auc:.2f})")

plt.legend(handles, new_labels, loc="lower right")
plt.title("ROC Curves (Test) – Logistic, KNN, Random Forest")
plt.tight_layout()

mm_roc_path = os.path.join(FIG_DIR_CLEV, "roc_multi_model.png")
plt.savefig(mm_roc_path, dpi=150)
plt.show()
plt.close()

print("Saved multi-model ROC plot to:", mm_roc_path)
print("\n=== Section 7.1 complete: multi-model comparison finished. ===")


# In[18]:


# 7.2 PCA + K-Means + KNN visualizations


# In[19]:



# 7.2 PCA + K-Means + KNN visualizations (Cleveland)


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 7.2.0 Figure directory for PCA/KNN plots


# Save PCA/KNN figures 
FIG_DIR_PCA_KNN = os.path.join(FIG_DIR_CLEV, "pca_knn")
os.makedirs(FIG_DIR_PCA_KNN, exist_ok=True)

# 7.2.1 Prepare data and preprocessing

# Target and features from the Cleveland modeling dataframe
y_full = df_clev_model["target"].astype(int)
X_full = df_clev_model.drop(columns=["target"])

# Reuse the same categorical and numeric columns as in the logistic model
cat_cols = cat_cols_model
num_cols = num_cols_model

print("X_full shape:", X_full.shape)
print("y_full shape:", y_full.shape)

# Train/test split 
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_full,
    y_full,
    test_size=0.25,
    stratify=y_full,
    random_state=RANDOM_STATE,
)

print("Train shape:", X_train_p.shape, "Test shape:", X_test_p.shape)

# Preprocessing: one-hot encode categorical variables, standardize numeric ones
preproc_p = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

# PCA down to 2 components so we can plot them
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)

# Pipeline: preprocess -> PCA
pipe_pca = Pipeline(steps=[
    ("pre", preproc_p),
    ("pca", pca2),
])

# Fit on train, transform both train and test
X_train_pca = pipe_pca.fit_transform(X_train_p, y_train_p)
X_test_pca  = pipe_pca.transform(X_test_p)

print("PCA shapes (train, test):", X_train_pca.shape, X_test_pca.shape)


# 7.2.2 K-Means clusters on PCA-2 (train set)


kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(X_train_pca)

plt.figure(figsize=(8, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means (k=2) Clusters on PCA-2D (Train)")
plt.tight_layout()

kmeans_path = os.path.join(FIG_DIR_PCA_KNN, "kmeans_pca_train.png")
plt.savefig(kmeans_path, dpi=300)
plt.show()
plt.close()
print("Saved K-Means PCA figure to:", kmeans_path)

# 7.2.3 Ground-truth labels on PCA-2 (train set)

plt.figure(figsize=(8, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_p)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Ground Truth Labels on PCA-2D (Train)")
plt.tight_layout()

labels_path = os.path.join(FIG_DIR_PCA_KNN, "labels_pca_train.png")
plt.savefig(labels_path, dpi=300)
plt.show()
plt.close()
print("Saved ground truth PCA figure to:", labels_path)


# 7.2.4 KNN predicted classes on PCA-2 (test set)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_pca, y_train_p)

y_test_pred_knn = knn.predict(X_test_pca)

plt.figure(figsize=(8, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pred_knn)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KNN Predicted Classes on PCA-2D (Test)")
plt.tight_layout()

knn_path = os.path.join(FIG_DIR_PCA_KNN, "knn_pred_pca_test.png")
plt.savefig(knn_path, dpi=300)
plt.show()
plt.close()
print("Saved KNN PCA prediction figure to:", knn_path)

print("\n=== Section 7.2 complete: PCA + K-Means + KNN visualizations saved. ===")



# In[20]:



# 7.3 False Negatives vs False Positives bar chart (thr=0.45)

from sklearn.metrics import confusion_matrix

# Reuse y_test and y_prob_test from the final calibrated logistic model.
# y_test: true labels (0 = no disease, 1 = disease)
# y_prob_test: predicted probabilities for class 1 (disease)

threshold = 0.45  # recall-first focus threshold

# Turn probabilities into 0/1 predictions
y_true = y_test
y_pred = (y_prob_test >= threshold).astype(int)

# Confusion matrix: [[TN, FP],
#                    [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Bar chart comparing FN and FP counts
labels = ["False Negatives", "False Positives"]
values = [fn, fp]

plt.figure(figsize=(4.5, 3.5))
bars = plt.bar(labels, values)

# Add counts above the bars
for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val,
        str(val),
        ha="center",
        va="bottom"
    )

plt.title(f"Errors at Threshold = {threshold}")
plt.ylabel("Count")
plt.tight_layout()

# Save figure inside the Cleveland figs folder
out_path = os.path.join(FIG_DIR_CLEV, f"fn_fp_bar_{str(threshold).replace('.', '')}.png")
plt.savefig(out_path, dpi=300)
plt.show()
plt.close()

print("Saved FN/FP chart to:", out_path)
print(f"False Negatives (FN) = {fn}, False Positives (FP) = {fp}")


# In[22]:



# 7.4 XGBoost-only model on Cleveland (for comparison)

from xgboost import XGBClassifier

cle_model_path = os.path.join(FIG_DIR_CLEV, "cleveland_featured_for_modeling.csv")
df_clev_model_xgb = pd.read_csv(cle_model_path)

print("Loaded Cleveland modeling dataset for XGBoost from:")
print(" ", cle_model_path)
print("Shape:", df_clev_model_xgb.shape)

# Target and features
y_xgb = df_clev_model_xgb["target"]
X_xgb = df_clev_model_xgb.drop(columns=["target"])

# Same categorical and numeric columns as in Section 5
cat_cols_xgb = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
num_cols_xgb = [c for c in X_xgb.columns if c not in cat_cols_xgb]

print("\nXGBoost feature setup:")
print("Numeric columns:", num_cols_xgb)
print("Categorical columns:", cat_cols_xgb)

# Train/test split
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb,
    test_size=0.25,
    stratify=y_xgb,
    random_state=RANDOM_STATE
)

print("\nXGBoost train/test shapes:")
print("  X_train_xgb:", X_train_xgb.shape)
print("  X_test_xgb :", X_test_xgb.shape)

# Preprocessing: impute + scale numerics, impute + one-hot categoricals
numeric_steps_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_steps_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preproc_xgb = ColumnTransformer(
    transformers=[
        ("num", numeric_steps_xgb, num_cols_xgb),
        ("cat", categorical_steps_xgb, cat_cols_xgb),
    ]
)

# Simple XGBoost model
xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

xgb_pipe = Pipeline(steps=[
    ("pre", preproc_xgb),
    ("clf", xgb_clf),
])

print("\nFitting XGBoost pipeline on Cleveland...")
xgb_pipe.fit(X_train_xgb, y_train_xgb)

# Predicted probabilities and basic metrics on test set
y_prob_xgb = xgb_pipe.predict_proba(X_test_xgb)[:, 1]
y_pred_xgb = (y_prob_xgb >= 0.50).astype(int)  # standard 0.5 threshold

roc_xgb = roc_auc_score(y_test_xgb, y_prob_xgb)
pr_xgb  = average_precision_score(y_test_xgb, y_prob_xgb)
rec_xgb = recall_score(y_test_xgb, y_pred_xgb)
pre_xgb = precision_score(y_test_xgb, y_pred_xgb)
f1_xgb  = f1_score(y_test_xgb, y_pred_xgb)
acc_xgb = (y_pred_xgb == y_test_xgb).mean()

print("\n=== XGBoost test metrics (Cleveland, thr = 0.50) ===")
print(f"ROC-AUC : {roc_xgb:.3f}")
print(f"PR-AUC  : {pr_xgb:.3f}")
print(f"Recall  : {rec_xgb:.3f}")
print(f"Precision: {pre_xgb:.3f}")
print(f"F1-score: {f1_xgb:.3f}")
print(f"Accuracy: {acc_xgb:.3f}")

# ROC curve for XGBoost
plt.figure(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test_xgb,
    y_prob_xgb,
    name="XGBoost (Cleveland test)"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve – XGBoost (Cleveland test set)")
plt.tight_layout()

xgb_roc_path = os.path.join(FIG_DIR_MODEL, "roc_xgboost_cleveland.png")
plt.savefig(xgb_roc_path, dpi=150)
plt.close()
print("Saved XGBoost ROC curve to:", xgb_roc_path)


# In[ ]:





# In[ ]:




