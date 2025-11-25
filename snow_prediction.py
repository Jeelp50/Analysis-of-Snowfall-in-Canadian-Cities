# ========================================================================
# SNOWFALL PREDICTIVE MODEL — SPATIAL + TEMPORAL + kNN GRAPH
# ========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Machine learning imports
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ========================================================================
# CONFIG
# ========================================================================

OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This file should already be generated from your EDA pipeline
MERGED_DATASET_PATH = "canswe_station_eda\station_winter_summary_with_meta.csv"

# ========================================================================
# LOAD MERGED DATASET
# ========================================================================

def load_merged_dataset(path):
    """
    Loads the merged station-year dataset.
    Must contain:
        station_index, station_name, lat, lon, elevation,
        year, snw_total, snw_mean, snd_total, snd_mean
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ERROR: Cannot find merged dataset at {path}\n"
            "Make sure you exported it from your EDA script."
        )

    df = pd.read_csv(path)

    # Basic validation
    required_cols = [
        "station_index", "station_name", "lat", "lon",
        "elevation", "year",
        "snw_total", "snw_mean"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    print(f"✓ Loaded dataset with {len(df)} station-year rows")
    return df

# ========================================================================
# PREDICTIVE BLOCK 1 — Build kNN Graph
# ========================================================================

def build_knn_graph(merged_df, k=5):
    """
    Builds a k-nearest-neighbors graph (kNN) using station latitude/longitude.
    
    Args:
        merged_df : DataFrame with station metadata and SWE
        k : number of neighbors for each station

    Returns:
        knn_graph : dict mapping station_index -> list of neighbor station_index
    """

    # Ensure we only keep unique station-level coordinates
    stations = merged_df[["station_index", "lat", "lon"]].drop_duplicates()

    station_ids = stations["station_index"].values
    coords = stations[["lat", "lon"]].values  # shape = (num_stations, 2)

    # Build kNN (k+1 because the closest is itself)
    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(coords)

    distances, indices = knn.kneighbors(coords)

    # Construct graph
    knn_graph = {}

    for i, station in enumerate(station_ids):
        neighbor_indices = indices[i][1:]   # skip itself (index 0)
        neighbor_station_ids = station_ids[neighbor_indices]
        knn_graph[station] = neighbor_station_ids.tolist()

    print(f"✓ kNN graph constructed: {len(station_ids)} stations, k={k}")
    return knn_graph

# ========================================================================
# PREDICTIVE BLOCK 2 — Build Spatio-Temporal Feature Table
# ========================================================================

def build_feature_table(merged_df, knn_graph):
    """
    Builds the feature table used for prediction.
    For each station-year row, computes:
        - last_year_swe
        - neighbor_last_year_swe
        - lat, lon, elevation
        - target (current year SWE)

    Returns:
        features_df : DataFrame ready to be split into train/test
    """

    df = merged_df.copy()

    # ---- TEMPORAL FEATURE: Last year's SWE ----
    df["last_year_swe"] = df.sort_values(["station_index", "year"]) \
                            .groupby("station_index")["snw_total"].shift(1)

    # ---- SPATIAL FEATURE: Neighbor SWE from last year ----
    neighbor_swe_list = []

    # Pre-group by year for faster lookup
    swe_by_station_year = df.set_index(["station_index", "year"])["snw_total"]

    for idx, row in df.iterrows():
        station = row["station_index"]
        year = row["year"]

        neighbors = knn_graph.get(station, [])
        neighbor_values = []

        for nb in neighbors:
            key = (nb, year - 1)  # neighbor's SWE from last year
            if key in swe_by_station_year:
                neighbor_values.append(swe_by_station_year[key])

        if len(neighbor_values) > 0:
            neighbor_swe_list.append(np.mean(neighbor_values))
        else:
            neighbor_swe_list.append(np.nan)

    df["neighbor_last_year_swe"] = neighbor_swe_list

    # ---- Drop rows where temporal or spatial features are missing ----
    df = df.dropna(subset=["last_year_swe", "neighbor_last_year_swe"])

    # ---- FINAL FEATURE TABLE ----
    feature_cols = [
        "station_index",
        "year",
        "lat",
        "lon",
        "elevation",
        "last_year_swe",
        "neighbor_last_year_swe",
    ]

    target_col = "snw_total"

    features_df = df[feature_cols + [target_col]].reset_index(drop=True)

    print(f"✓ Feature table created: {len(features_df)} rows")
    return features_df

# ========================================================================
# PREDICTIVE BLOCK 3 — Time-Aware Train/Test Split
# ========================================================================

def train_test_split_timewise(features_df, test_ratio=0.2):
    """
    Splits the feature table into train and test sets chronologically.
    
    Args:
        features_df : DataFrame produced by build_feature_table()
        test_ratio : fraction of years to reserve for testing (default: 20%)

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Sort by year to avoid leakage
    df = features_df.sort_values("year").reset_index(drop=True)

    # Identify all years in the dataset
    years = df["year"].unique()
    years.sort()

    # Compute cutoff year for test split
    num_test_years = max(1, int(len(years) * test_ratio))
    cutoff_year = years[-num_test_years]

    print(f"✓ Training on years < {cutoff_year}")
    print(f"✓ Testing on years ≥ {cutoff_year}")

    # boolean mask
    train_mask = df["year"] < cutoff_year
    test_mask = df["year"] >= cutoff_year

    train_df = df[train_mask]
    test_df  = df[test_mask]

    # Feature columns
    feature_cols = [
        "lat", "lon", "elevation",
        "last_year_swe", "neighbor_last_year_swe"
    ]

    target_col = "snw_total"

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    print(f"✓ Train size: {len(X_train)} rows")
    print(f"✓ Test size:  {len(X_test)} rows")

    return X_train, X_test, y_train, y_test

# ========================================================================
# PREDICTIVE BLOCK 4 — Train Linear Regression Model
# ========================================================================

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Fits a linear regression model on the spatio-temporal features and evaluates it.
    
    Returns:
        model : trained LinearRegression object
        y_pred : predictions on the test set
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(y_test - y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n=== LINEAR REGRESSION RESULTS ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R^2:  {r2:.3f}")

    return model, y_pred

# ========================================================================
# PREDICTIVE BLOCK 5 — Predicted vs Actual Plot
# ========================================================================

def plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual SWE"):
    """
    Creates a scatter plot comparing true winter SWE vs model predictions.
    Saves the figure to OUTPUT_DIR.
    """

    plt.figure(figsize=(8, 8))

    # Scatter
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="black", linewidth=0.5)

    # 1:1 reference line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 Line")

    plt.xlabel("Actual SWE", fontsize=13)
    plt.ylabel("Predicted SWE", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.legend()

    out_path = os.path.join(OUTPUT_DIR, "Predicted_vs_Actual_SWE.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved predicted-vs-actual plot: {out_path}")

# ========================================================================
# PREDICTIVE BLOCK 6 — Feature Importance (Linear Regression Coefficients)
# ========================================================================

def plot_feature_importance_linear(model, feature_names):
    """
    Plots linear regression feature coefficients.
    Positive coefficients increase predicted SWE,
    negative coefficients decrease it.
    """

    coeffs = model.coef_

    plt.figure(figsize=(8, 6))
    bars = plt.barh(feature_names, coeffs, color="#4E79A7")

    plt.xlabel("Coefficient Value", fontsize=12)
    plt.title("Feature Importance (Linear Regression Coefficients)", fontsize=14, fontweight="bold")
    plt.axvline(0, color="black", linewidth=1)

    # Add values next to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f"{width:.2f}", va="center", fontsize=10)

    out_path = os.path.join(OUTPUT_DIR, "LinearRegression_FeatureImportance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved feature importance plot: {out_path}")

# ========================================================================
# PREDICTIVE BLOCK 7 — Per-City Prediction Plot (OTTAWA)
# ========================================================================

def plot_city_predictions(merged_df, model, features_df, city_name="OTTAWA"):
    """
    Plots actual vs predicted SWE over time for a specific city.
    Uses the model to generate predictions across all years for that city.
    """

    # Filter merged dataset for OTTAWA-like stations
    city_df = merged_df[merged_df["station_name"].str.contains(city_name, case=False, na=False)]

    if city_df.empty:
        print(f"⚠ No station found for city name containing '{city_name}'.")
        return

    station_ids = city_df["station_index"].unique()

    # Pull feature rows for these stations
    city_features = features_df[features_df["station_index"].isin(station_ids)].copy()

    if city_features.empty:
        print(f"⚠ No feature rows found for {city_name}.")
        return

    # Extract input features
    X_city = city_features[["lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"]].values

    # Predict using model
    city_features["prediction"] = model.predict(X_city)

    # Plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        city_features["year"],
        city_features["snw_total"],
        marker="o",
        linewidth=2,
        label="Actual SWE",
        color="#4E79A7"
    )

    plt.plot(
        city_features["year"],
        city_features["prediction"],
        marker="o",
        linewidth=2,
        label="Predicted SWE",
        color="#F28E2B"
    )

    plt.title(f"Actual vs Predicted Winter SWE — {city_name.title()}", fontsize=15, fontweight="bold")
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("Winter SWE (mm)", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()

    out_path = os.path.join(OUTPUT_DIR, f"{city_name.title()}_Predictions.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved {city_name.title()} prediction plot: {out_path}")



# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    print("="*80)
    print("SNOWFALL PREDICTIVE MODEL (Spatial + Temporal + kNN Graph)")
    print("="*80)

    merged = load_merged_dataset(MERGED_DATASET_PATH)

    knn_graph = build_knn_graph(merged, k=5)
    features_df = build_feature_table(merged, knn_graph)
    X_train, X_test, y_train, y_test = train_test_split_timewise(features_df)

    # ========================================================================
    # FIX BLOCK — Drop Remaining NaNs
    # ========================================================================

    # Combine X and y temporarily to ensure we drop matching rows
    train_df = pd.DataFrame(X_train, columns=[
        "lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"
    ])
    train_df["target"] = y_train

    # Drop any rows with NaN
    train_df = train_df.dropna()

    # Extract cleaned arrays
    X_train = train_df[["lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"]].values
    y_train = train_df["target"].values

    # Repeat for test set
    test_df = pd.DataFrame(X_test, columns=[
        "lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"
    ])
    test_df["target"] = y_test

    test_df = test_df.dropna()

    X_test = test_df[["lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"]].values
    y_test = test_df["target"].values

    print("✓ Cleaned datasets — no NaNs remain.")
    print(f"New train shape: {X_train.shape}")
    print(f"New test shape:  {X_test.shape}")

    model_lr, y_pred_lr = train_linear_regression(X_train, y_train, X_test, y_test)
    plot_predicted_vs_actual(y_test, y_pred_lr, 
                         title="Linear Regression: Predicted vs Actual SWE")
    feature_names = ["lat", "lon", "elevation", "last_year_swe", "neighbor_last_year_swe"]
    plot_feature_importance_linear(model_lr, feature_names)
    plot_city_predictions(merged, model_lr, features_df, city_name="OTTAWA")
