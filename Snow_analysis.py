import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# === Paths & output directory ===
CANSWE_FILE = "CanSWE-CanEEN_1928-2020_v1.nc"
OUTPUT_DIR = "canswe_station_eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Major Canadian cities we want to highlight (used later)
CITY_KEYWORDS = {
    "Toronto": ["TORONTO"],
    "Montreal": ["MONTREAL"],
    "Vancouver": ["VANCOUVER"],
    "Calgary": ["CALGARY"],
    "Edmonton": ["EDMONTON"],
    "Ottawa": ["OTTAWA"],
    "Winnipeg": ["WINNIPEG"],
    "Quebec": ["QUEBEC"],
    "Halifax": ["HALIFAX"],
    "Regina": ["REGINA"],
    "Saskatoon": ["SASKATOON"],
    "Victoria": ["VICTORIA"],
    "St. John's": ["ST JOHN", "ST. JOHN", "ST. JOHN'S"],
}

# ========================================================================
# 1. LOAD DATASET AND METADATA
# ========================================================================

def load_canswe_dataset(path=CANSWE_FILE):
    """
    Load the CanSWE NetCDF dataset.
    Assumes:
      - snow water equivalent variable: 'snw'
      - snow depth variable: 'snd'
      - time coordinate: 'time'
    """
    ds = xr.open_dataset(path)
    print("Loaded dataset:")
    print(ds)
    return ds


def extract_station_metadata(ds):
    """
    Build a metadata DataFrame for all stations.

    Includes:
      - station_index
      - station_name
      - lat, lon
      - elevation
      - station_id (if present)
    """
    # Identify dimension names from snw
    station_dim, time_dim = ds["snw"].dims

    meta = pd.DataFrame({
        "station_index": np.arange(ds.dims[station_dim]),
        "station_name": ds["station_name"].values.astype(str),
        "lat": ds["lat"].values.astype(float),
        "lon": ds["lon"].values.astype(float),
        "elevation": ds["elevation"].values.astype(float),
    })

    if "station_id" in ds:
        meta["station_id"] = ds["station_id"].values.astype(str)

    if "source" in ds:
        meta["source"] = ds["source"].values.astype(str)

    out_path = os.path.join(OUTPUT_DIR, "station_metadata.csv")
    meta.to_csv(out_path, index=False)
    print(f"✓ Saved station metadata: {out_path}")

    return meta, station_dim, time_dim


# ========================================================================
# 2. OPTIMIZED WINTER AGGREGATION (SWE + SNOW DEPTH)
# ========================================================================

def compute_winter_aggregates_fast(ds, station_dim, time_dim, winter_months=(11, 12, 1, 2)):
    """
    Efficient winter aggregation using NumPy + Pandas.

    Produces one row per (station, winter_year) containing:
      - snw_total, snw_mean, snw_max
      - snd_total, snd_mean, snd_max

    Winter year definition:
      - Nov, Dec -> that year
      - Jan, Feb -> previous year
    """
    print("✓ Using optimized NumPy/Pandas winter aggregation")

    # 1. Extract time as pandas datetime
    time = ds[time_dim].values
    pd_time = pd.to_datetime(time)

    month = pd_time.month.astype(np.int16)
    year = pd_time.year.astype(np.int32)

    # 2. Compute winter_year (mutable array)
    winter_year = np.array(year, dtype=np.int32)
    winter_year[np.isin(month, [1, 2])] -= 1  # Jan + Feb -> previous winter

    # 3. Select only winter months
    winter_mask = np.isin(month, winter_months)
    winter_indices = np.where(winter_mask)[0]

    # 4. Extract SNW and SND arrays (shape = stations × time)
    snw = ds["snw"].values
    snd = ds["snd"].values

    snw_w = snw[:, winter_indices]
    snd_w = snd[:, winter_indices]
    wy = winter_year[winter_indices]

    nstations, nwin = snw_w.shape

    # 5. Flatten into long DataFrame
    df = pd.DataFrame({
        "station_index": np.repeat(np.arange(nstations), nwin),
        "winter_year": np.tile(wy, nstations),
        "snw": snw_w.reshape(-1),
        "snd": snd_w.reshape(-1),
    })

    # 6. Group + aggregate results
    summary = df.groupby(["station_index", "winter_year"]).agg(
        snw_total=("snw", "sum"),
        snw_mean=("snw", "mean"),
        snw_max=("snw", "max"),

        snd_total=("snd", "sum"),
        snd_mean=("snd", "mean"),
        snd_max=("snd", "max"),
    ).reset_index()

    # Rename winter_year → year
    summary = summary.rename(columns={"winter_year": "year"})

    out_path = os.path.join(OUTPUT_DIR, "station_winter_summary_optimized.csv")
    summary.to_csv(out_path, index=False)
    print(f"✓ Saved optimized winter summary: {out_path}")

    return summary


# ========================================================================
# 3. MERGE SUMMARY WITH METADATA
# ========================================================================

def merge_summary_with_metadata(summary_df, meta_df):
    """
    Attach station_name, lat, lon, and elevation to the winter summary.
    """
    merged = summary_df.merge(meta_df, on="station_index", how="left")
    out_path = os.path.join(OUTPUT_DIR, "station_winter_summary_with_meta.csv")
    merged.to_csv(out_path, index=False)
    print(f"✓ Saved winter summary with metadata: {out_path}")
    return merged

import matplotlib.pyplot as plt

# ========================================================================
# EDA BLOCK 1 — Total SWE per Year (Unfiltered)
# ========================================================================

def plot_total_swe_by_year(df):
    """
    Bar graph of total winter SWE grouped by year.
    No filtering — includes ALL stations.
    """
    # Total SWE per winter year across all stations
    yearly = df.groupby("year")["snw_total"].sum()

    plt.figure(figsize=(14, 6))
    plt.bar(yearly.index, yearly.values, color="#4C72B0", edgecolor="black")

    plt.title(
        "Total Winter SWE Across All Stations (Unfiltered)",
        fontsize=15, fontweight="bold"
    )
    plt.xlabel("Winter Year", fontsize=12)
    plt.ylabel("Total Snow Water Equivalent (SWE)", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Total_SWE_by_Year.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 2 — Major-City SWE Correlation Heatmap (0–1 scale, fuzzy city matching)
# ========================================================================

def detect_city(station_name):
    """
    Returns the major city a station belongs to (if any) using fuzzy keyword matching.
    """
    name = station_name.upper()
    for city, keywords in CITY_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return city
    return None  # Not a major city


def plot_station_correlation_heatmap(df):
    """
    Build a correlation heatmap across major Canadian cities
    using fuzzy matching to identify city groups.
    """
    # Add a city column using fuzzy detection
    df["city_group"] = df["station_name"].apply(detect_city)

    # Keep only identified major cities
    major_df = df[df["city_group"].notna()]

    if major_df.empty:
        print("⚠ No major city stations detected after fuzzy matching.")
        return

    # Pivot: rows = years, columns = city groups
    pivot = major_df.pivot_table(
        index="year",
        columns="city_group",
        values="snw_total",
        aggfunc="mean"  # average across stations in same metro area
    )

    # Drop columns with all-NaN values
    pivot = pivot.dropna(axis=1, how="all")

    corr = pivot.corr().clip(0, 1)  # limit to 0–1

    # --- Plot ---
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(im, label="Correlation (0–1)")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=9)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=9)

    plt.title(
        "Winter SWE Correlation Between Major Canadian Cities (Fuzzy Matching)",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Major_City_SWE_Correlation_Heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 3 — Scatter of Total SWE per Station (Filtered)
# ========================================================================

def plot_swe_scatter_by_station(df, threshold=100000):
    """
    Scatter plot of total SWE per station (summed over all winters),
    but only for stations that exceed a SWE threshold.

    This removes noise from very low-SWE stations and improves clarity.
    """
    # Compute total SWE per station across all winters
    station_totals = df.groupby("station_name")["snw_total"].sum().reset_index()

    # Filter by threshold
    filtered = station_totals[station_totals["snw_total"] > threshold]

    if filtered.empty:
        print(f"⚠ No stations exceed SWE threshold = {threshold}. Scatter plot skipped.")
        return

    # Sort for better layering (smaller first)
    filtered = filtered.sort_values(by="snw_total")

    plt.figure(figsize=(16, 7))
    plt.scatter(
        filtered["station_name"],
        filtered["snw_total"],
        c="#1f77b4",        # visible blue
        edgecolor="black",  # thin outline
        linewidth=0.6,
        alpha=0.75,
        s=55                # larger markers
    )

    plt.xticks(rotation=90, fontsize=8)
    plt.title(
        f"Total Winter SWE per Station (Stations with SWE > {threshold})",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("Station Name", fontsize=12)
    plt.ylabel("Total Snow Water Equivalent (SWE)", fontsize=12)
    plt.grid(alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Total_SWE_Scatter_Filtered.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK — Scatter of Total SWE for Major Canadian Cities (Fuzzy Matching)
# ========================================================================

def plot_major_city_swe_scatter(df):
    """
    Scatter plot of total SWE aggregated by major Canadian cities,
    using fuzzy keyword matching to group multiple stations in each metro area.
    """

    # Reuse the fuzzy city-detection function
    df["city_group"] = df["station_name"].apply(detect_city)

    # Keep only matched major cities
    major_df = df[df["city_group"].notna()]

    if major_df.empty:
        print("⚠ No major cities detected for scatter plot.")
        return

    # Aggregate total SWE by city
    city_totals = major_df.groupby("city_group")["snw_total"].sum().reset_index()

    # Sort for clarity
    city_totals = city_totals.sort_values(by="snw_total")

    plt.figure(figsize=(12, 6))
    plt.scatter(
        city_totals["city_group"],
        city_totals["snw_total"],
        c="#D95F02",        # orange for contrast
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        s=120               # larger markers since fewer points
    )

    plt.xticks(rotation=45, fontsize=10)
    plt.title(
        "Total Winter SWE by Major Canadian City (Fuzzy Station Matching)",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Total Snow Water Equivalent (SWE)", fontsize=12)
    plt.grid(alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Major_City_Total_SWE_Scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 4 — Winter SWE Anomaly (Deviation from Long-Term Mean)
# ========================================================================

def plot_yearly_swe_anomalies(df):
    """
    Bar plot of yearly SWE anomalies relative to the long-term mean.
    Positive anomaly = snowier winter than average.
    Negative anomaly = milder winter than average.
    """
    # Total SWE per winter year across all stations
    yearly = df.groupby("year")["snw_total"].sum()

    # Mean SWE across all years
    mean_swe = yearly.mean()

    # Anomaly = difference from long-term mean
    anomaly = yearly - mean_swe

    plt.figure(figsize=(14, 6))
    # Positive = red, Negative = blue
    colors = ["#D62728" if val > 0 else "#1F77B4" for val in anomaly.values]

    plt.bar(anomaly.index, anomaly.values, color=colors, edgecolor="black")

    # Reference line
    plt.axhline(0, color="black", linewidth=1.2)

    plt.title(
        "Winter SWE Anomalies (Deviation from Long-Term Mean)",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("Winter Year", fontsize=12)
    plt.ylabel("SWE Anomaly (mm SWE)", fontsize=12)
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Winter_SWE_Anomalies.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 5 — Winter SWE Trends for Major Canadian Cities (Fuzzy Matching)
# ========================================================================

def plot_station_trends_major_cities(df):
    """
    Plot long-term winter SWE trends for major Canadian cities using fuzzy
    station matching to group multiple stations under each metro area.
    """
    # Tag each station with its detected city group
    df["city_group"] = df["station_name"].apply(detect_city)

    # Keep only rows that belong to major cities
    major_df = df[df["city_group"].notna()]

    if major_df.empty:
        print("⚠ No major city data found for trend plotting.")
        return

    # Unique detected cities
    major_cities_found = sorted(major_df["city_group"].unique())

    plt.figure(figsize=(14, 8))

    for city in major_cities_found:
        city_df = major_df[major_df["city_group"] == city]

        # Group all stations in that city by year and average SWE
        city_yearly = (
            city_df.groupby("year")["snw_total"]
            .mean()  # average across all stations in that city
            .reset_index()
        )

        plt.plot(
            city_yearly["year"],
            city_yearly["snw_total"],
            marker="o",
            linewidth=2.2,
            alpha=0.9,
            label=city
        )

    plt.title(
        "Winter SWE Trends for Major Canadian Cities\n(Fuzzy Station Grouping)",
        fontsize=17,
        fontweight="bold"
    )
    plt.xlabel("Winter Year", fontsize=13)
    plt.ylabel("Total Snow Water Equivalent (SWE)", fontsize=13)
    plt.grid(alpha=0.30)
    plt.legend(title="City", fontsize=10)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Major_City_Winter_SWE_Trends.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 6 — Spatial Map of Mean Winter SWE
# ========================================================================

def plot_spatial_mean_swe(df):
    """
    Spatial scatter plot of stations colored by mean winter SWE.

    Uses latitude (lat) and longitude (lon) from metadata,
    and averages snw_mean across all years for each station.
    """

    # Group by station to compute long-term average SWE
    station_means = (
        df.groupby(["station_index", "station_name", "lat", "lon"])["snw_mean"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(11, 7))
    sc = plt.scatter(
        station_means["lon"],
        station_means["lat"],
        c=station_means["snw_mean"],
        cmap="Blues",
        s=22,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9
    )

    plt.colorbar(sc, label="Mean Winter SWE (mm)")
    plt.title(
        "Spatial Distribution of Mean Winter SWE Across Canada",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Spatial_Mean_Winter_SWE.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 7 — Monthly Winter SWE Distribution (using raw dataset)
# ========================================================================

def plot_monthly_swe_distribution_raw(ds):
    """
    Boxplot comparing monthly SWE (snw variable) for winter months.
    Extracts daily SWE (snw) for Nov, Dec, Jan, Feb across all years/stations.
    """

    snw = ds["snw"].values   # shape: (stations, time)
    time = pd.to_datetime(ds["time"].values)

    month = time.month
    winter_mask = np.isin(month, [11, 12, 1, 2])

    snw_winter = snw[:, winter_mask]
    month_winter = month[winter_mask]

    # Flatten into long format
    df = pd.DataFrame({
        "month": np.repeat(month_winter, snw.shape[0]),
        "snw": snw_winter.T.reshape(-1)
    })

    # Map month numbers → names for readability
    df["month_name"] = df["month"].map({
        11: "November",
        12: "December",
        1: "January",
        2: "February"
    })

    # Drop missing / negative SWE
    df = df[df["snw"] >= 0]

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    df.boxplot(
        column="snw",
        by="month_name",
        grid=True,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", alpha=0.5)
    )

    plt.title("Monthly Winter SWE Distribution")
    plt.suptitle("")  # Remove pandas default
    plt.xlabel("Month")
    plt.ylabel("Daily Snow Water Equivalent (SWE)")
    plt.grid(alpha=0.25)

    out_path = os.path.join(OUTPUT_DIR, "Monthly_SWE_Distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 8 — Monthly Mean SWE Trend Over Time (Nov–Feb)
# ========================================================================

def plot_monthly_mean_swe_trend(ds):
    """
    Computes the mean daily SWE for each winter month (Nov, Dec, Jan, Feb)
    for each year, then plots a separate trend line for each month.
    """

    snw = ds["snw"].values   # shape: (stations, time)
    time = pd.to_datetime(ds["time"].values)

    # Extract month & year
    months = time.month
    years = time.year

    winter_months = [11, 12, 1, 2]
    month_names = {11: "November", 12: "December", 1: "January", 2: "February"}

    # Filter only winter days
    winter_mask = np.isin(months, winter_months)
    snw_w = snw[:, winter_mask]
    months_w = months[winter_mask]
    years_w = years[winter_mask]

    # Build long DataFrame
    df = pd.DataFrame({
        "year": np.repeat(years_w, snw.shape[0]),
        "month": np.repeat(months_w, snw.shape[0]),
        "snw": snw_w.T.reshape(-1)
    })

    df = df[df["snw"] >= 0]  # drop invalid values
    df["month_name"] = df["month"].map(month_names)

    # Compute monthly mean per year
    monthly_trend = df.groupby(["year", "month_name"])["snw"].mean().reset_index()

    # --- Plot ---
    plt.figure(figsize=(14, 8))

    for m in ["November", "December", "January", "February"]:
        sub = monthly_trend[monthly_trend["month_name"] == m]
        if not sub.empty:
            plt.plot(
                sub["year"],
                sub["snw"],
                marker="o",
                linewidth=2,
                alpha=0.9,
                label=m
            )

    plt.title(
        "Monthly Mean SWE Trend Over Time (Nov–Feb)",
        fontsize=17,
        fontweight="bold"
    )
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("Mean Daily SWE (mm)", fontsize=13)
    plt.grid(alpha=0.25)
    plt.legend(title="Month", fontsize=11)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Monthly_Mean_SWE_Trend.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 9 — Ranking Major Cities by Long-Term Winter SWE
# ========================================================================

def plot_city_swe_ranking(df):
    """
    Creates a bar chart ranking major Canadian cities by their
    long-term average winter SWE (snw_mean).
    Uses fuzzy station-name matching to group multiple stations per city.
    """

    # Tag each station with its detected city group
    df["city_group"] = df["station_name"].apply(detect_city)

    # Keep only major cities
    major_df = df[df["city_group"].notna()].copy()

    if major_df.empty:
        print("⚠ No major cities detected for ranking.")
        return

    # Compute long-term mean SWE for each city
    city_swe = (
        major_df.groupby("city_group")["snw_mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 7))
    bars = plt.bar(
        city_swe["city_group"],
        city_swe["snw_mean"],
        color="#4E79A7",
        edgecolor="black",
        linewidth=1.1
    )

    # Add values above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{height:.0f} mm",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.title(
        "Ranking of Major Canadian Cities by Long-Term Winter SWE",
        fontsize=17,
        fontweight="bold"
    )
    plt.ylabel("Mean Winter SWE (mm)")
    plt.xlabel("City")
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "City_SWE_Ranking.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")

# ========================================================================
# EDA BLOCK 10 — Yearly Winter SWE Variability (Std Dev per Year)
# ========================================================================

def plot_yearly_swe_variability(df):
    """
    Computes the standard deviation of winter SWE across all stations
    for each year. Higher std = more variability in snowfall across Canada.
    """

    # df contains snw_total (total winter SWE per station-year)
    yearly_var = (
        df.groupby("year")["snw_total"]
        .std()   # standard deviation across stations
        .reset_index()
        .rename(columns={"snw_total": "std_dev"})
    )

    plt.figure(figsize=(14, 7))

    plt.plot(
        yearly_var["year"],
        yearly_var["std_dev"],
        marker="o",
        linewidth=2.2,
        color="#C44E52",
        label="SWE Variability (Std Dev)"
    )

    plt.title(
        "Yearly Variability in Winter SWE Across Stations",
        fontsize=17,
        fontweight="bold"
    )
    plt.xlabel("Winter Year", fontsize=13)
    plt.ylabel("Standard Deviation of SWE (mm)", fontsize=13)
    plt.grid(alpha=0.25)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "Yearly_Winter_SWE_Variability.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved: {out_path}")


# ========================================================================
# 4. MAIN PIPELINE
# ========================================================================

def run_canswe_station_eda_core_only():
    """
    Core part of the pipeline — no plotting yet.
    """
    print("Loading CanSWE dataset...")
    ds = load_canswe_dataset()

    print("Extracting station metadata...")
    meta, station_dim, time_dim = extract_station_metadata(ds)

    print("Computing winter aggregates...")
    winter_summary = compute_winter_aggregates_fast(ds, station_dim, time_dim)

    print("Merging summary with metadata...")
    merged = merge_summary_with_metadata(winter_summary, meta)

    plot_total_swe_by_year(merged)
    plot_station_correlation_heatmap(merged)
    plot_swe_scatter_by_station(merged)
    plot_major_city_swe_scatter(merged)
    plot_yearly_swe_anomalies(merged)
    plot_station_trends_major_cities(merged)
    plot_spatial_mean_swe(merged)
    plot_monthly_swe_distribution_raw(ds)
    plot_monthly_mean_swe_trend(ds)
    plot_city_swe_ranking(merged)
    plot_yearly_swe_variability(merged)

    print("\n======================================")
    print("CORE DATA PROCESSING COMPLETE")
    print(f"Outputs saved in: {OUTPUT_DIR}/")
    print("  - station_metadata.csv")
    print("  - station_winter_summary_optimized.csv")
    print("  - station_winter_summary_with_meta.csv")
    print("======================================\n")

    return merged


if __name__ == "__main__":
    run_canswe_station_eda_core_only()
