# Canadian Winter Snowfall Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

WINTER_MONTHS = [11, 12, 1, 2]
OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================================
# PARSE DLY FILE
# ============================================================================

def parse_winter_data(filepath):    
    print("Parsing .dly file for winter months (November, December, January, February)")
    
    data = []
    lines_processed = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                continue
            
            fields = line.split()
            if len(fields) < 5:
                continue
            
            try:
                station_id = fields[0]
                yyyymm = int(fields[1])
                year = yyyymm // 100
                month = yyyymm % 100
                
                if month not in WINTER_MONTHS:
                    continue
                
                daily_fields = fields[3:]
                
                day = 1
                for i in range(0, len(daily_fields), 2):
                    if day > 31:
                        break
                    
                    try:
                        value = int(daily_fields[i])
                        if value >= 0:
                            snow_depth_cm = value / 10.0
                            data.append({
                                'station_id': station_id,
                                'year': year,
                                'month': month,
                                'day': day,
                                'snow_depth_cm': snow_depth_cm
                            })
                    except (ValueError, IndexError):
                        pass
                    
                    day += 1
                
                lines_processed += 1
                if lines_processed % 5000 == 0:
                    print(f"  Processed {lines_processed} lines, {len(data):,} winter data points...")
            
            except Exception as e:
                continue
    
    print(f"  ✓ Total: {len(data):,} points\n")
    
    if len(data) == 0:
        print("ERROR: No data found!")
        return None
    
    df = pd.DataFrame(data)
    return df


def create_date_column(df):
    """Create proper date column"""
    
    df['date_year'] = df['year'].copy()
    df.loc[df['month'].isin([1, 2]), 'date_year'] = df.loc[df['month'].isin([1, 2]), 'year'] - 1
    
    df['date'] = pd.to_datetime(
        df[['date_year', 'month', 'day']].rename(columns={'date_year': 'year'}),
        errors='coerce'
    )
    
    df = df.dropna(subset=['date'])
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_winter_detailed(df):
    """Snowfall analysis"""
    
    print("="*80)
    print("CANADIAN WINTER SNOWFALL ANALYSIS - STATISTICS")
    print("="*80)
    print()
    
    print("DATASET SUMMARY:")
    print(f"  Total Records: {len(df):,} observations")
    print(f"  Unique Stations: {df['station_id'].nunique()} weather stations")
    print(f"  Time Period: {df['year'].min()}-{df['year'].max()} ({df['year'].max() - df['year'].min() + 1} years)")
    print()
    
    print("OVERALL WINTER SNOWFALL (All Months Combined):")
    snow = df['snow_depth_cm']
    print(f"  Mean:           {snow.mean():7.2f} cm")
    print(f"  Median:         {snow.median():7.2f} cm")
    print(f"  Std Deviation:  {snow.std():7.2f} cm")
    print(f"  Minimum:        {snow.min():7.2f} cm")
    print(f"  Maximum:        {snow.max():7.2f} cm")
    print(f"  25th Percentile:{snow.quantile(0.25):7.2f} cm")
    print(f"  75th Percentile:{snow.quantile(0.75):7.2f} cm")
    print()
    
    print("MONTHLY BREAKDOWN:")
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February'}
    for month in [11, 12, 1, 2]:
        month_data = df[df['month'] == month]['snow_depth_cm']
        print(f"  {month_names[month]:10s}: Mean={month_data.mean():6.2f}cm, Median={month_data.median():6.2f}cm, Std={month_data.std():6.2f}cm")
    print()
    
    print("TOP 10 STATIONS (Highest Average Snowfall):")
    top_10 = df.groupby('station_id')['snow_depth_cm'].agg(['count', 'mean', 'std']).nlargest(10, 'mean')
    for i, (stn, row) in enumerate(top_10.iterrows(), 1):
        print(f"  {i:2d}. Station {stn:10s} | Mean: {row['mean']:6.2f}cm | Std: {row['std']:5.2f}cm")
    print()


# ============================================================================
# VISUALIZATIONS - NO SAMPLE SIZE LABELS
# ============================================================================

def create_enhanced_dashboard(df, output_dir):
    """Create detailed dashboard WITHOUT n=XXX labels"""
    
    print("Creating visualizations...")
    
    # ========== FIGURE 1: MAIN WINTER ANALYSIS ==========
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Canadian Winter Snowfall Analysis Dashboard\nNovember - February (All Years & Stations)', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # 1. Distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df['snow_depth_cm'], bins=70, color='steelblue', alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Snow Depth (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Daily Snow Depth Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box (NO n=XXX)
    stats_text = f'Mean: {df["snow_depth_cm"].mean():.2f} cm\nMedian: {df["snow_depth_cm"].median():.2f} cm\nStd: {df["snow_depth_cm"].std():.2f} cm\nMax: {df["snow_depth_cm"].max():.2f} cm'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.8))
    
    # 2. SEASONAL VARIATION by month - NO n= labels
    ax = fig.add_subplot(gs[0, 1])
    winter_data = [df[df['month'] == m]['snow_depth_cm'].values for m in [11, 12, 1, 2]]
    month_labels = ['November', 'December', 'January', 'February']
    
    bp = ax.boxplot(winter_data, labels=month_labels, patch_artist=True, widths=0.6)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_linewidth(2)
    
    ax.set_ylabel('Snow Depth (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Seasonal Variation: Snowfall by Month', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add monthly statistics annotations (NO n=XXX)
    for i, month in enumerate([11, 12, 1, 2]):
        month_data = df[df['month'] == month]['snow_depth_cm']
        mean_val = month_data.mean()
        ax.text(i+1, mean_val, f'μ={mean_val:.1f}', ha='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Q-Q Plot
    ax = fig.add_subplot(gs[0, 2])
    stats.probplot(df['snow_depth_cm'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Normality Check', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4. LONG-TERM TREND by Year
    ax = fig.add_subplot(gs[1, :2])
    yearly = df.groupby('year')['snow_depth_cm'].agg(['count', 'mean', 'std']).sort_index()
    
    ax.plot(yearly.index, yearly['mean'], marker='o', linewidth=3, markersize=8, 
            color='darkred', label='Average Snow Depth', zorder=3)
    ax.fill_between(yearly.index, 
                    yearly['mean'] - yearly['std'], 
                    yearly['mean'] + yearly['std'],
                    alpha=0.3, color='red', label='±1 Std Dev')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Snow Depth (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Long-term Winter Snowfall Trend', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=11, loc='best')
    
    # Add year labels every 10 years
    for year in yearly.index[::max(1, len(yearly.index)//10)]:
        ax.axvline(year, color='gray', linestyle=':', alpha=0.3)
    
    # 5. CDF
    ax = fig.add_subplot(gs[1, 2])
    sorted_vals = np.sort(df['snow_depth_cm'])
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, linewidth=2.5, color='darkblue')
    ax.fill_between(sorted_vals, cdf, alpha=0.3, color='steelblue')
    ax.set_xlabel('Snow Depth (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    # Add reference line
    median_val = df['snow_depth_cm'].median()
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median={median_val:.2f}cm')
    ax.legend(fontsize=10)
    
    # 6. DETAILED METRICS TABLE
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    snow = df['snow_depth_cm']
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        ['Total Sample Size', f'{len(df):,} days', 'Total observations across all stations and years'],
        ['Unique Weather Stations', f'{df["station_id"].nunique()}', 'Number of weather monitoring stations'],
        ['Years Covered', f'{df["year"].min()}-{df["year"].max()}', 'Time span of data'],
        ['Mean Snow Depth', f'{snow.mean():.2f} cm', 'Average daily snow depth'],
        ['Median Snow Depth', f'{snow.median():.2f} cm', 'Middle value (50% above, 50% below)'],
        ['Standard Deviation', f'{snow.std():.2f} cm', 'Measure of variability'],
        ['Minimum Snow Depth', f'{snow.min():.2f} cm', 'Lowest recorded daily depth'],
        ['Maximum Snow Depth', f'{snow.max():.2f} cm', 'Highest recorded daily depth'],
        ['25th Percentile', f'{snow.quantile(0.25):.2f} cm', '25% of winter days had ≤ this depth'],
        ['75th Percentile', f'{snow.quantile(0.75):.2f} cm', '75% of winter days had ≤ this depth'],
        ['Interquartile Range', f'{snow.quantile(0.75)-snow.quantile(0.25):.2f} cm', 'Difference between 75th and 25th percentiles'],
    ]
    
    table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                    colWidths=[0.25, 0.15, 0.60])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Style table
    for i in range(3):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Winter_Analysis_Dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: Winter_Analysis_Dashboard.png")
    plt.close()
    
    # ========== FIGURE 2: MONTHLY DETAIL ==========
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Monthly Winter Snowfall Breakdown', fontsize=18, fontweight='bold', y=0.995)
    
    month_info = [
        (11, 'November', '#FF6B6B'),
        (12, 'December', '#4ECDC4'),
        (1, 'January', '#45B7D1'),
        (2, 'February', '#96CEB4')
    ]
    
    for idx, (month_num, month_name, color) in enumerate(month_info):
        ax = axes[idx // 2, idx % 2]
        month_data = df[df['month'] == month_num]['snow_depth_cm']
        
        ax.hist(month_data, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Snow Depth (cm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{month_name} Snow Depth Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics box (NO n=XXX)
        stats_text = (f'Mean: {month_data.mean():.2f} cm\n'
                     f'Median: {month_data.median():.2f} cm\n'
                     f'Std: {month_data.std():.2f} cm\n'
                     f'Min: {month_data.min():.2f} cm\n'
                     f'Max: {month_data.max():.2f} cm')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'Monthly_Breakdown.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: Monthly_Breakdown.png")
    plt.close()


# ============================================================================
# EXPORT CSV
# ============================================================================

def export_detailed_csv(df, output_dir):
    """Export CSV files with proper date handling"""
    
    print("\nExporting CSV files...")
    
    # 1. Full dataset - use already validated date column
    export_df = df[['station_id', 'year', 'month', 'day', 'snow_depth_cm', 'date']].copy()
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February'}
    export_df['month_name'] = export_df['month'].map(month_names)
    
    # Reorder columns
    export_df = export_df[['date', 'station_id', 'year', 'month', 'month_name', 'day', 'snow_depth_cm']]
    
    export_file = os.path.join(output_dir, 'Winter_Data_Full.csv')
    export_df.to_csv(export_file, index=False)
    print(f"  ✓ Full dataset: {export_file}")
    
    # 2. Monthly summary
    monthly_summary = df.groupby('month').agg({
        'snow_depth_cm': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(2)
    monthly_summary.columns = ['Count', 'Mean_cm', 'Median_cm', 'Std_cm', 'Min_cm', 'Max_cm']
    monthly_summary.index = ['November', 'December', 'January', 'February']
    monthly_summary = monthly_summary.reset_index()
    
    export_file = os.path.join(output_dir, 'Winter_Monthly_Summary.csv')
    monthly_summary.to_csv(export_file, index=False)
    print(f"  ✓ Monthly summary: {export_file}")
    
    # 3. Yearly summary
    yearly_summary = df.groupby('year').agg({
        'snow_depth_cm': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(2)
    yearly_summary.columns = ['Count', 'Mean_cm', 'Median_cm', 'Std_cm', 'Min_cm', 'Max_cm']
    yearly_summary = yearly_summary.reset_index()
    yearly_summary['year'] = yearly_summary['year'].astype(int)
    
    export_file = os.path.join(output_dir, 'Winter_Yearly_Summary.csv')
    yearly_summary.to_csv(export_file, index=False)
    print(f"  ✓ Yearly summary: {export_file}")
    
    # 4. Station summary
    station_summary = df.groupby('station_id').agg({
        'snow_depth_cm': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(2)
    station_summary.columns = ['Count', 'Mean_cm', 'Median_cm', 'Std_cm', 'Min_cm', 'Max_cm']
    station_summary = station_summary.sort_values('Mean_cm', ascending=False).reset_index()
    
    export_file = os.path.join(output_dir, 'Winter_Station_Summary.csv')
    station_summary.to_csv(export_file, index=False)
    print(f"  ✓ Station summary: {export_file}")


# ============================================================================
# MAIN FILE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("CANADIAN WINTER SNOWFALL ANALYSIS")
    print("="*80 + "\n")
    
    filepath = "Canadian-Historical-Snow-Depth-Dataset-2019-Update.dly"
    
    df = parse_winter_data(filepath)
    
    if df is not None and len(df) > 0:
        
        df = create_date_column(df)
        
        if len(df) > 0:
            
            analyze_winter_detailed(df)
            create_enhanced_dashboard(df, OUTPUT_DIR)
            export_detailed_csv(df, OUTPUT_DIR)
            
            print("\n" + "="*80)
            print(f"✓ ANALYSIS COMPLETE!")
            print(f"\nOutput Files:")
            print(f" Visualizations:")
            print(f"     • Winter_Analysis_Dashboard.png")
            print(f"     • Monthly_Breakdown.png")
            print(f"\n Data Files (CSV):")
            print(f"     • Winter_Data_Full.csv")
            print(f"     • Winter_Monthly_Summary.csv")
            print(f"     • Winter_Yearly_Summary.csv")
            print(f"     • Winter_Station_Summary.csv")
            print(f"\n  Location: {OUTPUT_DIR}")
            print("="*80 + "\n")
        
        else:
            print("No valid data after date creation")
    
    else:
        print("Failed to parse winter data")