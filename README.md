# Canadian Winter Snowfall Analysis

This project processes and analyzes Canadian winter snowfall data using the Canadian Historical Daily Snow Depth Dataset (.dly format). It extracts winter-specific data (November–February), performs detailed statistical analysis, and generates visualizations and labeled CSV summaries.
## Project Structure

## Overview of Features

### 1. Winter-Only Parsing
The script extracts valid daily snowfall measurements for:
- November (11)  
- December (12)  
- January (1)  
- February (2)
### 2. Statistical Analysis
The script prints:
- Dataset summary  
- Monthly breakdown  
- Yearly trends  
- Top stations  

### 3. Visualizations
- Winter_Analysis_Dashboard.png  
- Monthly_Breakdown.png  

### 4. CSV Exports
- Winter_Data_Full_Labeled.csv  
- Winter_Monthly_Summary.csv  
- Winter_Yearly_Summary.csv  
- Winter_Station_Summary.csv  

## How to Run

```
python Snow_analysis.py
```

## Dependencies

```
pandas
numpy
matplotlib
scipy
```

Install with:

```
pip install pandas numpy matplotlib scipy
```

## Dataset Requirements

Must be a NOAA-format `.dly` file:
`Canadian-Historical-Snow-Depth-Dataset-2019-Update.dly`

## Main Functions
- parse_winter_data()  
- create_date_column()  
- analyze_winter_detailed()  
- create_enhanced_dashboard()  
- export_detailed_csv()

