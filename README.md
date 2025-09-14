# Marketing Mix Modeling (MMM) Pipeline with XGBoost

A comprehensive two-stage machine learning pipeline for Marketing Mix Modeling that predicts revenue based on various marketing channels and business metrics using XGBoost with enhanced visualization capabilities.

## Overview

This pipeline implements a two-stage approach:
1. **Stage 1**: Predicts Google Spend based on other social media spending (Facebook, TikTok, Snapchat)
2. **Stage 2**: Predicts Revenue using business metrics and predicted Google Spend

The model uses XGBoost regressors with time series cross-validation and provides comprehensive visualizations for model evaluation.

## Features

- **Two-Stage Modeling**: Hierarchical approach for better feature engineering
- **Advanced Preprocessing**: Log transformation for spend variables with proper handling of zero values
- **Time Series Cross-Validation**: Proper validation technique for sequential data
- **Comprehensive Metrics**: R², RMSE, MAE, and MAPE calculations
- **Rich Visualizations**: 
  - Actual vs Predicted scatter plots
  - Residual distribution plots
  - Time series plots
  - MAPE heatmaps
- **Robust Error Handling**: Built-in data validation and preprocessing

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Data Format

Your CSV file should contain the following columns:

### Required Columns:
- `facebook_spend`: Facebook advertising spend
- `tiktok_spend`: TikTok advertising spend  
- `snapchat_spend`: Snapchat advertising spend
- `google_spend`: Google advertising spend (target for Stage 1)
- `email_sends`: Number of emails sent
- `sms_sends`: Number of SMS messages sent
- `avg_price`: Average product price
- `followers`: Social media followers count
- `promotions`: Number of promotions run
- `revenue`: Total revenue (final target)

### Optional Column:
- `date`: Date column for time series visualization (format: YYYY-MM-DD)

## Usage

### Basic Usage

```python
import pandas as pd
from mmm_pipeline import MMMPipelineXGB

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize and run pipeline
mmm = MMMPipelineXGB()
mmm.run_pipeline(df)
```

### Custom Parameters

```python
# Initialize with custom parameters
mmm = MMMPipelineXGB()

# Modify XGBoost parameters
mmm.stage1_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,  # Lower learning rate
    'max_depth': 4,         # Deeper trees
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': 42
}

# Run pipeline
mmm.run_pipeline(df)
```

### Cross-Validation Only

```python
# Run only cross-validation
cv_results = mmm.time_series_cv(df, n_splits=5)
print(cv_results)
```

## Output

The pipeline provides:

### Console Output:
- Stage 1 performance metrics (Google Spend prediction)
- Stage 2 performance metrics (Revenue prediction)
- Cross-validation results with mean and standard deviation

### Visualizations:
1. **Actual vs Predicted Plots**: Scatter plots showing model fit quality
2. **Residual Plots**: Distribution of prediction errors
3. **Time Series Plot**: Revenue trends over time (if date column provided)
4. **MAPE Heatmap**: Mean Absolute Percentage Error visualization

### Metrics Explained:
- **R² (R-squared)**: Proportion of variance explained (higher is better, max 1.0)
- **RMSE**: Root Mean Square Error in original units (lower is better)
- **MAE**: Mean Absolute Error in original units (lower is better)
- **MAPE**: Mean Absolute Percentage Error as percentage (lower is better)

## Model Architecture

### Stage 1: Google Spend Prediction
```
Input Features: log(Facebook Spend), log(TikTok Spend), log(Snapchat Spend)
↓
StandardScaler
↓
XGBoost Regressor
↓
Output: Predicted Google Spend
```

### Stage 2: Revenue Prediction
```
Input Features: Email Sends, SMS Sends, Avg Price, Followers, Promotions, Predicted Google Spend
↓
StandardScaler
↓
XGBoost Regressor
↓
Output: Predicted Revenue
```

## Configuration

### XGBoost Parameters:
- `objective`: 'reg:squarederror' for regression
- `learning_rate`: 0.1 (default)
- `max_depth`: 3 (default)
- `subsample`: 0.8 for regularization
- `colsample_bytree`: 0.8 for regularization
- `num_boost_round`: 200 with early stopping

### Cross-Validation:
- Uses `TimeSeriesSplit` to maintain temporal order
- Default 5 splits (adjustable)
- Early stopping after 10 rounds without improvement

## Sample Data Structure

```csv
date,facebook_spend,tiktok_spend,snapchat_spend,google_spend,email_sends,sms_sends,avg_price,followers,promotions,revenue
2023-01-01,1000,500,300,800,5000,1000,25.99,10000,2,15000
2023-01-08,1200,600,350,900,5200,1100,26.50,10200,1,16500
2023-01-15,1100,550,320,850,4800,950,25.75,10150,3,14800
...
```

## Troubleshooting

### Common Issues:

1. **Missing Columns**: Ensure all required columns are present in your CSV
2. **Date Format**: If using date column, ensure format is readable by pandas
3. **Missing Values**: The pipeline handles missing values by filling with 0, but clean data is recommended
4. **Insufficient Data**: Time series CV requires sufficient data points (recommended: >100 rows)

### Error Messages:

- `KeyError`: Check column names match exactly
- `ValueError`: Verify data types are numeric for all feature columns
- `Warning: Early stopping`: Model converged early (usually good)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

## Changelog

### v1.0.0
- Initial release with two-stage XGBoost pipeline
- Time series cross-validation
- Comprehensive visualization suite
- Log transformation for spend variables
- Enhanced error handling and metrics
