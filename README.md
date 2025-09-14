# 📊 Marketing Mix Modeling (MMM) Pipeline with XGBoost

A comprehensive two-stage machine learning pipeline for Marketing Mix Modeling that predicts revenue based on various marketing channels and business metrics using XGBoost with enhanced visualization capabilities.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-v1.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This pipeline implements a sophisticated two-stage approach for Marketing Mix Modeling:

1. **Stage 1**: Predicts Google Spend based on social media advertising spend (Facebook, TikTok, Snapchat)
2. **Stage 2**: Predicts Revenue using business metrics and the predicted Google Spend from Stage 1

The model achieves excellent performance with **R² > 0.97** for Google Spend prediction and **R² > 0.99** for Revenue prediction.

## ✨ Key Features

- 🚀 **Two-Stage XGBoost Pipeline** with hierarchical feature engineering
- 📈 **Advanced Preprocessing** with log transformations and proper zero handling
- ⏰ **Time Series Cross-Validation** for robust model evaluation
- 📊 **Comprehensive Visualizations** including actual vs predicted plots, residual analysis
- 📋 **Multiple Performance Metrics** (R², RMSE, MAE, MAPE)
- 🛡️ **Robust Error Handling** and data validation

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mmm-pipeline-xgboost.git
   cd mmm-pipeline-xgboost
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
   
   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your CSV file in the project directory
   - Ensure it follows the required format (see Data Format section)

4. **Run the pipeline**
   ```bash
   python mmm_pipeline.py
   ```

### 🔧 Alternative Installation with Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv mmm_env

# Activate virtual environment
# On Windows:
mmm_env\Scripts\activate
# On macOS/Linux:
source mmm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python mmm_pipeline.py
```

## 📋 Data Format Requirements

Your CSV file must contain the following columns:

### 🎯 Required Columns:
| Column | Description | Example |
|--------|-------------|---------|
| `facebook_spend` | Facebook advertising spend ($) | 1000.50 |
| `tiktok_spend` | TikTok advertising spend ($) | 500.25 |
| `snapchat_spend` | Snapchat advertising spend ($) | 300.75 |
| `google_spend` | Google advertising spend ($) | 800.00 |
| `email_sends` | Number of emails sent | 5000 |
| `sms_sends` | Number of SMS messages sent | 1000 |
| `avg_price` | Average product price ($) | 25.99 |
| `followers` | Social media followers count | 10000 |
| `promotions` | Number of promotions run | 2 |
| `revenue` | Total revenue ($) | 15000.00 |

### 📅 Optional Column:
| Column | Description | Format |
|--------|-------------|--------|
| `date` | Date for time series visualization | YYYY-MM-DD |

### 📄 Sample Data Structure:
```csv
date,facebook_spend,tiktok_spend,snapchat_spend,google_spend,email_sends,sms_sends,avg_price,followers,promotions,revenue
2023-01-01,1000,500,300,800,5000,1000,25.99,10000,2,15000
2023-01-08,1200,600,350,900,5200,1100,26.50,10200,1,16500
2023-01-15,1100,550,320,850,4800,950,25.75,10150,3,14800
```

## 💻 Usage Examples

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
# Initialize with custom XGBoost parameters
mmm = MMMPipelineXGB()

# Modify Stage 1 parameters
mmm.stage1_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': 42
}

# Run with custom parameters
mmm.run_pipeline(df)
```

### Cross-Validation Only
```python
# Run only time series cross-validation
cv_results = mmm.time_series_cv(df, n_splits=5)
for metric, value in cv_results.items():
    print(f"{metric}: {value:.3f}")
```

## 📊 Expected Output

When you run the pipeline, you'll see:

### 🖥️ Console Output:
```
Stage 1 - Google Spend | R²: 0.972, RMSE: 180, MAE: 128, MAPE: 1.5%
Stage 2 - Revenue | R²: 0.994, RMSE: 1839, MAE: 817, MAPE: 1.2%

Cross-Validation Results:
R2_mean: 0.797
R2_std: 0.178
RMSE_mean: 4086.097
RMSE_std: 2247.309
MAE_mean: 1752.219
MAE_std: 1752.219
MAPE_mean: 5.548
MAPE_std: 2.438
```

### 📈 Visualizations Generated:

1. **Stage 1 - Actual vs Predicted (Google Spend)**
   - Scatter plot showing model fit quality
   - R² = 0.972 indicates excellent prediction accuracy

2. **Stage 1 - Residual Distribution**
   - Histogram showing prediction error distribution
   - Well-centered around zero indicates good model performance

3. **Stage 2 - Actual vs Predicted (Revenue)**
   - Revenue prediction scatter plot
   - R² = 0.994 shows outstanding model performance

4. **Stage 2 - Residual Distribution**
   - Revenue prediction error analysis
   - Normal distribution of residuals indicates robust model

## 🏗️ Model Architecture

### Stage 1: Google Spend Prediction
```
📊 Input Features: log(Facebook), log(TikTok), log(Snapchat)
    ⬇️
🔧 StandardScaler Normalization
    ⬇️
🚀 XGBoost Regressor (200 rounds)
    ⬇️
📈 Output: Predicted Google Spend
```

### Stage 2: Revenue Prediction
```
📊 Input Features: Email Sends, SMS Sends, Avg Price, Followers, Promotions, Predicted Google Spend
    ⬇️
🔧 StandardScaler Normalization
    ⬇️
🚀 XGBoost Regressor (200 rounds)
    ⬇️
💰 Output: Predicted Revenue
```

## ⚙️ Configuration

### Default XGBoost Parameters:
```python
{
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
```

### Performance Metrics Explained:
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Square Error in original units (lower is better)
- **MAE**: Mean Absolute Error in original units (lower is better)
- **MAPE**: Mean Absolute Percentage Error (%, lower is better)

## 📁 Project Structure

```
mmm-pipeline-xgboost/
│
├── mmm_pipeline.py          # Main pipeline code
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── sample_data.csv         # Sample dataset
├── LICENSE                 # MIT License
│
└── outputs/                # Generated visualizations
    ├── stage1_actual_vs_predicted.png
    ├── stage1_residuals.png
    ├── stage2_actual_vs_predicted.png
    └── stage2_residuals.png
```

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### ❌ `FileNotFoundError: 'MMM Weekly.csv' not found`
**Solution:** Update the CSV filename in the code:
```python
csv_file = 'your_actual_filename.csv'
```

#### ❌ `KeyError: Column not found`
**Solution:** Ensure your CSV contains all required columns with exact names

#### ❌ `ImportError: Module not found`
**Solution:** Install missing packages:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

#### ❌ Poor model performance (Low R²)
**Solutions:**
- Check for data quality issues
- Ensure sufficient data points (>100 rows recommended)
- Verify column data types are numeric
- Consider feature scaling or additional preprocessing

#### ❌ Memory issues with large datasets
**Solutions:**
- Reduce `num_boost_round` parameter
- Use data sampling for initial testing
- Increase system RAM or use cloud computing

## 🔍 Advanced Usage

### Custom Preprocessing
```python
class CustomMMMPipeline(MMMPipelineXGB):
    def preprocess(self, df):
        df = super().preprocess(df)
        # Add custom preprocessing steps
        df['total_social_spend'] = df['facebook_spend'] + df['tiktok_spend'] + df['snapchat_spend']
        return df
```

### Model Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Implement custom tuning (advanced users)
```

## 📈 Performance Benchmarks

Based on typical marketing data:
- **Stage 1 Performance**: R² > 0.95, MAPE < 2%
- **Stage 2 Performance**: R² > 0.99, MAPE < 1.5%
- **Cross-validation Stability**: Standard deviation < 0.2 for R²

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup:
```bash
# Clone your fork
git clone https://github.com/yourusername/mmm-pipeline-xgboost.git

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 black

# Run tests
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- 🐛 **Bug Reports**: [Open an Issue](https://github.com/yourusername/mmm-pipeline-xgboost/issues)
- 💡 **Feature Requests**: [Start a Discussion](https://github.com/yourusername/mmm-pipeline-xgboost/discussions)
- 📧 **Email**: your.email@example.com

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting framework
- Scikit-learn community for preprocessing and validation tools
- Marketing analytics community for MMM methodology insights

## 📚 Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Marketing Mix Modeling Guide](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

**⭐ If this project helps you, please give it a star! ⭐**

*Made with ❤️ for the marketing analytics community*
