# 📊 Marketing Mix Modeling (MMM) Pipeline with XGBoost

This repository provides a **two-stage Marketing Mix Modeling (MMM) pipeline** implemented in Python, using **XGBoost** for regression and enhanced visualization for diagnostics.  
The pipeline is designed to model relationships between marketing spend and business outcomes (e.g., revenue), using **time-series cross-validation** to evaluate performance.

---

## 🚀 Features
- **Stage 1:** Predict Google spend using other media channel spends (`facebook`, `tiktok`, `snapchat`).  
- **Stage 2:** Predict **Revenue** using marketing features, promotions, and predicted Google spend.  
- **Models:** Gradient Boosted Trees via [XGBoost](https://xgboost.ai).  
- **Preprocessing:**  
  - Log transformation of ad spends.  
  - Standard scaling of features.  
  - Missing value handling (set to 0 by default).  
- **Metrics:** R², RMSE, MAE, and MAPE.  
- **Validation:** Time-series cross-validation (`TimeSeriesSplit`).  
- **Visualization:**  
  - Actual vs Predicted scatterplots  
  - Residual distribution plots  
  - Time-series line plots  
  - MAPE heatmaps  

---
