# Insurance Cross-Selling Prediction 🎯

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)

## Project Overview 📋

A machine learning solution for predicting customer response to vehicle insurance cross-selling campaigns. The model uses an ensemble approach combining CatBoost, LightGBM, and XGBoost to achieve superior predictive performance.

### Problem Statement

In the insurance industry, optimizing cross-selling strategies is crucial for business growth. This project aims to predict which health insurance customers are most likely to be interested in an additional vehicle insurance product, enabling targeted marketing campaigns and improved conversion rates.

### Key Objectives
- Predict customer likelihood to purchase vehicle insurance
- Identify key factors influencing purchase decisions
- Enable targeted marketing through accurate risk scoring
- Optimize resource allocation for marketing campaigns

## Data Analysis 📊

### Dataset Statistics
- **Records**: 11 million entries
- **Features**: 12 initial features
- **Generated Features**: 4 interaction features
- **Target Variable**: Binary (Response: 1 = Interested, 0 = Not Interested)

### Key Features
1. **Academic Factors**:
   - Age
   - Driving License
   - Vehicle Age
   - Vehicle Damage

2. **Insurance History**:
   - Previously Insured
   - Policy Sales Channel

3. **Financial Factors**:
   - Annual Premium
   - Vintage (Customer Tenure)

### Feature Importance (Normalized Scores)
```
Feature                             Score
Previously_Insured                  1.000
Annual_Premium                      0.876
Vehicle_Damage                      0.754
Age                                0.721
Vehicle_Age                        0.687
```

### Key Insights 📈

#### Age Impact on Conversion
```
Age Group    Conversion Rate    Population Share
15-20        28.2%             42.3%
21-25        24.8%             31.7%
26-30        22.1%             12.1%
31-35        19.4%             7.4%
36-40        17.2%             4.2%
41-50        15.8%             2.3%
```

#### Vehicle Age and Response Rate
```
Vehicle Age    Response Rate    Sample Size
< 1 Year       18.4%           42.3%
1-2 Years      25.7%           31.7%
> 2 Years      35.2%           26.0%
```

#### Prior Insurance Status Impact
- Customers without previous vehicle insurance showed 3.2x higher conversion rates
- 72.3% of conversions came from previously uninsured customers

## Methodology 🔬

### Data Preprocessing
1. **Categorical Encoding**
   - Gender: Male = 1, Female = 0
   - Vehicle Damage: Yes = 1, No = 0
   - Vehicle Age: Ordinal encoding (0, 1, 2)

2. **Feature Engineering**
   - Created interaction features with Previously_Insured
   - Standardized numerical features
   - Applied rare label encoding for Region_Code

3. **Data Optimization**
   - Reduced memory usage through downcasting
   - Optimized categorical encodings
   - Streamlined numerical precision

### Model Architecture

#### Ensemble Components
1. **CatBoost**
   ```python
   params = {
       'learning_rate': 0.075,
       'depth': 9,
       'l2_leaf_reg': 0.5,
       'max_leaves': 512
   }
   ```

2. **LightGBM**
   ```python
   params = {
       'learning_rate': 0.050,
       'max_depth': 10,
       'num_leaves': 31,
       'min_child_samples': 100
   }
   ```

3. **XGBoost**
   ```python
   params = {
       'eta': 0.05,
       'max_depth': 16,
       'min_child_weight': 5,
       'subsample': 0.839
   }
   ```

### Model Performance 📊

#### Individual Model Metrics
```
Model       ROC-AUC     Std Dev
CatBoost    0.8967      ±0.0024
LightGBM    0.8952      ±0.0021
XGBoost     0.8944      ±0.0019
```

#### Ensemble Performance
- **Final ROC-AUC**: 0.8970
- **Improvement**: +0.0003 over best single model
- **Cross-validation**: 5-fold stratified CV

#### Confusion Matrix (Normalized)
```
Predicted →    Positive    Negative
Positive       0.842       0.158
Negative       0.179       0.821
```
