# Binary Classification of Insurance Cross Selling

#### Playground Series - Season 4, Episode 7

## Model Blend with CatBoost, LightGBM, and XGBoost

**Author:** Paulo Araya-Santiago

Welcome to my comprehensive notebook for the Kaggle Playground Series: Binary Classification of Insurance Cross Selling. This notebook demonstrates my end-to-end workflow, from data exploration and preprocessing to feature engineering and model tuning. The goal is to build a robust model that accurately predicts insurance responses. This version of the notebook is intended for local execution and display on GitHub.

## Preliminary Tests and Findings

This notebook is the culmination of at least 50 different iterations throughout the competition. As of this version, I have not yet achieved my ideal score, but I am sharing my favorite parts of the process. Initial tests involved running the dataset through different models with minimal transformations to establish baseline performance.

### Key Insights:
- **Feature Engineering:** Iteratively improved feature selection by removing correlated features and adding interaction terms.
- **Model Performance:** Tested various models, including XGBoost, LightGBM, and CatBoost, before finalizing a blend of all three.
- **Sampling Strategy:** Used a stratified sampling approach to handle class imbalances effectively.
- **Hyperparameter Tuning:** Employed Bayesian optimization with Optuna to fine-tune the best-performing models.
- **Evaluation Metrics:** Focused on ROC AUC score as the primary metric while monitoring precision-recall trade-offs.
- **Generalization Strategies:** Prevented overfitting by applying K-fold cross-validation and feature reduction methods.

## Technical Aspects

### Data Preprocessing
- Handled missing values through imputation strategies.
- Applied **one-hot encoding** for categorical variables and **MinMax scaling** for numerical features.
- Removed outliers based on feature distributions to improve model stability.

### Model Training and Tuning
- Implemented **XGBoost, LightGBM, and CatBoost**, comparing their performance through validation.
- Utilized **Optuna for hyperparameter optimization**, leveraging Bayesian search for efficient tuning.
- Applied **Stratified K-Fold Cross-Validation (CV)** to ensure model robustness.

### Ensemble Strategy
- Combined predictions from multiple models using a weighted blending approach.
- Optimized blend weights based on validation performance to maximize ROC AUC.

### Computational Considerations
- Managed large datasets using **Colab & Local Jupyter Notebooks**.
- Implemented memory-efficient processing techniques.
- Regularly cleared redundant variables to optimize RAM usage.

## Next Steps
- Further fine-tune model blending to push the score higher.
- Explore deep learning models like TabNet for comparison.
- Experiment with adversarial validation to detect distribution shifts.

## Acknowledgments
Special thanks to the Kaggle community for valuable discussions and insights throughout the competition.


