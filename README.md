# Housing Price Prediction Project

## Overview
This project focuses on predicting housing prices using machine learning models, specifically Random Forest and Decision Tree regressors. The dataset used is sourced from 'housing.csv', containing various features such as location, housing age, and median income.

## Data Preprocessing
- Handling null values: Filling 'total_bedrooms' null values with zero.
- Feature selection: Selected features include 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'median_income', and 'ocean_proximity'.
- Dummy variables: Used OneHotEncoder for categorical variable 'ocean_proximity'.
- Data scaling: Applied StandardScaler to standardize the features.

## Model Training and Evaluation
### Random Forest
- Hyperparameter: `n_estimators=100`.
- Metrics:
  - R-squared (R2) Score: {r2}
  - Mean Absolute Error (MAE): {mae}
  - Root Mean Squared Error (RMSE): {rmse}

### Decision Tree
- Hyperparameters: `max_depth=10`, `min_samples_split=50`.
- Metrics:
  - R-squared (R2) Score: {r2}
  - Mean Absolute Error (MAE): {mae}
  - Root Mean Squared Error (RMSE): {rmse}

## Conclusion
- Both Random Forest and Decision Tree models were trained and evaluated for housing price prediction.
- Random Forest outperformed the Decision Tree in terms of R2 score, MAE, and RMSE.
- These models can be further tuned, and additional features can be explored to improve performance.

Feel free to reach out if you have any questions or suggestions for improvement!
