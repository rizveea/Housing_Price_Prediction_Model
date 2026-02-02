# Predictive Analytics: Housing Price Model

## Business Context
Predictive modeling enables data-driven decisions across industries—from demand forecasting in supply chains to price optimization in procurement. This project builds **machine learning models to predict asset values**, demonstrating the full ML pipeline applicable to spend analytics, cost estimation, and market intelligence.

## Project Highlights

### Problem Addressed
Developed a **predictive model for Austin, TX housing prices** comparing multiple algorithms to identify the optimal approach for price estimation tasks.

### Key Outcomes
- **Model Performance**: Random Forest achieved **78.6% R²**, explaining most price variability
- **Algorithm Comparison**: Systematically evaluated Stepwise Regression, Random Forest, and Gradient Boosting
- **Feature Engineering**: Created derived variables (property age, amenity flags) to improve predictions
- **Robust Validation**: Train/test split with residual analysis confirming model validity

## Methodology
| Stage | Activities |
|-------|------------|
| **Data Preparation** | Cleaning, missing value handling, outlier treatment |
| **Feature Engineering** | Created `has_pool`, `has_renovation`, `age` variables |
| **Model Training** | Stepwise Regression, Random Forest, GBM with hyperparameter tuning |
| **Evaluation** | RMSE, MAE, R² metrics on held-out test data |

## Model Comparison
| Model | R² Score | Notes |
|-------|----------|-------|
| Random Forest | **78.6%** | Best performer |
| Gradient Boosting | ~75% | Strong but slower |
| Stepwise Regression | ~65% | Interpretable baseline |

## Technologies
| Category | Tools |
|----------|-------|
| Language | R |
| ML Framework | Tidymodels, RandomForest, GBM |
| Data Processing | Tidyverse, dplyr |
| Visualization | ggplot2 |

## How to Run
```r
source("Analysis.R")
```

## Skills Demonstrated
- **Machine Learning** - Ensemble methods, model selection, hyperparameter tuning
- **Predictive Analytics** - Regression modeling, feature importance analysis
- **Statistical Analysis** - Residual diagnostics, cross-validation, performance metrics
- **R Programming** - Tidymodels ecosystem, data manipulation, visualization
- **Feature Engineering** - Domain-driven variable creation, transformation

