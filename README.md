# MSCS-634 Lab 4

## Regression Analysis with Regularization Techniques

**Mohammed Rehaan**  
**005046243**  
**University of the Cumberlands**  
**MSCS-634-B01: Advanced Big Data and Data Mining**  
**Dr. Penmatsa**

## Purpose

This lab compares several regression approaches using the scikit-learn Diabetes dataset, which contains 442 patient observations, ten baseline health measurements, and a one-year disease progression target. The notebook implements a simple linear regression model using BMI, multiple linear regression using all available features, polynomial regression at degrees 1, 2, 3, and 5, plus Ridge and Lasso regression. Each model is evaluated with Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

## Files

- `Mohammed_Rehaan_MSCS_634_Lab_4.ipynb`: Executed notebook containing data checks, model training, metric tables, visualizations, and analysis.
- `requirements.txt`: Python packages used by the notebook.

## Key insights

The dataset had no missing feature values, missing target values, or duplicate feature rows, so no cleaning rows were removed. The simple BMI model achieved an RMSE of 63.73 and an R-squared of 0.2334 on the held-out test set. Using all ten measurements substantially improved the result, as multiple linear regression reduced RMSE to 53.85 and increased R-squared to 0.4526.

Polynomial regression shows why model complexity must be checked against unseen data. Degree 2 had a training R-squared of 0.6062 but a lower test R-squared of 0.4156. Degrees 3 and 5 fit the training data far too closely, with test R-squared values below zero. This demonstrates overfitting because the extra polynomial terms learned patterns that did not generalize to the test set.

Ridge and Lasso alpha values were selected with five-fold cross-validation on the training data. Ridge selected alpha 33.598183 and achieved RMSE 53.49 with R-squared 0.4601. Lasso selected alpha 1.128838 and produced the best held-out performance with RMSE 53.08 and R-squared 0.4682. Lasso also set one of ten coefficients to zero, which shows how L1 regularization can simplify a model while maintaining good predictive performance.

## Decisions and challenges

All models used the same 80/20 train-test split with `random_state=42`, which keeps the performance comparison fair and reproducible. Polynomial models used scaling after feature expansion because interaction and squared terms can have different ranges. Regularized models also used scaling because penalty terms depend on coefficient magnitude.

The main challenge was balancing model flexibility with generalization. Higher-degree polynomial models can look strong on training data, but the test results showed that they are not appropriate for this small dataset. The regularized linear models were more stable and easier to interpret.

## Running the notebook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
