# Credit-Risk-Default-Prediction



Predicting whether a borrower will become seriously delinquent (90+ days past due) within two years, using the Kaggle "Give Me Some Credit" dataset (150,000 borrowers, \~6.7% defaults).



\## Approach

\- \*\*Cleaning:\*\*

&#x20; - Removed one row with age 0.

&#x20; - Replaced placeholder codes (96/98) in the three late-payment columns with the median and added a `LateCodeFlag` column, because those rows defaulted at 54.6% versus 6.7% overall.

&#x20; - Capped credit card utilisation at 2.

&#x20; - Filled missing income and dependents with the median, and added an `IncomeMissing` flag.

&#x20; - Capped `DebtRatio` and `MonthlyIncome` at the 99th percentile.

\- \*\*Split:\*\* 80/20 train/test, stratified to keep the default rate equal in both.

\- \*\*Imbalance:\*\* class weights (`class\_weight="balanced"`, `scale\_pos\_weight`) so that missing a defaulter costs more than a false alarm.

\- \*\*Models:\*\* logistic regression (baseline), random forest, XGBoost.

\- \*\*Metrics:\*\* ROC-AUC, PR-AUC, precision, recall and F1. Accuracy is not used, because predicting "no default" for everyone would already score \~93%.



\## Results (test set, threshold 0.5)



| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |

|---|---|---|---|---|---|

| Logistic Regression | 0.862 | 0.383 | 0.225 | 0.749 | 0.346 |

| Random Forest | 0.858 | 0.381 | 0.299 | 0.630 | 0.406 |

| XGBoost | 0.869 | 0.397 | 0.220 | 0.777 | 0.343 |



!\[Model comparison](images/model\_comparison.png)

!\[XGBoost confusion matrix](images/confusion\_matrix\_xgboost.png)



\## Findings

\- XGBoost performed best on ROC-AUC, PR-AUC and recall (it catches about 78% of defaulters).

\- The gap over logistic regression is small (0.869 vs 0.862 ROC-AUC), so on this dataset a simple model is already competitive. The curves overlap heavily for this reason.

\- Precision is low (\~22%) because the models flag many borrowers in order to catch most defaulters. In lending, a missed defaulter is usually costlier than a false alarm, so this trade-off can be acceptable.

\- Random forest has the best precision and F1, but catches fewer defaulters (63% recall).



\## Limitations and next steps

\- Missing values were filled before the train/test split. A stricter pipeline would fit the imputer and the percentile caps on training data only.

\- `DebtRatio` has a large share of extreme values (about 16% above 100) that were capped rather than investigated. They may be linked to missing income.

\- The decision threshold is fixed at 0.5. Tuning it to a business cost would change the precision/recall trade-off.

\- No hyperparameter tuning or cross-validation yet.

\- Next: add feature importance (SHAP) and try the Home Credit or Lending Club datasets.



\## How to run

1\. Download `cs-training.csv` from \[Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) and place it in a folder called `data/`.

2\. Install the dependencies: `pip install -r requirements.txt`

3\. Open `notebooks/credit\_risk\_analysis.ipynb` and run all cells.

