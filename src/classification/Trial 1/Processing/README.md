# Processing

preprocessing.py contains the original Excel/CSV input reader, the 23-feature allowlist, missing-value handling, and the preprocessing builder.

Numeric missing values use the training median. Categorical missing values use the training mode. Missing Agent uses category 0. Categories are converted with one-hot encoding; unseen validation/test categories are ignored safely. IsCanceled and SourceRow are excluded from predictors.

fitted_preprocessor.joblib is the fitted preprocessing object extracted from Model/decision_tree_baseline.joblib. The full pipeline in Model already includes this preprocessing; do not apply preprocessing twice when calling that pipeline's predict method.
