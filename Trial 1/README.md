# Trial 1: original Decision Tree

## Folder layout

```text
Trial 1/
  Excel Sheet Inputs/
    H2_training_70.xlsx
    H2_validation_15.xlsx
    H2_testing_15.xlsx
  Model/
    train_baseline.py
    decision_tree_baseline.joblib
  Processing/
    preprocessing.py
    fitted_preprocessor.joblib
  Outputs/
    all_split_metrics.csv
    classification_*_metrics.csv
    classification_*_confusion_matrix.csv
    classification_*_report.txt
    execution_log.txt
    00_overview.html
  run_trial.py
  requirements.txt
```

Excel Sheet Inputs contains the unchanged 70/15/15 data: 20,249 training, 4,339 validation, and 4,340 testing bookings. Each file keeps 23 predictors, IsCanceled for learning/scoring, and SourceRow for tracing. IsCanceled and SourceRow are excluded from predictors.

Model contains the original untuned DecisionTreeClassifier(random_state=42) code and complete fitted pipeline. Processing contains the original input reader and preprocessing code, plus the fitted preprocessing object extracted from that pipeline. The pipeline replaces missing values and converts categories to numeric one-hot features. Preprocessing is fitted on training only.

Outputs contains the original execution reports copied from Trial 0, using the exact same data and fitted model. These are historical results, not a newly trained experiment. The HTML report titles and run timestamps preserve that provenance. Recorded accuracy: training 99.64%, validation 87.09%, testing 86.57%.

## Run from Trial 1

```powershell
python -m pip install --use-feature=truststore -r requirements.txt
python run_trial.py
```

This fits on training and scores training and validation, saves the fitted pipeline under Model, saves the fitted preprocessor under Processing, and writes reports under Outputs. To reproduce the fixed experiment including testing, use python run_trial.py --evaluate-test. Test scores must not be used to tune the model. The new folder structure preserves the original model settings and preprocessing.

To load the existing fitted pipeline without training:

```python
import joblib
model = joblib.load("Model/decision_tree_baseline.joblib")
```

The inherited pre-split IQR filtering and uncertain booking-time availability of ADR remain limitations of this filtered-data experiment.
