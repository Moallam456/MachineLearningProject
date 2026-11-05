# Machine Learning Project

We are a team of six working on a Semester Seven machine learning project using the H2 city-hotel booking dataset. We divided the work into three pairs: regression for estimating average daily rate (ADR), classification for predicting cancellation, and clustering for grouping bookings.

Repository: [MachineLearningProject on GitHub](https://github.com/Moallam456/MachineLearningProject)

## Milestone 1 status

Our regression workflow includes data inspection and cleaning, a chronological split, reusable preprocessing, a mean predictor, default linear regression, and direct-versus-GD comparisons. Both the original GD and the momentum version are preserved.

For classification, we use an untuned Decision Tree to predict whether a booking is canceled. For clustering, we use K-Means with three clusters to explore groups of similar bookings. The sections below cover the actual inputs, preprocessing, commands and results for both tasks.

| Task | What we predict or study | Current baseline | Data used |
| --- | --- | --- | --- |
| Regression | Average daily rate (`ADR`) | Mean predictor and linear regression, including direct and GD fitting | Original H2 data with an arrival-date split |
| Classification | Cancellation (`IsCanceled`) | `DecisionTreeClassifier(random_state=42)` | 28,928 filtered bookings in supplied 70/15/15 splits |
| Clustering | Groups of similar bookings | `KMeans(n_clusters=3, random_state=42, n_init=10)` | 53,428 bookings after removing repeated rows |

Our regression test set remains reserved. The classification package already contains test results from its original experiment; we report those separately below. Clustering uses the full deduplicated dataset for exploration.

We keep our working notes in [DocumentationNotes.txt](DocumentationNotes.txt) and [NotesOnRegression.txt](NotesOnRegression.txt). The `reports/` folder is reserved for our project reports.

## Team

We work in three pairs. We will complete the names, IDs, accounts and individual contributions after confirming them as a team.

| Member | Name | Student ID | Task |
| --- | --- | --- | --- | --- | --- |
| 1 | [Mohamed Ahmed Ali] | [23P0212] | Regression |
| 2 | [Mohamed Ali Geasa] | [23P0366] | Regression |
| 3 | [Mostafa Ahmed Elmoriah] | [23P0036] | Classification |
| 4 | [Basmala Helmy Lotfy] | [23P0359] | Classification |
| 5 | [Mahmoud Karam] | [23P0218] | Clustering |
| 6 | [Youssef Osama] | [23P0131] | Clustering |

## Folder structure

| Path | Purpose |
| --- | --- |
| `data/raw/` | Original H2 files; retain unchanged. |
| `data/processed/` | Optional intermediate datasets. Current regression commands load the raw CSV directly. |
| `src/data/` | Loading, inspection, cleaning, splitting and training exploration used by regression. |
| `src/preprocessing/` | Reusable preparation of task inputs. |
| `src/regression/` | Regression baselines and learning experiments. |
| `Trial 1/` | Classification inputs, preprocessing, Decision Tree, saved results and HTML reports. |
| `src/classification/Trial 1/` | Additional copy of the same classification package. |
| `src/clustering/` | Clustering preprocessing and the K-Means baseline. |
| `src/evaluation/` | Regression metric functions and result plots. |
| `notebooks/` | Optional supplementary exploration. |
| `outputs/figures/` | Generated figures. |
| `outputs/metrics/` | Generated metrics and GD cost histories. |
| `reports/` | Space for milestone reports and workflow documentation. |
| `artifacts/` | Space for final fitted models/pipelines. The classification trial currently stores its fitted objects in its own folders. |
| `requirements.txt` | Team package versions. |

## Set up the environment

We used Python 3.14.3 on Windows for the local regression development environment. The project interpreter is stored in `.venv`. Each teammate creates their own environment; do not commit `.venv`.

In PowerShell:

```powershell
git clone https://github.com/Moallam456/MachineLearningProject.git
cd MachineLearningProject
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

If the repository is already cloned, start from its root and skip the clone command. The root is the folder containing `src`, `data` and this README.

In VS Code, use **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`. The commands below explicitly name that interpreter, so activating the environment is optional.

The main packages are NumPy, pandas, Matplotlib, seaborn, scikit-learn, ipykernel and joblib. `requirements.txt` also includes installed dependencies and versions. Classification reads Excel files and also needs `openpyxl`, which is included in `Trial 1/requirements.txt`; its installation command is shown in the classification section. We update dependency versions together when needed.

On macOS/Linux, create the environment using a compatible Python interpreter and use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Dataset and regression split

The expected file is `data/raw/H2.csv`, containing 79,330 rows and 31 columns. Each row represents a booking. We found 25,902 exact repeated rows in the raw CSV. Regression keeps them for its initial baseline, while clustering removes them. Classification uses its own supplied filtered Excel splits. The source is Antonio, de Almeida and Nunes, [Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/), DOI [10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).

| Partition | Arrival dates | Rows |
| --- | --- | ---: |
| Training | July 2015 to December 2016 | 51,822 |
| Validation | January to April 2017 | 12,297 |
| Reserved test | May to August 2017 | 15,211 |

Our regression split is implemented in `src/data/split_h2.py`. Training fits the model and preprocessing; validation supports development. After final model selection, the reserved test set is used for the final assessment, without further model selection based on its results.

Our current regression experiment assumes a pre-arrival snapshot and estimates recorded ADR retrospectively. We will confirm the final intended prediction point in our team report. An arrival-date split alone does not establish a live forecast: historical feature snapshots, canceled-booking eligibility and training-label availability still need review.

## Regression cleaning and preprocessing

Our regression cleaning strips outer text whitespace, converts `Children="NA"` to numeric missing values, converts `Country="NULL"` to missing, and parses `ReservationStatusDate` for auditing. `Agent` and `Company` retain meaningful NULL absence markers. The initial regression baseline retains repeated rows, zero values and extreme ADR records while documenting them for review.

Our selected regression numerical inputs are `LeadTime`, `Adults`, `Children`, `Babies`, `StaysInWeekendNights` and `StaysInWeekNights`. The categorical inputs are `ArrivalDateMonth`, `ReservedRoomType`, `Meal` and `MarketSegment`.

Numeric missing values use training medians and numeric columns use `StandardScaler`. Categorical inputs use `OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)`. The checked output has 34 columns. Validation reuses the fitted preprocessing. ADR stays in its original units.

`IsCanceled`, `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType` and other unselected columns are excluded by the feature allowlist. These fields include cancellation outcomes and room assignments that need careful timing checks.

Solver comparison scripts additionally standardize all encoded columns, including category indicators. This repeats the numerical scaling. Moving standardization to one step after imputation and encoding is a planned refactor, not a change already implemented.

## Run our regression workflow

Run all commands from the repository root.

### Inspect and prepare

```powershell
.\.venv\Scripts\python.exe -m src.data.inspect_h2
.\.venv\Scripts\python.exe -m src.data.clean_h2
.\.venv\Scripts\python.exe -m src.data.split_h2
.\.venv\Scripts\python.exe -m src.data.explore_training
.\.venv\Scripts\python.exe -m src.preprocessing.regression
```

### Run the toy comparison

```powershell
.\.venv\Scripts\python.exe -m src.regression.compare_linear
```

This uses three hypothetical examples, not the H2 dataset. Expected original-unit intercept: 90; slope: 0.25; predictions: 95, 100 and 105. Both methods have final J 25. The script checks prediction agreement.

### Train the mean and linear baselines

```powershell
.\.venv\Scripts\python.exe -m src.regression.train_baseline
```

### Compare fitting methods on H2

```powershell
.\.venv\Scripts\python.exe -m src.regression.compare_h2_solvers
.\.venv\Scripts\python.exe -m src.regression.compare_h2_solvers_V2
```

The V2 comparison imports `src.regression.linear_solvers_V2`. Keep this import consistent if files are renamed.

All fitting approaches use the same linear model family. The educational normal-equation implementation requires full column rank and solves a linear system without explicitly forming an inverse. Plain GD and momentum GD are iterative alternatives. The final test partition is not evaluated by these commands.

## Generate the regression report plots

Our plotting module uses the existing toy GD implementation, the regression pipeline and saved solver histories. It is included in `src/evaluation/plot_regression_results.py`. After generating the baseline and comparison CSVs, run:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.plot_regression_results
```

We save `toy_comparison.png`, `validation_diagnostics.png` and `h2_cost_comparison.png` in `outputs/figures/`, and validation error checks in `outputs/metrics/validation_plot_checks.csv`. The script checks that the recreated validation metrics agree with our saved baseline.

## Preliminary regression results

| Model | Partition | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Mean predictor | Training | 27.5601 | 43.2680 | 0.0000 |
| Mean predictor | Validation | 24.8580 | 35.3890 | -0.0017 |
| Linear regression | Training | 18.0817 | 34.5782 | 0.3613 |
| Linear regression | Validation | 17.4110 | 23.9614 | 0.5408 |

MAE and RMSE use ADR units; lower is better. R2 is a variance-relative score. J equals half MSE. Validation RMSE is approximately 32.3% lower than the mean predictor's. We use the validation diagnostics to inspect residuals and unusual predictions before choosing our final model.

| GD experiment | Learning rate | Momentum | Gradient tolerance | Checked updates |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.5 | 0 | 1e-10 | 9,882 |
| V2 | 0.5 | 0.9 | 1e-8 | 587 |

Both reach training J about 597.827508 and essentially the same validation predictions. Both momentum and tolerance changed; compare at the same tolerance to isolate momentum's effect. Runtime and exact update counts can vary by machine and numerical-library version.

## Generated outputs

| Command | Output |
| --- | --- |
| `src.evaluation.plot_regression_results` | `outputs/figures/toy_comparison.png`, `validation_diagnostics.png`, `h2_cost_comparison.png`; `outputs/metrics/validation_plot_checks.csv` |
| `src.data.explore_training` | `outputs/metrics/training_summary.csv`, `training_quality_checks.csv`; `outputs/figures/training_adr_distribution.png` |
| `src.regression.train_baseline` | `outputs/metrics/regression_baselines.csv` |
| `src.regression.compare_h2_solvers` | `outputs/metrics/linear_solver_comparison.csv`, `h2_gd_cost_history.csv` |
| `src.regression.compare_h2_solvers_V2` | `outputs/metrics/linear_solver_comparison_momentum.csv`, `h2_gd_momentum_cost_history.csv` |
| `Trial 1/run_trial.py` | Fitted pipeline in `Trial 1/Model/`, fitted preprocessor in `Trial 1/Processing/`, and training/validation reports in `Trial 1/Outputs/` |
| `Trial 1/run_trial.py --evaluate-test` | The same training outputs plus test metrics, confusion matrix and classification report |
| `src/clustering/preprocessing_clustering.py` | `H2_cleaned_for_clustering.csv` in the working directory; our commands below place it in `data/processed/` |
| `src/clustering/clustering.py` | Console output containing Davies-Bouldin index, sampled silhouette score, scaled cluster means and cluster sizes |

The original and momentum histories use different filenames. Training figures can use a log count axis or a restricted plotting range without modifying the modeling dataset.

## Code file guide

| Module or file | Function |
| --- | --- |
| `src/data/inspect_h2.py` | Raw shape, types, markers and repeated-row inspection. |
| `src/data/clean_h2.py` | Reusable fixed cleaning rules used by regression. |
| `src/data/split_h2.py` | Reproducible arrival-date partitions. |
| `src/data/explore_training.py` | Training summaries, quality flags and ADR histogram. |
| `src/preprocessing/regression.py` | Target, input lists and regression preprocessor factory. |
| `src/evaluation/plot_regression_results.py` | Saves our toy comparison, validation diagnostics and GD cost plots. |
| `src/evaluation/regression_metrics.py` | MAE, MSE, RMSE, R2 and J. |
| `src/regression/gradient_descent.py` | One-feature GD for the toy exercise. |
| `src/regression/compare_linear.py` | Toy direct/GD fits and prediction-agreement check. |
| `src/regression/linear_solvers.py` | Multiple-feature plain GD, normal equation and prediction. |
| `src/regression/compare_h2_solvers.py` | Original solver comparison and saved metrics/history. |
| `src/regression/linear_solvers_V2.py` | Momentum-capable GD and optimizer metadata. |
| `src/regression/compare_h2_solvers_V2.py` | Momentum comparison with separate saved outputs. |
| `src/regression/train_baseline.py` | Mean and default linear-regression training/evaluation. |
| `src/__init__.py`, `src/evaluation/__init__.py`, `src/regression/__init__.py` | Python package markers. |
| `Trial 1/run_trial.py` | Entry point for classification training and optional test scoring. |
| `Trial 1/Processing/preprocessing.py` | Loads the classification splits, selects 23 features and builds preprocessing. |
| `Trial 1/Model/train_baseline.py` | Builds the Decision Tree pipeline, evaluates it and saves fitted objects and reports. |
| `src/clustering/preprocessing_clustering.py` | Removes duplicates, selects inputs, imputes, scales and ordinal-encodes the clustering data. |
| `src/clustering/clustering.py` | Fits the three-cluster K-Means baseline and prints evaluation and cluster summaries. |
| `DocumentationNotes.txt`, `NotesOnRegression.txt` | Working documentation and learning notes. |

## Classification

We use classification to predict booking cancellation. Our target is `IsCanceled`, where `1` means canceled and `0` means not canceled. We started with an untuned Decision Tree so that we have a baseline before comparing models or changing parameters.

### Data and split

Our classification files are in `Trial 1/Excel Sheet Inputs/`. They contain 28,928 bookings in a prepared, shuffled 70/15/15 split of a filtered version of H2.

| Partition | File | Rows | Canceled bookings |
| --- | --- | ---: | ---: |
| Training | `H2_training_70.xlsx` | 20,249 | 9,464 |
| Validation | `H2_validation_15.xlsx` | 4,339 | 2,028 |
| Test | `H2_testing_15.xlsx` | 4,340 | 2,029 |

Each file contains 23 predictors, the target and `SourceRow`, which lets us trace a booking to its source. Neither `IsCanceled` nor `SourceRow` is passed to the model. Cancellation makes up about 46.74% of each split. The supplied files have no repeated `SourceRow` values within or across the three partitions.

These splits are separate from the chronological regression split. The classification package records that IQR filtering happened before splitting, but it does not include the code that created the filtered splits. We can rerun the model from the supplied Excel files; reproducing the filtering from the raw CSV still needs that earlier preparation step.

### Inputs and preprocessing

We use the following numerical inputs:

`LeadTime`, `ArrivalDateYear`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `ADR`, `RequiredCarParkingSpaces` and `TotalOfSpecialRequests`.

Our categorical inputs are:

`ArrivalDateMonth`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `Agent` and `CustomerType`.

The loader strips whitespace, treats empty strings and `NULL`/`null` as missing, checks that the target contains only `0` and `1`, and converts numerical inputs to numbers. We fill missing `Agent` values with category `"0"`. The pipeline fills other numerical missing values with the training median and categorical missing values with the training mode, then applies `OneHotEncoder(handle_unknown="ignore")`. We do not scale the numerical features for this tree baseline.

We fit preprocessing and the classifier together on training data only, then reuse the same fitted pipeline on validation and test inputs. Our feature allowlist excludes `ReservationStatus`, `ReservationStatusDate` and `AssignedRoomType`, along with all other unselected columns.

### Model and run commands

Our baseline is `DecisionTreeClassifier(random_state=42)` with the remaining parameters left at their defaults. We have not limited its depth or tuned its settings in this trial.

From the repository root, install the classification dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ".\Trial 1\requirements.txt"
.\.venv\Scripts\python.exe ".\Trial 1\run_trial.py"
```

This trains on the training split, scores training and validation, and saves the fitted pipeline and preprocessor. It does not score the test split by default. To reproduce the fixed experiment including test scoring, run:

```powershell
.\.venv\Scripts\python.exe ".\Trial 1\run_trial.py" --evaluate-test
```

The second command also trains the pipeline again before scoring. We keep test results out of parameter selection. A copy of the same trial is also present under `src/classification/Trial 1/`; the commands and links here use the root-level `Trial 1/` folder.

### Results

The following values come from our saved [all-split metrics](Trial%201/Outputs/all_split_metrics.csv). Trial 1 packages the original Trial 0 inputs and fitted model, and its saved reports were copied from that original run. Reorganizing the files did not create a new experiment.

| Partition | Accuracy | Precision for cancellation | Recall for cancellation | F1 for cancellation | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 99.64% | 1.0000 | 0.9923 | 0.9961 | 1.0000 |
| Validation | 87.09% | 0.8692 | 0.8521 | 0.8606 | 0.8714 |
| Test, original recorded run | 86.57% | 0.8548 | 0.8586 | 0.8567 | 0.8666 |

Values are rounded; the CSV keeps the full precision. Always predicting that a booking is not canceled gives about 53.26% validation accuracy, so the tree improves on that simple reference by about 33.83 percentage points.

Our validation confusion matrix is:

| Actual outcome | Predicted not canceled | Predicted canceled |
| --- | ---: | ---: |
| Not canceled | 2,051 | 260 |
| Canceled | 300 | 1,728 |

The model correctly identifies 1,728 of 2,028 validation cancellations and misses 300. The gap between 99.64% training accuracy and 87.09% validation accuracy suggests overfitting, so pruning or limiting tree depth is a useful next comparison.

These results describe the filtered dataset. Filtering before splitting can let held-out data affect which records are retained. We also still need to confirm when `ADR` and other changeable booking details would be available if we wanted to use this as a booking-time prediction model.

### Code and saved outputs

| Path | What we keep there |
| --- | --- |
| `Trial 1/run_trial.py` | Classification entry point. |
| `Trial 1/Processing/preprocessing.py` | Input reader, feature lists and preprocessing builder. |
| `Trial 1/Model/train_baseline.py` | Model builder, evaluation functions and training command. |
| `Trial 1/Model/decision_tree_baseline.joblib` | Complete fitted preprocessing-and-model pipeline. |
| `Trial 1/Processing/fitted_preprocessor.joblib` | The preprocessing object extracted from that pipeline. |
| `Trial 1/Outputs/classification_*_metrics.csv` | Per-split accuracy, majority-class reference, precision, recall, F1 and ROC-AUC. |
| `Trial 1/Outputs/classification_*_confusion_matrix.csv` | Confusion matrices with actual rows and predicted columns in class order `[0, 1]`. |
| `Trial 1/Outputs/classification_*_report.txt` | Per-class and average classification scores. |
| `Trial 1/Outputs/classification_decision_tree_baseline.csv` | Validation summary with training accuracy and split sizes. |
| `Trial 1/Outputs/all_split_metrics.csv` | Combined metrics from the original recorded run. |
| `Trial 1/Outputs/00_overview.html` | Existing overview, with separate training, validation and testing HTML reports. |
| `Trial 1/Outputs/run_manifest.json`, `organization_manifest.json`, `execution_log.txt` | Original run details and records of the file reorganization. |

For a readable view of the existing results, open the [classification overview](Trial%201/Outputs/00_overview.html). Regular training runs refresh the per-split CSV/text reports and fitted objects. They do not rebuild the combined CSV, HTML reports or provenance manifests, so those files remain records of the original run. The full saved pipeline already includes preprocessing; we pass raw feature rows to its `predict` method.

## Clustering

We use clustering to explore bookings with similar characteristics without using cancellation labels. Our first model is K-Means with three clusters. We chose `k=3` as a starting point for Milestone 1; we have not selected it through an elbow or silhouette comparison yet.

### Data and preprocessing

We start with `data/raw/H2.csv` and remove exact duplicate rows. This leaves 53,428 bookings from the original 79,330. We then drop `ReservationStatus`, `ReservationStatusDate`, `IsCanceled`, `Country`, `Agent` and `Company`, leaving 25 input columns.

We convert `ArrivalDateMonth` to numbers from 1 to 12 and strip whitespace from the remaining categorical values. The numerical pipeline fills missing values with the median and applies `StandardScaler`. The categorical pipeline uses a constant missing-value imputer followed by `OrdinalEncoder`. The script converts categorical values to strings before that pipeline, so a missing categorical value can become the text `"nan"` rather than being filled by the imputer.

The resulting matrix has 18 numerical inputs and seven encoded categorical inputs. The categorical inputs are `Meal`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `AssignedRoomType`, `DepositType` and `CustomerType`. All other retained columns, including `ADR`, are processed as numerical inputs. The script saves a fully numerical matrix with shape `(53428, 25)`.

We fit clustering preprocessing and K-Means on the full deduplicated dataset across all arrival dates. We do not use the regression or classification partitions for this exploratory baseline. The retained `AssignedRoomType`, `BookingChanges` and other booking details mean these groups describe the recorded bookings; we have not established that all inputs would be known when a new booking is made.

### Model and run commands

Our settings are:

```python
KMeans(n_clusters=3, random_state=42, n_init=10)
```

The scripts read and write filenames relative to the current working directory. To keep the raw data unchanged and place the prepared matrix under `data/processed/`, run the following from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path ".\data\processed" | Out-Null
Push-Location ".\data\processed"
try {
    Copy-Item -LiteralPath "..\raw\H2.csv" -Destination ".\H2.csv" -Force
    & "..\..\.venv\Scripts\python.exe" "..\..\src\clustering\preprocessing_clustering.py"
    & "..\..\.venv\Scripts\python.exe" "..\..\src\clustering\clustering.py"
}
finally {
    Pop-Location
}
```

This leaves a working copy of `H2.csv` and the generated `H2_cleaned_for_clustering.csv` in `data/processed/`. The model script prints its scores, cluster means and cluster sizes to the console. It does not currently save cluster labels, centroids, figures, metric files or a fitted K-Means model.

### Baseline results

We ran the existing clustering scripts on 5 November 2026 using the project environment. The prepared matrix contained 53,428 rows and 25 columns.

| Metric | Result | How we calculate it |
| --- | ---: | --- |
| Davies-Bouldin index | 1.9192 | All 53,428 prepared bookings; lower values indicate better separation relative to cluster spread. |
| Silhouette score | 0.2232 | A 10,000-row sample selected with `random_state=42`; higher values indicate better separation and cohesion. |

The silhouette score shows some grouping, but the clusters still overlap. We treat this as a baseline to improve rather than a final segmentation.

To make the groups easier to describe, we matched the cluster labels back to the deduplicated raw rows and calculated the following means in their original units. These are different from the standardized numerical means printed by `clustering.py`.

| Cluster | Bookings | Share | Mean lead time, days | Mean ADR | Mean weekend nights | Mean week nights | Mean special requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32,066 | 60.02% | 87.90 | 100.51 | 0.89 | 2.28 | 0.71 |
| 1 | 7,066 | 13.23% | 35.70 | 88.84 | 0.61 | 1.73 | 0.47 |
| 2 | 14,296 | 26.76% | 75.50 | 145.43 | 0.96 | 2.54 | 0.82 |

Our initial interpretation is:

- **Cluster 0:** the largest group, with the longest average lead time and an ADR between the other two groups.
- **Cluster 1:** shorter-notice bookings with lower ADR, shorter stays and fewer special requests on average.
- **Cluster 2:** higher-ADR bookings with longer stays and more special requests on average.

Cluster IDs are labels, not rankings. These descriptions summarize bookings rather than unique guests, and the table only shows a few of the 25 inputs used to form the groups. Percentages may not sum to exactly 100% because of rounding.

### Limitations and next steps

Our ordinal encoding gives categories numerical distances that may not reflect meaningful similarity. We also standardize numerical inputs while leaving category codes on their encoded scale, so their influence on K-Means needs review. Encoding months as 1 to 12 does not represent the closeness of December and January.

Next, we want to compare several values of `k`, review feature selection and categorical encoding, and save cluster labels and summaries so the analysis is easier to reproduce and inspect. We have not run PCA or compared other clustering algorithms in this baseline.

The two implementation files are [preprocessing_clustering.py](src/clustering/preprocessing_clustering.py) and [clustering.py](src/clustering/clustering.py).

Each task currently has its own preprocessing. Regression keeps the original rows and fits transformations on its chronological training partition; classification uses its supplied filtered splits and fits transformations on training only; clustering removes duplicates and fits on the complete exploratory dataset.

## Submission

Submitted source commit: **[Fill with the actual output of `git rev-parse HEAD` for the reviewed submission source.]**

Include the GitHub URL and source commit in the submission copy's README. The team ZIP must contain the applicable report, source, dependency file, metrics and figures. Exclude `.venv`, `.git`, and serialized fitted models/pipelines. Keep the code and commands needed to regenerate artifacts. Fitted artifacts must be available locally for the final discussion, as required by the course brief.

## References

- [Hotel booking demand datasets and field definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [Avoiding preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Stanford CS229 linear regression notes](https://cs229.stanford.edu/summer2023/cs229-notes1.pdf)
- [Momentum](https://d2l.ai/chapter_optimization/momentum.html)
- [Additional EDA example](https://www.kaggle.com/code/ahmedwaelz/complete-eda-and-visualization-for-csv-files#Import-Libraries)
- [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [pandas read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Matplotlib histograms](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)
