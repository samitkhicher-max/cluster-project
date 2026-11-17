# 🏥 Health Data — Disease Prediction (Diabetes) on Apache Spark
### Distributed Machine Learning & Dashboard for Cluster Computing

An end-to-end distributed health data analytics platform designed for **Cluster Computing**. Built using **Apache Spark 4.2.0 (PySpark MLlib)**, distributed feature normalization, physiological missingness imputation, distributed **Logistic Regression** optimization, and an interactive **Streamlit Dashboard** featuring real-time Confusion Matrix analysis and patient risk prediction.

---

## 📌 Project Overview & Clinical Context

- **Objective**: Predict whether a patient is diabetic (`Outcome = 1`) or healthy (`Outcome = 0`) based on 8 clinical diagnostic measurements.
- **Dataset**: Pima Indians Diabetes Dataset (`diabetes.csv`, 768 patient records).
- **Core Engineering Focus**:
  - **Cluster Parallelization**: Distributed DataFrame transformations across Spark executors and partitions.
  - **Clinical Preprocessing**: Auditing and imputing biologically impossible zero readings in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI.
  - **Data Leakage Prevention**: Strict featurization ordering (Train/Test split performed before fitting Imputer and Scaler).
  - **Distributed Optimization**: Distributed Quasi-Newton L-BFGS optimizer via Spark MLlib with tree-aggregated gradient reduction.
  - **Interactive Dashboard**: Streamlit + Plotly visualizer with live confusion matrix updates, ROC curves, feature odds ratios, and patient risk prediction.

---

## 🏗️ Architecture & Cluster Computing Principles

```
                                  ┌───────────────────────────────┐
                                  │   Spark Driver (app.py / CLI) │
                                  │   SparkSession / DAG Planner  │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │    Worker Executor 1      │                                     │    Worker Executor 2      │
   │  ┌─────────────────────┐  │                                     │  ┌─────────────────────┐  │
   │  │ Partition 1 (RDD)   │  │                                     │  │ Partition 2 (RDD)   │  │
   │  │ • Median Imputation │  │                                     │  │ • Median Imputation │  │
   │  │ • VectorAssembler   │  │                                     │  │ • VectorAssembler   │  │
   │  │ • StandardScaler    │  │                                     │  │ • StandardScaler    │  │
   │  └──────────┬──────────┘  │                                     │  └──────────┬──────────┘  │
   └─────────────┼─────────────┘                                     └─────────────┼─────────────┘
                 │                                                                 │
                 └───────────────────────────────┬─────────────────────────────────┘
                                                 │ TreeAggregate Gradients
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │ Distributed L-BFGS Optimizer  │
                                  │ (Iterative Parameter Update)  │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │     Fitted PipelineModel      │
                                  │   (Evaluated on Test RDD)     │
                                  └───────────────────────────────┘
```

### Key Distributed Computing Concepts Demonstrated:
1. **Partitioning & Resilient Distributed Datasets (RDDs)**: Data is partitioned across cluster cores to balance computational load.
2. **Distributed L-BFGS (Quasi-Newton)**: Unlike sequential SGD, Spark MLlib uses distributed L-BFGS where each worker calculates partial gradients on its partition. These gradients are aggregated via a binary communication tree (`treeAggregate`), minimizing driver bandwidth bottlenecks.
3. **Lazy Evaluation & DAG Optimization**: Catalyst Optimizer plans logical and physical execution trees, chaining projections and filters before execution.
4. **Zero-Leakage Featurization**: Preprocessing parameters (medians, means $\mu$, and standard deviations $\sigma$) are computed exclusively on the training partition and broadcast to transform test partitions.

---

## 📊 Preprocessing & Clinical Missingness Handling

In health datasets, measurements such as **Glucose, Blood Pressure, Insulin, Skin Thickness, and BMI cannot physically be 0 in a living human**. In the raw dataset, 0 represents an unrecorded or missing observation:

| Feature | Raw Zero Count | Missing Percentage | Imputation Strategy |
| :--- | :---: | :---: | :---: |
| **Glucose** | 5 | 0.7% | Median Imputation |
| **BloodPressure** | 35 | 4.6% | Median Imputation |
| **SkinThickness** | 227 | 29.6% | Median Imputation |
| **Insulin** | 374 | 48.7% | Median Imputation |
| **BMI** | 11 | 1.4% | Median Imputation |

### Standardization (Feature Normalization)
All 8 features are transformed via `StandardScaler(withMean=True, withStd=True)`:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$
This ensures large numeric scales (e.g., Insulin values up to 846) do not dominate smaller scale features (e.g., Diabetes Pedigree Function ~0.5).

---

## 📈 Evaluation Results & Confusion Matrix

Evaluated on held-out test partition:

| Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **79.67%** | Overall correct diagnoses |
| **F1-Score** | **0.6667** | Harmonic mean of precision and recall |
| **Precision** | **0.7812** | Likelihood that a flagged patient is truly diabetic |
| **Recall (Sensitivity)** | **0.5814** | Fraction of diabetic patients detected |
| **Specificity** | **0.9125** | Fraction of healthy patients correctly cleared |
| **ROC-AUC** | **0.8640** | High discriminatory ability across all decision thresholds |

### Confusion Matrix (Test Partition: N = 123)

```
                       Predicted: Healthy (0)     Predicted: Diabetic (1)
 Actual: Healthy (0)       TN = 73 (59.3%)             FP = 7  (5.7%)
 Actual: Diabetic (1)      FN = 18 (14.6%)             TP = 25 (20.3%)
```

### Standardized Model Coefficients & Odds Ratios ($e^{\beta}$)
The model reveals clear physiological risk factors:
1. **Glucose ($\beta = +0.9877, \text{OR} = 2.69$)**: Every $1\sigma$ increase in plasma glucose multiplies diabetes odds by **2.69x**.
2. **BMI ($\beta = +0.6025, \text{OR} = 1.83$)**: Elevated Body Mass Index increases diabetes risk by **1.83x**.
3. **Pregnancies ($\beta = +0.3621, \text{OR} = 1.44$)**: History of gestational stress shows a **1.44x** risk multiplier.
4. **Diabetes Pedigree Function ($\beta = +0.3112, \text{OR} = 1.36$)**: Genetic inheritance score increases risk by **1.36x**.

---

## 🚀 Running the Project

### 1. Standalone Cluster CLI Training Script
Run distributed model training directly from terminal or submit to a Spark cluster:

```bash
# Standard local multi-core execution
python train_spark.py --data ./diabetes.csv

# Configure master, regularization, and max iterations
python train_spark.py --master "local[4]" --reg-param 0.05 --max-iter 150

# Spark-Submit command for standalone cluster or YARN/K8s
spark-submit --master local[*] train_spark.py
```

### 2. Interactive Streamlit Dashboard
Launch the web visualization platform:

```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`*

---

## 🖥️ Dashboard Visualization Features

The dashboard includes 6 interactive tabs:
1. **📈 Confusion Matrix & Evaluation**: High-contrast interactive Plotly heatmap, classification KPIs, ROC curve with AUC shading, and odds ratio charts.
2. **⚡ Cluster Computing Architecture**: Live SparkSession specifications, cluster topology diagrams, partition scaling benchmark runner, and execution DAG explorer.
3. **📊 Data Explorer & Missingness**: Raw record tables, physiological missingness audit charts, stratified histograms, and correlation heatmap.
4. **🧠 Preprocessing & ML Pipeline**: Visual representation of the 4-stage pipeline, hyperparameter sliders, and dual engine switch (PySpark MLlib vs. Scikit-Learn baseline).
5. **🩺 Patient Risk Predictor**: Clinical input form with standard physiological ranges, real-time risk gauge (0–100%), risk category badges, and radar charts comparing the patient against population medians.
6. **📑 Project Report & Export**: Summary report and one-click JSON/CSV download buttons.

---

## 🧪 Running Unit Tests

The test suite validates data integrity, zero-to-null replacement, Spark MLlib pipeline fitting, and metrics:

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 📁 Repository Structure

```
diabetes-cluster-computing/
├── README.md                          # Comprehensive documentation
├── requirements.txt                   # Project dependencies
├── diabetes.csv                       # Pima Indians Diabetes dataset
├── train_spark.py                     # CLI cluster training script
├── app.py                             # Interactive Streamlit Web Dashboard
├── .streamlit/
│   └── config.toml                    # Streamlit custom theme configuration
├── src/
│   ├── __init__.py
│   ├── spark_pipeline.py              # PySpark MLlib distributed pipeline
│   ├── sklearn_pipeline.py            # Comparative Scikit-Learn baseline
│   ├── cluster_benchmarks.py          # Partition scalability benchmark suite
│   └── utils.py                       # Plotly visualization and plotting helpers
└── tests/
    └── test_pipeline.py               # Automated pytest validation suite
```
