# 🩺 Breast Cancer Predictive Modeling

<p align="center">

**A Machine Learning project for breast cancer classification using Python and Scikit-learn**

</p>

---

## 📌 Project Overview

This project focuses on building and evaluating machine learning models for **breast cancer classification** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The objective is to use diagnostic features extracted from breast tumor samples to classify tumors into two categories:

* 🟢 **Benign**
* 🔴 **Malignant**

The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis, feature scaling, model training, model evaluation, and visualization of classification results.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze the breast cancer diagnostic dataset.
* Perform data preprocessing and preparation.
* Explore relationships between different diagnostic features.
* Train multiple machine learning classification models.
* Compare model performance using standard evaluation metrics.
* Analyze classification errors using a confusion matrix.
* Evaluate model discrimination using ROC-AUC.
* Visualize the model results.
* Build a foundation for an interactive machine learning dashboard.

---

## 🧠 Machine Learning Models

The project implements and evaluates the following classification algorithms:

### 1. Logistic Regression

A statistical classification algorithm used to model the probability of a binary outcome.

### 2. Decision Tree

A tree-based supervised learning algorithm that makes predictions using a sequence of feature-based decisions.

### 3. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve predictive performance and robustness.

---

## 🔬 Machine Learning Workflow

```text
                 Breast Cancer Dataset
                          │
                          ▼
                 Data Loading
                          │
                          ▼
                 Data Exploration
                          │
                          ▼
                Data Preprocessing
                          │
                          ▼
                 Feature Selection
                          │
                          ▼
                  Train/Test Split
                          │
                          ▼
                  Feature Scaling
                          │
                          ▼
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        Logistic       Decision     Random
        Regression       Tree       Forest
              │           │           │
              └───────────┼───────────┘
                          ▼
                  Model Evaluation
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Accuracy    Confusion     ROC-AUC
                       Matrix
                          │
                          ▼
                   Final Analysis
```

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, a widely used dataset for binary breast cancer classification.

The dataset contains diagnostic measurements derived from digitized images of breast mass samples.

The target classification consists of:

| Class         | Description         |
| ------------- | ------------------- |
| **Benign**    | Non-cancerous tumor |
| **Malignant** | Cancerous tumor     |

The dataset contains numerical diagnostic features related to characteristics such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

> The dataset is used for machine learning research and educational purposes. This project is not intended for medical diagnosis or clinical decision-making.

---

## 🔍 Exploratory Data Analysis

The project includes exploratory analysis to understand the dataset before model training.

The analysis includes:

* Dataset structure
* Data types
* Statistical summaries
* Class distribution
* Feature distributions
* Feature relationships
* Correlation analysis
* Data visualization

Visualizations are created using **Matplotlib** and **Seaborn**.

---

## ⚙️ Data Preprocessing

The preprocessing workflow includes:

1. Loading the dataset
2. Inspecting the dataset
3. Checking data types
4. Checking for missing values
5. Separating features and target
6. Splitting the dataset into training and testing sets
7. Scaling numerical features where required
8. Preparing the data for machine learning models

---

## 📈 Model Evaluation

The models are evaluated using several classification metrics.

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many observations predicted as positive are actually positive.

### Recall

Measures how many actual positive observations are correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

---

## 📊 Evaluation Visualizations

The project includes visual analysis such as:

* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC
* Model performance comparison
* Feature correlation visualization

These visualizations help understand not only whether the models make accurate predictions, but also how they make classification errors.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Interactive Dashboard

* Streamlit

### Development Environment

* Google Colab
* Jupyter Notebook

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
breast-cancer-predictive-modeling/
│
├── 📓 Breast_Cancer_Wisconsin.ipynb
│
├── 📄 README.md
│
├── 📄 requirements.txt
│
├── 🚀 app.py
│
├── 📁 screenshots/
    ├── dashboard.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── model_comparison.png

```

> If a file or folder has not yet been uploaded to the repository, remove it from this section until it exists.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Saheli04/breast-cancer-predictive-modeling.git
```

### 2. Open the Project

```bash
cd breast-cancer-predictive-modeling
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

Open:

```text
Breast_Cancer_Wisconsin.ipynb
```

using Jupyter Notebook or Google Colab.

---

## 💻 Streamlit Dashboard

If the Streamlit application is included in the repository, it can be launched using:

```bash
streamlit run app.py
```

The dashboard provides an interactive interface for exploring the machine learning workflow and model results.

---

## 🚀 Open in Google Colab

You can open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saheli04/breast-cancer-predictive-modeling/blob/main/Breast_Cancer_Wisconsin.ipynb)

---

## 📸 Project Preview

Add screenshots of your dashboard and analysis here.

### 🖥️ Dashboard

```text
screenshots/dashboard.png
```

### 📊 Confusion Matrix

```text
screenshots/confusion_matrix.png
```

### 📈 ROC Curve

```text
screenshots/roc_curve.png
```

### 📋 Model Comparison

```text
screenshots/model_comparison.png
```

---

## 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering and scaling
* Supervised machine learning
* Binary classification
* Model comparison
* Performance evaluation
* Confusion matrix analysis
* ROC-AUC analysis
* Data visualization
* Streamlit dashboard development
* Organizing machine learning projects using GitHub

---

## 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Feature selection optimization
* Additional classification algorithms
* Model explainability using SHAP
* Improved Streamlit interface
* Interactive prediction functionality
* Model deployment
* Automated model evaluation
* Experiment tracking

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

The machine learning models demonstrated here should **not be used as a substitute for professional medical diagnosis or clinical decision-making**.

---

## 👩‍💻 Author

### Saheli Debnath

🎓 Data Science / Computer Science Student

**Interests:**

* Data Analytics
* Machine Learning
* Deep Learning
* Computer Vision
* Data Visualization
* Artificial Intelligence

### 🔗 GitHub

[Saheli04](https://github.com/Saheli04)

---

## ⭐ If You Find This Project Interesting

Feel free to explore the repository, review the notebook, and learn from the implementation.
