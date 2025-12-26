# 🧠 Stroke Prediction Using Machine Learning

This project focuses on predicting the **risk of stroke** using machine learning techniques. By analyzing patient health data, the system aims to identify whether an individual is likely to experience a stroke based on various medical and demographic features.

The project is implemented in **Python** using a Jupyter Notebook and is intended for educational and exploratory purposes in data science and machine learning.

---

## 📌 Project Objectives

The main objectives of this project are to:

* Analyze a healthcare dataset related to stroke risk
* Perform data preprocessing and exploratory data analysis (EDA)
* Train machine learning models to predict stroke occurrence
* Evaluate model performance using standard metrics
* Gain insights into which factors contribute most to stroke risk

---

## 🧠 Dataset Description

The dataset typically includes the following features:

* **Gender**
* **Age**
* **Hypertension**
* **Heart Disease**
* **Marital Status**
* **Work Type**
* **Residence Type**
* **Average Glucose Level**
* **Body Mass Index (BMI)**
* **Smoking Status**
* **Stroke** (target variable)

> The target variable indicates whether the patient has had a stroke (1) or not (0).

---

## ⚙️ Workflow

1. Load and inspect the dataset
2. Handle missing values and categorical variables
3. Perform exploratory data analysis (EDA)
4. Encode categorical features and scale numerical values
5. Split data into training and test sets
6. Train machine learning models
7. Evaluate and compare model performance

---

## 🛠 Technologies & Libraries

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical computations
* **Matplotlib / Seaborn** – Data visualization
* **Scikit-learn** – Machine learning models and evaluation

---

## 🤖 Models Used

Depending on the implementation, the project may include:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)

Models are evaluated and compared to determine the most effective approach for stroke prediction.

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F4-Score
* Confusion Matrix

These metrics help assess how well the model predicts stroke cases, especially in imbalanced datasets.

---

## 📂 Project Structure

```
stroke-prediction/
├── StrokePrediction.ipynb
├── dataset/
│   └── healthcare-dataset.csv
├── README.md
```

> File and folder names may vary depending on implementation.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/seymencanaydogan/stroke-prediction
   ```

2. Navigate to the project directory and install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open and run `StrokePrediction.ipynb`.

---

## 📈 Results & Insights

* The trained models demonstrate the feasibility of predicting stroke risk using health-related data.
* Age, average glucose level, BMI, and pre-existing conditions (hypertension, heart disease) are among the most influential features.
* The project highlights the importance of proper preprocessing and evaluation when working with medical datasets.

---

## ⚠️ Disclaimer

This project is **not intended for medical diagnosis**. It is strictly for educational and research purposes. Real-world medical decisions should always be made by qualified healthcare professionals.

---

## 👤 Author

**Seymen Can Aydoğan**
