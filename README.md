# 📱 SMS Spam Classifier | NLP Project  
**Author:** Nitish Raj Vinnakota | [LinkedIn](https://linkedin.com/in/vnr-nitish)

---

## 🔍 Project Overview

This project is an end-to-end **Natural Language Processing (NLP)** pipeline to classify SMS messages as either **Spam** or **Ham (Not Spam)**. It demonstrates how to preprocess text, extract features using **TF-IDF**, apply multiple classification models, and evaluate model performance using real-world data.

It was developed as part of my **Capstone Project** during a Data Science internship at **Teachnook**.

---

## 🎯 Objective

To build a robust SMS classifier that automatically filters out spam messages using classic machine learning algorithms on textual data.

---

## 📁 Dataset Info

- **Source:** UCI Machine Learning Repository (SMS Spam Collection)
- **Instances:** 5,572 SMS messages
- **Columns:**
  - `v1`: Label (spam / ham)
  - `v2`: Message text

---

## 🧠 Machine Learning Workflow

### ✅ Data Preprocessing:
- Dropped irrelevant columns
- Renamed columns for clarity (`v1 → label`, `v2 → message`)
- Checked for missing/null values

### ✅ Text Vectorization:
- Used **TF-IDF Vectorizer** to convert text into numerical features
- Explored `CountVectorizer` vs. `TfidfVectorizer`

### ✅ Model Building:
- Trained and tested:
  - **Gaussian Naive Bayes**
  - **K-Nearest Neighbors (KNN)**
- Split dataset using `train_test_split` (80/20)

### ✅ Evaluation:
- Used:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
- Visualized confusion matrices using `ConfusionMatrixDisplay`

---

## 📊 Key Results

| Model                 | Accuracy |
|----------------------|----------|
| Gaussian Naive Bayes | ~87%     |
| KNN Classifier        | ~88%     |

- **KNN outperformed Naive Bayes** slightly in accuracy, but Naive Bayes remains more scalable and interpretable for NLP tasks.
- The model correctly identifies spam with **high recall**, minimizing false negatives.

---

## 🧰 Tools & Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Seaborn**, **Matplotlib**
- **scikit-learn**
- **TF-IDF Vectorization**
- **Jupyter Notebook**

---

## 💡 Highlights

- ✅ Implemented NLP preprocessing from scratch
- ✅ Vectorized raw text using TF-IDF
- ✅ Compared model performances using industry-standard metrics
- ✅ Clean and modular ML pipeline for reproducibility

---

## 🚀 Future Work

- Integrate advanced models like **SVM**, **Logistic Regression**, or **XGBoost**
- Add **Grid Search CV** or **RandomizedSearchCV** for hyperparameter tuning
- Explore **deployment options** using Streamlit or Flask
- Incorporate **word embeddings** for semantic text understanding

---

## 📫 Contact

📧 **Email:** nvinnako2@gitam.in  
🔗 **LinkedIn:** [linkedin.com/in/vnr-nitish](https://linkedin.com/in/vnr-nitish)

---

> “Built to detect spam. Powered by text, logic, and machine learning.”  
