# 📊 Customer Churn Prediction in the Telecom Industry

An end-to-end **Machine Learning framework** for predicting customer churn in the **telecommunications industry**.  
This project combines **predictive modeling, explainable AI (SHAP), and business-driven profit optimization**
to deliver both **technical accuracy** and **real-world impact**.

---

## 🔍 Problem Statement

Customer churn is a critical challenge for telecom companies, leading to substantial revenue loss.  
Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project focuses on:
- Early identification of potential churners  
- Understanding *why* customers churn  
- Maximizing business profit through optimized decision thresholds  

---

## 📂 Datasets Used

| Dataset Name | Records |
|-------------|---------|
| IBM Telco Customer Churn | 7043 |
| Churn-in-Telecom | 3333 |
| UCI Churn Dataset | 5000 |

---

## ⚙️ Methodology / Pipeline

1. **Data Loading & Exploration**  
2. **Data Preprocessing**  
   - Missing value treatment  
   - Encoding categorical variables  
   - Feature scaling and engineering  

3. **Class Imbalance Handling**  
   - SMOTE  
   - SMOTEENN  
   - SMOTETomek  

4. **Model Training**  
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting  

5. **Model Evaluation**  
   - Accuracy  
   - ROC-AUC  
   - F1-Score  
   - Matthews Correlation Coefficient (MCC)  
   - Confusion Matrix  

6. **Explainability & Business Impact**  
   - SHAP global and local explanations  
   - Cost-Benefit Analysis  
   - Profit-based threshold optimization  

7. **Deployment**  
   - Gradio web application hosted on Hugging Face  

---

## 🧠 Model Performance Highlights

- Accuracy achieved: **82% – 86%**  
- ROC-AUC up to **0.91**  
- Random Forest + SMOTETomek delivered the **highest business ROI (>550%)**  
- Consistent churn drivers across datasets:
  - Tenure  
  - Contract type  
  - Monthly charges  

---

## 🔎 Explainable AI (SHAP)

To overcome the **black-box nature** of ensemble models, SHAP is used:
- **Beeswarm plots** → Global feature importance  
- **Bar plots** → Ranked average feature impact  
- **Waterfall plots** → Individual customer-level explanations  

This enables transparent, trustworthy, and actionable predictions.

---

## 💰 Business-Driven Optimization

Instead of using a fixed probability threshold (0.5):
- Profit-based thresholds are optimized  
- Net profit and ROI are maximized  
- Missed churn losses are minimized  

This aligns machine learning outcomes with **business objectives**.

---

## 🚀 Deployment

- Built an interactive **Gradio application**
- Accepts customer attributes as input  
- Outputs:
  - Churn probability  
  - Risk classification  
  - SHAP explanation  
  - Estimated profit and ROI  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - NumPy, Pandas  
  - Scikit-learn  
  - Imbalanced-learn  
  - SHAP  
  - Matplotlib, Seaborn  
  - Gradio  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app/app.py
