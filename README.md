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

- The proposed churn prediction framework is evaluated on three telecom datasets: UCI Churn, Churn-in-Telecom, and IBM Telco  
- Best-performing models differ across datasets based on data characteristics and imbalance handling strategies  

- **UCI Churn Dataset**
  - Random Forest with SMOTE achieves the highest performance
  - Accuracy: **97.9%**, ROC-AUC: **0.915**, MCC: **0.889**
  - Maximum profit: **$65,580**, ROI: **896.8%**

- **Churn-in-Telecom Dataset**
  - Gradient Boosting with SMOTE performs best
  - Accuracy: **96.7%**, ROC-AUC: **0.886**, MCC: **0.757**
  - Maximum profit: **$43,060**, ROI: **892.9%**

- **IBM Telco Dataset**
  - Random Forest with SMOTETomek yields the highest business impact
  - Accuracy: **76.7%**, ROC-AUC: **0.839**, MCC: **0.511**
  - Maximum profit: **$158,880**, ROI: **551.5%**

- SHAP-based explainability consistently identifies **tenure, contract type, and monthly charges** as key churn-driving factors across datasets

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

## 🌐 Live Demo

The deployed Gradio application is available on Hugging Face Spaces:

🔗 **Live App:** https://huggingface.co/spaces/DheivaCodes/CHURN_PREDICTION

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
