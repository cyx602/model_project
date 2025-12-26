# Breast Cancer Prediction Web System

## 1. Project Objective
The objective of this project is to build and train a machine learning model 
using the **Breast Cancer Wisconsin dataset** and deploy it as a functional 
web application using **Flask**. This system allows users to input tumor 
features through a web form and receive an immediate prediction on whether 
the tumor is malignant or benign.

* **Live Demo**: https://cyx_cancer_project.com
* **Local Host**: http://127.0.0.1:5000/

---

## 2. Requirement Fulfillment Checklist
This project successfully implements all the steps outlined in the assignment requirements:

| Requirement Stage | Implementation Status |
| **1. Install Libraries** | Installed flask, scikit-learn, pandas, numpy, joblib, and gunicorn. |
| **2. Load Dataset** | Used the Breast Cancer Wisconsin dataset from sklearn.datasets. |
| **3. Train Model** | Trained a Random Forest Classifier to distinguish between tumors. |
| **4. Save Model** | Used joblib to persist the trained model and feature scaler. |
| **5. Flask Application** | Developed app.py to handle routing and input processing. |
| **6. Web Interface** | Created an HTML form in index.html for user inputs. |
| **7. Prediction Result** | Displays the final prediction (Malignant or Benign) on the webpage. |
| **8. Deployment** | Successfully deployed the application to the Render platform. |

---
## 3. Project Structure
```text
.
├── app.py              # Main Flask server
├── train_model.py      # Script for training and scaling
├── model.pkl           # Trained Random Forest model
├── scaler.pkl          # Saved StandardScaler
├── accuracy.pkl        # Saved accuracy metric
├── requirements.txt    # Project dependencies
├── Procfile            # Deployment instructions
├── runtime.txt         # Python version specification
├── static/
│   ├── style.css       # UI styling (Pink/Glassmorphism)
│   └── scripts.js      # Frontend validation script
└── templates/
    └── index.html      # Main HTML interface
       
To run this project locally, follow these steps:

Install dependencies:

Bash

pip install -r requirements.txt
Train the model:

Bash

python train_model.py
Start the Flask server:

Bash

python app.py