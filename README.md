# Multi-Model Machine Learning Web System (README)

## 1. Project Overview
This project is an integrated machine learning web platform that implements two core classification tasks: **Iris Flower Species Recognition** and **Breast Cancer Nature Prediction**.
The project demonstrates the complete engineering workflow from data preprocessing, independent model training to web deployment using **Flask**.

* **Online Demo URL**: https://cyx-model-project.onrender.com
* **Local Access Address**: http://127.0.0.1:5000/

---

## 2. Requirement Checklist
This project fully meets all the experimental requirements for both the Iris project and Breast Cancer project as specified in the documentation:

| Task Phase | Implementation Description |
| **1. Environment Configuration** | Installed `flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`, and `gunicorn`. |
| **2. Data Loading** | Used built-in datasets in `train_iris.py` and `train_cancer.py` respectively. |
| **3. Model Training** | Adopted **Random Forest** algorithm with `test_size=0.3`. |
| **4. Feature Scaling** | Introduced `StandardScaler` for data standardization to ensure scientific model evaluation. |
| **5. Model Persistence** | Used `joblib` to independently save models, scalers, and accuracy data as `.pkl` files. |
| **6. Web Application** | Built with Flask, supporting routing to `/predict_iris` and `/predict_cancer`. |
| **7. Interactive Interface** | Includes two independent forms for feature input and real-time display of prediction results. |
| **8. Deployment** | Successfully deployed on the Render platform. |

---

## 3. Setup and Local Run

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
2. **Train Models Individually and Generate Model Files**:
```bash
python train_iris.py
python train_cancer.py
```
3. **Start Flask Application**:
```bash
python app.py
After starting, access http://127.0.0.1:5000/ in your browser.
```

## 4. Project Structure
.
├── app.py                 
├── train_iris.py           
├── train_cancer.py        
├── iris_model.pkl         
├── cancer_model.pkl        
├── iris_scaler.pkl       
├── cancer_scaler.pkl       
├── iris_accuracy.pkl       
├── cancer_accuracy.pkl    
├── requirements.txt      
├── Procfile              
├── runtime.txt            
├── static/
│   ├── style.css           
│   └── scripts.js         
└── templates/
    └── index.html          
