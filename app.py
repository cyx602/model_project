from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# 加载所有模型
models = {'iris': joblib.load('iris_model.pkl'), 'cancer': joblib.load('cancer_model.pkl')}
scalers = {'iris': joblib.load('iris_scaler.pkl'), 'cancer': joblib.load('cancer_scaler.pkl')}
accs = {'iris': joblib.load('iris_accuracy.pkl'), 'cancer': joblib.load('cancer_accuracy.pkl')}

# 定义特征范围 (Min, Max)
RANGES = {
    'iris': [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)],
    'cancer': [(6.98, 28.1), (9.71, 39.2), (43.7, 188.5), (143.5, 2501.0)]
}

@app.route('/')
def home():
    return render_template('index.html',
                           iris_acc=f"{accs['iris']*100:.1f}%",
                           cancer_acc=f"{accs['cancer']*100:.1f}%")

def process_prediction(model_type, form, labels):
    try:
        features = []
        for i in range(1, 5):
            val = float(form.get(f'f{i}'))
            # 后端双重校验范围
            min_val, max_val = RANGES[model_type][i-1]
            if val < min_val or val > max_val:
                return f"Error: Input {val} is outside valid range ({min_val}-{max_val})"
            features.append(val)

        # 标准化并预测
        final = np.array(features).reshape(1, -1)
        scaled = scalers[model_type].transform(final)
        pred = models[model_type].predict(scaled)[0]
        return f"Result: {labels[pred]}"
    except:
        return "Error: Invalid format"

@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    res = process_prediction('iris', request.form, ["Setosa", "Versicolour", "Virginica"])
    return render_template('index.html', iris_text=res, iris_acc=f"{accs['iris']*100:.1f}%", cancer_acc=f"{accs['cancer']*100:.1f}%")

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    res = process_prediction('cancer', request.form, ["Malignant", "Benign"])
    return render_template('index.html', cancer_text=res, iris_acc=f"{accs['iris']*100:.1f}%", cancer_acc=f"{accs['cancer']*100:.1f}%")

if __name__ == '__main__':
    app.run(debug=True)