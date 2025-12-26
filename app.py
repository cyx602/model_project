from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# 加载所有模型
models = {'iris': joblib.load('iris_model.pkl'), 'cancer': joblib.load('cancer_model.pkl')}
scalers = {'iris': joblib.load('iris_scaler.pkl'), 'cancer': joblib.load('cancer_scaler.pkl')}
accs = {'iris': joblib.load('iris_accuracy.pkl'), 'cancer': joblib.load('cancer_accuracy.pkl')}


@app.route('/')
def home():
    # 首页同时传递两个模型的准确率
    return render_template('index.html',
                           iris_acc=f"{accs['iris'] * 100:.1f}%",
                           cancer_acc=f"{accs['cancer'] * 100:.1f}%")


def process_prediction(model_type, form, labels):
    try:
        # 获取 4 个输入值
        features = [form.get(f'f{i}') for i in range(1, 5)]
        if not all(features): return "Error: Missing input"

        # 标准化并预测
        final = np.array([float(x) for x in features]).reshape(1, -1)
        scaled = scalers[model_type].transform(final)
        pred = models[model_type].predict(scaled)[0]
        return f"Result: {labels[pred]}"
    except:
        return "Error: Invalid input"


@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    res = process_prediction('iris', request.form, ["Setosa", "Versicolour", "Virginica"])
    return render_template('index.html', iris_text=res,
                           iris_acc=f"{accs['iris'] * 100:.1f}%",
                           cancer_acc=f"{accs['cancer'] * 100:.1f}%")


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    res = process_prediction('cancer', request.form, ["Malignant", "Benign"])
    return render_template('index.html', cancer_text=res,
                           iris_acc=f"{accs['iris'] * 100:.1f}%",
                           cancer_acc=f"{accs['cancer'] * 100:.1f}%")


if __name__ == '__main__':
    app.run(debug=True)