from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型、缩放器和准确率指标
# 加载所有必要文件
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
accuracy = joblib.load('accuracy.pkl') # 加载保存的准确率

@app.route('/')
def home():
    # 将准确率传给模板 [cite: 50, 64]
    return render_template('index.html', model_accuracy=f"{accuracy*100:.1f}%")




@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_features = [request.form.get('f1'), request.form.get('f2'),
                        request.form.get('f3'), request.form.get('f4')]

        if not all(raw_features):
            return render_template('index.html', prediction_text="Error: Missing input",
                                   model_accuracy=f"{accuracy * 100:.1f}%")

        # 数据标准化处理
        final_features = np.array([float(x) for x in raw_features]).reshape(1, -1)
        scaled_features = scaler.transform(final_features)

        prediction = model.predict(scaled_features)
        res_label = "Benign (良性)" if prediction[0] == 1 else "Malignant (恶性)"

        # 预测后也要带上准确率，否则页面刷新后数据会消失
        return render_template('index.html',
                               prediction_text=f'Result: {res_label}',
                               model_accuracy=f"{accuracy * 100:.1f}%")
    except Exception:
        return render_template('index.html', prediction_text="Error: Invalid format",
                               model_accuracy=f"{accuracy * 100:.1f}%")

if __name__ == '__main__':
    app.run(debug=True)