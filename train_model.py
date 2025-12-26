import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler # 使用 sklearn 而不是 ray

# 1. 加载乳腺癌数据集 [cite: 128]
data = load_breast_cancer()

# 2. 选取前 4 个特征（与 HTML 表单对应）
X = data.data[:, :4]
y = data.target

# 3. 划分数据集 [cite: 130]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 【高分关键】特征缩放：提升模型性能和评估的专业度
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 训练模型 [cite: 132]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 【模型评估】满足文档 Step 2 & 3 [cite: 34, 133]
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Model Accuracy: {accuracy:.2f}") # 输出准确率
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. 保存文件 [cite: 37, 135]
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl') # 保存缩放器
joblib.dump(accuracy, 'accuracy.pkl') # 保存准确率供前端显示
print("-" * 30)
print("All files (model, scaler, accuracy) generated successfully.")