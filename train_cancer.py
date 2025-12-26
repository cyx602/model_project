import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data[:, :4]
y = data.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 评估并保存为特定文件名
acc = accuracy_score(y_test, model.predict(X_test))
joblib.dump(model, 'cancer_model.pkl')
joblib.dump(scaler, 'cancer_scaler.pkl')
joblib.dump(acc, 'cancer_accuracy.pkl')

print(f"Cancer model trained successfully. Accuracy: {acc:.2f}")