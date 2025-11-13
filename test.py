import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

# 你的数据
data ={'01': {'pred': [0, 2, 3], 'truth': [0, 2, 3]}, '02': {'pred': [2], 'truth': [2]}, '03': {'pred': [1, 2], 'truth': [1, 2]}, '04': {'pred': [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3]}, '05': {'pred': [0, 0, 0], 'truth': [0, 0, 2]}, '06': {'pred': [0], 'truth': [0]}, '07': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '08': {'pred': [0, 0, 0], 'truth': [0, 0, 2]}, '09': {'pred': [2], 'truth': [2]}, '10': {'pred': [0, 0], 'truth': [0, 0]}, '11': {'pred': [2, 2, 2, 2], 'truth': [0, 0, 0, 1]}, '12': {'pred': [0, 2, 0, 0, 1, 1, 0, 2, 1, 2], 'truth': [0, 0, 0, 0, 1, 1, 1, 2, 3, 3]}, '13': {'pred': [2, 2, 2, 3], 'truth': [0, 1, 2, 3]}, '14': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 2]}, '15': {'pred': [0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 2]}, '16': {'pred': [3, 3, 2, 1, 3], 'truth': [0, 1, 2, 3, 3]}, '17': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2]}, '18': {'pred': [0, 1, 2, 0, 2, 2], 'truth': [0, 0, 0, 0, 2, 2]}, '19': {'pred': [0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 2, 2]}, '20': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3]}, '21': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, '23': {'pred': [0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 2, 2, 2]}, '24': {'pred': [0, 0, 0, 0, 0, 1, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 1, 2]}, '25': {'pred': [2, 0, 2, 3], 'truth': [0, 0, 1, 3]}, '26': {'pred': [2, 2, 0, 2], 'truth': [0, 0, 0, 0]}, '27': {'pred': [0], 'truth': [0]}, '28': {'pred': [1, 1], 'truth': [1, 1]}, '29': {'pred': [3, 3, 3, 3, 3], 'truth': [3, 3, 3, 3, 3]}, '30': {'pred': [3, 3, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]}, '31': {'pred': [2, 2, 1, 2, 0, 2, 3, 0, 2, 0], 'truth': [0, 0, 1, 2, 2, 2, 3, 3, 3, 3]}}

    #模型最优{'01': {'pred': [0, 2, 3], 'truth': [0, 2, 3]}, '02': {'pred': [2], 'truth': [2]}, '03': {'pred': [1, 2], 'truth': [1, 2]}, '04': {'pred': [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3]}, '05': {'pred': [0, 0, 0], 'truth': [0, 0, 2]}, '06': {'pred': [0], 'truth': [0]}, '07': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '08': {'pred': [0, 0, 0], 'truth': [0, 0, 2]}, '09': {'pred': [2], 'truth': [2]}, '10': {'pred': [0, 0], 'truth': [0, 0]}, '11': {'pred': [2, 2, 2, 2], 'truth': [0, 0, 0, 1]}, '12': {'pred': [0, 2, 0, 0, 1, 1, 0, 2, 1, 2], 'truth': [0, 0, 0, 0, 1, 1, 1, 2, 3, 3]}, '13': {'pred': [2, 2, 2, 3], 'truth': [0, 1, 2, 3]}, '14': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 2]}, '15': {'pred': [0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 2]}, '16': {'pred': [3, 3, 2, 1, 3], 'truth': [0, 1, 2, 3, 3]}, '17': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2]}, '18': {'pred': [0, 1, 2, 0, 2, 2], 'truth': [0, 0, 0, 0, 2, 2]}, '19': {'pred': [0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 2, 2]}, '20': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3]}, '21': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, '23': {'pred': [0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 2, 2, 2]}, '24': {'pred': [0, 0, 0, 0, 0, 1, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 1, 2]}, '25': {'pred': [2, 0, 2, 3], 'truth': [0, 0, 1, 3]}, '26': {'pred': [2, 2, 0, 2], 'truth': [0, 0, 0, 0]}, '27': {'pred': [0], 'truth': [0]}, '28': {'pred': [1, 1], 'truth': [1, 1]}, '29': {'pred': [3, 3, 3, 3, 3], 'truth': [3, 3, 3, 3, 3]}, '30': {'pred': [3, 3, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]}, '31': {'pred': [2, 2, 1, 2, 0, 2, 3, 0, 2, 0], 'truth': [0, 0, 1, 2, 2, 2, 3, 3, 3, 3]}}



# 提取所有的预测和真实值
y_true = []
y_pred = []

for entry in data.values():
    y_true.extend(entry["truth"])
    y_pred.extend(entry["pred"])

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

# 计算指标
acc = accuracy_score(y_true, y_pred)  # Accuracy
uf1 = f1_score(y_true, y_pred, average="macro")  # UF1 (Unweighted F1-score)
uar = recall_score(y_true, y_pred, average="macro")  # UAR (Unweighted Average Recall)

# 打印指标
print(f"Accuracy (ACC): {acc:.4f}")
print(f"Unweighted F1-score (UF1): {uf1:.4f}")
print(f"Unweighted Average Recall (UAR): {uar:.4f}")

# 定义类别名称
class_names = ['Negative', 'Positive', 'Surprise', 'Others']

# 绘制混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Four-Class Classification")
plt.show()
