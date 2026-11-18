import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
data = {
 'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 5],
 'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass',
 'Pass', 'Pass', 'Pass', 'Fail', 'Pass']
}
df = pd.DataFrame(data)
X = df[['Study_Hours']]
y = df['Result']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
predicted = model.predict(X)
print("Actual:", list(y))
print("Predicted:", list(predicted))
cm = confusion_matrix(y, predicted, labels=['Pass', 'Fail'])
print("Confusion Matrix:")
print(cm)
ConfusionMatrixDisplay(cm, display_labels=['Pass', 'Fail']).plot()
plt.title("Confusion Matrix")
plt.show()