import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = {
 'Study_Hours': [2, 3, 4, 5, 6, 7, 1, 3, 6, 8],
 'Attendance': [60, 65, 70, 75, 80, 85, 55, 68, 78, 90],
 'Result': ['Fail', 'Fail', 'Pass', 'Pass', 'Pass',
 'Pass', 'Fail', 'Fail', 'Pass', 'Pass']
}
df = pd.DataFrame(data)
X = df[['Study_Hours', 'Attendance']]
y = df['Result']
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)
plot_tree(model, feature_names=['Study Hours', 'Attendance'],
 class_names=['Fail', 'Pass'], filled=True)
plt.show()
print("Prediction:", model.predict([[5, 76]])[0])