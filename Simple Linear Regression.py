import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
 'Marks': [35, 40, 50, 55, 65, 70, 75, 85]}
df = pd.DataFrame(data)
X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)
print("Predicted Marks:", model.predict([[6]])[0])
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.show()

''' 1. Import the required libraries.
2. Create a dataset containing Study Hours and Marks.
3. Separate input (Hours) and output (Marks).
4. Create and train the Linear Regression model.
5. Predict marks for 6 study hours.
6. Display the predicted marks and regression graph.'''