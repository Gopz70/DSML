import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
data = {
 'Weather': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy',
 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Rainy'],
 'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
 'No', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(data)
le_weather, le_play = LabelEncoder(), LabelEncoder()
X = le_weather.fit_transform(df['Weather']).reshape(-1, 1)
y = le_play.fit_transform(df['Play'])
model = CategoricalNB()
model.fit(X, y)
new_data = le_weather.transform(['Sunny']).reshape(-1, 1)
prediction = model.predict(new_data)
print("Prediction:", le_play.inverse_transform(prediction)[0])
