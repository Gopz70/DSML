import sys

sys.path.append(r"C:\Users\mgopi\AppData\Roaming\Python\Python312\site-packages")

import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.neighbors import KNeighborsClassifier

#Dataset

data = {
    'Country':['France','Spain','Germany','Spain','Germany',
               'France','Spain','France','Germany','France'],
    'Age':[44,27,30,38,40,35,35,48,50,37],
    'Salary':[72000,48000,54000,61000,61000,
              58000,52000,79000,83000,67000],
    'Purchased':['No','Yes','No','No','Yes','Yes'
                 ,'No','Yes','No','Yes']
    }

df = pd.DataFrame(data)

le = LabelEncoder()

df['Country'] = le.fit_transform(df['Country'])
df['Purchased'] = le.fit_transform(df['Purchased'])

x = df[['Country','Age','Salary']]
y = df['Purchased']


scaler = StandardScaler()
x = scaler.fit_transform(x)

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x,y)

new_person = pd.DataFrame([[0, 36, 60000]],
                          columns=['Country', 'Age', 'Salary'])

new_person = scaler.transform(new_person)
prediction = model.predict(new_person)

print("Prediction:",prediction)
