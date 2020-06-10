import pandas as pd
df = pd.read_csv('C:/Users/vaibhav/Downloads/Deployment/Advertising.csv', index_col=0)

x = df.drop('sales', axis=1)
y = df['sales']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=20)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)

import pickle
pickle.dump(linreg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
