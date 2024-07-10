'''Create a predictive model using Gradient Boosting to forecast housing prices based on various features such as square footage, number of bedrooms, number of bathrooms, and location.'''
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#step no 2
data = {'SquareFootage': [1500,2000,1200,1800,1350],'Bedrooms': [3,4,2,3,3],'Bathrooms': [2,2.5,1.5,2,2],'Location': ['Suburb','City','Rural','City','Suburb'],'Price': [250000,300000,180000,280000,220000]}

df = pd.DataFrame(data)
# print(df)

# converting the location column to dummy
df = pd.get_dummies(df,columns=['Location'])


X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(mse)

# user input
print("Enter the details for house prediction:")
sq_footage=float(input("Square Footage: "))
bedrooms = int(input("Number of bedrooms: "))
bathrooms = float(input("Numbers of Bathrooms: "))
location = input("Location (Suburb/City/Rural): ")

input_location = [0,0,0]
if location == 'Suburb':
    input_location[0]=1
elif location == 'City':
    input_location[1]=1
elif location == 'Rural':
    input_location[2]=1
    
user_input = pd.DataFrame({'SquareFootage': [sq_footage],'Bedrooms': [bedrooms],'Bathrooms': [bathrooms],'Location_City': [input_location[1]],'Location_Rural': [input_location[2]],'Location_Suburb': [input_location[0]]})

# make the prediction
predicted_price = model.predict(user_input)
print(f"Predicted Price for the House : {predicted_price[0]}")