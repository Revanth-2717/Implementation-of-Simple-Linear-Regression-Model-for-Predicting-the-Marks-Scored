# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.start

step 2.Import the standard Libraries.

step 3.Set variables for assigning dataset values.

step 4.Import linear regression from sklearn.

step 5.Assign the points for representing in the graph.

step 6.Predict the regression for marks by using the representation of the graph.

step 7.Compare the graphs and hence we obtained the linear regression for the given datas.

step 8.stop 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: REVANTH.P

RegisterNumber: 212223040143
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

DATASET:

![DATASET](https://github.com/user-attachments/assets/c74e5516-83bc-4f84-aefe-6169954a2dfe)

Hard Values:

![Screenshot 2025-03-10 102101](https://github.com/user-attachments/assets/6900fb4a-83a8-46c2-bd02-10bf877265cb)

Tail Values:

![tail](https://github.com/user-attachments/assets/f130726b-3c3a-4279-8519-a54d35100bcc)

Xand Y Values:

![x and y values](https://github.com/user-attachments/assets/3dfda76d-87f6-4f91-a056-e3909078180c)

Prediction of X and Y:

![Screenshot 2025-03-10 102449](https://github.com/user-attachments/assets/ff829f1f-3078-4527-bd94-4dfcebf9ae62)

MSE, MAE and RMSE:

![Screenshot 2025-03-10 102520](https://github.com/user-attachments/assets/0c88b5d7-469a-4aff-8ac4-c6810ffeb4a5)

Training Set:

![Screenshot 2025-03-10 102541](https://github.com/user-attachments/assets/221527cf-0404-454e-823b-ca41d36f0357)


![Screenshot 2025-03-10 102600](https://github.com/user-attachments/assets/9413b070-2d0d-4f44-af92-04097ed3dc47)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
