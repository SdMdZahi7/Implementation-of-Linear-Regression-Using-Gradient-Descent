# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
~~~
1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function.
3.Declare the default values for linear regression.
4.Calculate the loss usinng Mean Square Error.
5.Predict the value of y.
6.Plot the graph respect to hours and scores using scatter plot function.
~~~
## Program:
~~~
Program to implement the linear regression using gradient descent.
Developed by: SYED MUHAMMED ZAHI
RegisterNumber: 212221230114
~~~
~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h - y)**2

  return 1/(2*m) * np.sum(square_err)     #returning

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
 
computeCost(X,y,theta)  # Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)  ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="g")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
~~~
## Output:
![image](https://user-images.githubusercontent.com/94187572/204815806-ac137ad2-95f8-4371-a5c9-87d942767924.png)
![image](https://user-images.githubusercontent.com/94187572/204815839-f9de0ea1-0754-4ee1-a6a4-5c3b96b79510.png)
![image](https://user-images.githubusercontent.com/94187572/204815870-2954716c-8e16-4fb3-b864-591b3df42c31.png)
![image](https://user-images.githubusercontent.com/94187572/204815978-8c7e5d1d-e4be-4340-83ec-0ed9b7f67c8d.png)
![image](https://user-images.githubusercontent.com/94187572/204816033-97ce0a4d-1f9f-440e-800d-b7dc5f2da0c0.png)
![image](https://user-images.githubusercontent.com/94187572/204816063-962630f9-fb0b-4d74-8098-0536319d8bdb.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

