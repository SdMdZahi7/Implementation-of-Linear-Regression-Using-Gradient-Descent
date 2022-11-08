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
#import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Take in a np array X,y,theta and generate the cost function of using theta as parameter in a linear regression model
"""
def computeCost(X,y,theta):
    m=len(y) #length of the training data
    h=X.dot(theta) #hypothesis
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) #returning J

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) #call the function

"""
Take in np array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
return theta and the list of the cost of theta during each iteration
"""
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha*(1/m )*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Takes in numpy array of x and theta and return the predicted value of y based on theta
"""
def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
~~~
## Output:
![image](https://user-images.githubusercontent.com/94187572/200654786-79b7b1fd-a01e-48e2-b0ec-51cf9c96cb37.png)
![image](https://user-images.githubusercontent.com/94187572/200654819-d7612b18-27b8-443d-b399-1c31ff117e66.png)
![image](https://user-images.githubusercontent.com/94187572/200654844-588c0140-55eb-4930-980a-133e89107110.png)
![image](https://user-images.githubusercontent.com/94187572/200654895-d5f96d33-03a2-4463-968f-bb2e2dc5257e.png)
![image](https://user-images.githubusercontent.com/94187572/200654916-3603d8fc-d173-47d5-9bed-bd60d3ab9f76.png)
![image](https://user-images.githubusercontent.com/94187572/200654968-72094227-0b5d-45c4-96b6-d0f6e2d00bcb.png)
![image](https://user-images.githubusercontent.com/94187572/200655020-e0daf4cb-f423-48f1-80c3-9e7a5a9a05b6.png)
![image](https://user-images.githubusercontent.com/94187572/200655059-b3b4c1a1-75f2-454f-9ad6-2cbdb6133e56.png)
![image](https://user-images.githubusercontent.com/94187572/200655083-191f6a1f-d0d1-43dd-873f-2aa8c8ffde61.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
