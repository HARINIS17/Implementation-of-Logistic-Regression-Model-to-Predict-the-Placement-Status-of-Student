# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries.

2.Load the dataset into a DataFrame.

3.Create a copy of the original dataset.

4.Drop unnecessary columns (like sl_no, salary).

5.Encode all categorical columns using LabelEncoder.

6.Split the data into input features (X) and target variable (Y).

7.Divide the data into training and testing sets.

8.Initialize and train the Logistic Regression model.

9.Predict the placement status using the test data.

10.Evaluate the model using accuracy and other metrics.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: HARINI S

RegisterNumber:212224230083

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report

data=pd.read_csv("/content/Placement_Data (1).csv")

data.head()

data1=data.loc[:,~data.columns.isin(["sl_no","salary"])]

data1.head()

data1.isnull()

data1.duplicated().sum()

le=LabelEncoder()

cols=["gender","ssc_b","hsc_b","degree_t","workex","specialisation","status"]

data1[cols]=data1[cols].apply(lambda col:le.fit_transform(col))

data1

x=data1.iloc[:,:-1]

x

y=data1["status"]

y

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

x_encoded = pd.get_dummies(x)

x_tr, x_te, y_tr, y_te = train_test_split(x_encoded, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")

lr.fit(x_tr, y_tr)

predictions = lr.predict(x_te)

print(predictions)

accuracy=accuracy_score(y_te,predictions)

print(accuracy)

classification_report1=classification_report(y_te,predictions)

print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:

![Screenshot 2025-04-24 085753](https://github.com/user-attachments/assets/8fa91920-c1e4-49ff-8e6d-7726c43043ef)

![Screenshot 2025-04-24 085813](https://github.com/user-attachments/assets/eafb8fb3-32f7-41f7-beaf-4826d8b60ac0)

![Screenshot 2025-04-24 085827](https://github.com/user-attachments/assets/c65f0e75-22a8-495f-8826-68796dd3997b)

![Screenshot 2025-04-24 085843](https://github.com/user-attachments/assets/3989fe68-72af-454f-83fc-d1a822196932)

![Screenshot 2025-04-24 085901](https://github.com/user-attachments/assets/0590fc3e-921d-493e-bf68-35bc633213eb)

![Screenshot 2025-04-24 085914](https://github.com/user-attachments/assets/6005b2c8-7812-4ad5-b22a-7596c7cde1a0)

![Screenshot 2025-04-24 085926](https://github.com/user-attachments/assets/8e96ae62-10a8-45bb-898d-77984957cb08)

![Screenshot 2025-04-24 085940](https://github.com/user-attachments/assets/0115bf62-a1a0-4164-9016-491e611456db)

![Screenshot 2025-04-24 095731](https://github.com/user-attachments/assets/70878ce9-2fcf-40c7-a15d-f4ca3a0ca15f)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
