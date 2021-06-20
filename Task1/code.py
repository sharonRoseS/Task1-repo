

# SHARON ROSE S

#Step 1: Importing the required libraries for the execution
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Step 2: Importing the DataSet to be worked on using pandas
df=pandas.read_csv('http://bit.ly/w-data')

#step 3: Viewing the DataSet
print("**View of DataSet**")
print("")
print(df.head(25))
print("")

#Step 4: Inspecting the Data(basic functions are performed)
print("**Description of data** ")
print("")
print(df.describe())
print("")

#Step 5: Checking / Droping rows having null values
df.dropna(axis=0, how="any", inplace=True)

#Step 6: Visualising data in graphical format
df.plot(x='Hours', y='Scores', style='+',color="red")  
plt.title('Graph of relation between hours studied and percentage score of a student')  
plt.xlabel('Hours studied')  
plt.ylabel('Percentage score')  
plt.show()


#Step 7: Divide the data into "attributes" (inputs) and "labels" (outputs)
x=df.iloc[:, :-1].values  
y=df.iloc[:, 1].values

#Step 8: Split this data into training and test sets
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=0)

#Step 9: Train the algoritm
regressor=LinearRegression()
regressor.fit(x_train, y_train) #Traning the linear regression model with the available dataset


#Step 10: Testing algorithm
y_predict=regressor.predict(x_test)

#Step 11: Visualising data in graphical format
plt.scatter(x_train, y_train, color="red", marker="*")
plt.plot( x_train,regressor.predict(x_train), color = "green")
plt.title('The Linear Regression Model')
plt.xlabel('Hours studied')  
plt.ylabel('Percentage score')  
plt.show()

#Step 12: Comparing acutal values with predicted values
df2=pandas.DataFrame({"predicted values": y_predict, "acutal values": y_test, "Difference": y_predict-y_test})
print(df2)
print("")

#Step 13: Evaluate the model
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print("")

#Step 14: Now Predict required data
print("Enter the required data to be predicted: ")
z=float(input())
StdScore=regressor.predict([[z]])
print("")
print("The predicted score if a student studies for 9.25 hrs/day is: ", StdScore)

