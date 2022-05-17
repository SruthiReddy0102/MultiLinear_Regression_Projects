#Input Variables (x) = Other Variables
#Output Variable(y) = Sales Price

import pandas as pd
import scipy 
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Importing Dataset
Toyota = pd.read_csv("C:/Users/personal/Desktop/ToyotaCorolla.csv", encoding= 'unicode_escape')
Toyota

# Removing of unnecessary columns
Toyota1 = Toyota.drop(columns = "Id")
Toyota1

#Exploratory Data Analysis

Toyota1.describe()


import matplotlib.pyplot as plt

Toyota1.info()

# Checking the NA values and Count of the variables

cat_Toyota1 = Toyota.select_dtypes(include = ['object']).copy()
cat_Toyota1
print(cat_Toyota1.isnull().values.sum()) 

print(cat_Toyota1['Model'].value_counts())
print(cat_Toyota1['Fuel_Type'].value_counts())
print(cat_Toyota1['Color'].value_counts())

# Plot Representation of the String Variables

import seaborn as sns

# Model
Model_count = cat_Toyota1.Model.value_counts()
sns.set(style = "darkgrid")
sns.barplot(Model_count.index,Model_count.values,alpha = 0.9)
plt.show()

# Fuel_Type
Fuel_Type_count = cat_Toyota1.Fuel_Type.value_counts()
sns.set(style = "dark")
sns.barplot(Fuel_Type_count.index,Fuel_Type_count.values,alpha = 0.9)
plt.show()

# Color
Color_count = cat_Toyota1.Color.value_counts()
sns.set(style = "dark")
sns.barplot(Color_count.index,Color_count.values,alpha = 0.9)
plt.show()

# # Creation of Dummy Variabels
cat_Toyota1_onehot_sklearn = cat_Toyota1.copy()
cat_Toyota1_onehot_sklearn

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb_results1 = lb.fit_transform(cat_Toyota1_onehot_sklearn['Fuel_Type'])
lb_results1_df = pd.DataFrame(lb_results1, columns=lb.classes_)

print(lb_results1_df.head())

lb_results2 = lb.fit_transform(cat_Toyota1_onehot_sklearn['Color'])
lb_results2_df = pd.DataFrame(lb_results2, columns=lb.classes_)

print(lb_results2_df.head())

# concate the dummy variable to the data sheet

Toyota1_df = pd.concat([Toyota1,lb_results1_df,lb_results2_df], axis=1)
Toyota1_df
Toyota1_df = Toyota1_df.drop(['Model','Fuel_Type','Color','Radio','Radio_cassette','CD_Player','Cylinders'], axis=1)
Toyota1_df

# Scatter plot
sns.pairplot(Toyota1_df.iloc[:, :])
                             
# Correlation matrix
Toyota1_df.corr()

# The output shows there are many insignificant variables

y = Toyota1_df.iloc[:,0]
y
x = Toyota1_df.iloc[: , 1 :]
x

# MODEL BUILDING
import statsmodels.formula.api as sm

model = sm.ols('Toyota1_df.iloc[:,0] ~ Toyota1_df.iloc[: , 1 :]', data = Toyota1_df).fit()
model.summary()

# r square value is 0.975
# Fuel(0.156) , Color and multi are insignificant

# Finding the influence values

import statsmodels.api as smf

smf.graphics.influence_plot(model)


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])] 
vif["Features"] = x.columns

vif.round(1)

# Final Model
y = Toyota1_df.iloc[:, 0]
X = Toyota1_df.iloc[ : , 1 :]

model1 = sm.ols('y ~ X', data = Toyota1_df).fit()
model1.summary()

# this is the best model to build for train and test 

# Prediction
pred = model1.predict(Toyota1_df)

# Q-Q plot
res = model1.resid
smf.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Toyota1_df.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

smf.graphics.influence_plot(model1)




### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Toyota1_train, Toyota1_test = train_test_split(Toyota1_df, test_size = 0.2) # 20% test data
Toyota1_train.columns


# preparing the model on train data 
model_train = sm.ols('Price ~ Age_08_04+Mfg_Month+Mfg_Year+KM+HP+Met_Color+Automatic+cc+Doors+Gears+Quarterly_Tax+Weight+Mfr_Guarantee+BOVAG_Guarantee+Guarantee_Period+ABS+Airbag_1+Airbag_2+Airco+Automatic_airco+Boardcomputer+Central_Lock+Powered_Windows+Power_Steering+Mistlamps+Sport_Model+Backseat_Divider+Metallic_Rim+Tow_Bar+CNG+Diesel+Petrol+Beige+Black+Blue+Green+Grey+Red+Silver+Violet+White+Yellow', data = Toyota1_train).fit()

# prediction on test data set 
test_pred = model_train.predict(Toyota1_test)

# test residual values 
test_resid = test_pred - Toyota1_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 3230.05


# train_data prediction
train_pred = model_train.predict(Toyota1_train)

# train residual values 
train_resid  = train_pred - Toyota1_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse # 1052.59
 


#  Right Fit 
 
