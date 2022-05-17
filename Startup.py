# Output Variable(y) = Profit
# Input Variables (x) = R.D Spend, Administration, Marketing Spend, State.


import pandas as pd
import scipy 
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Import Dataset

Startup = pd.read_csv("C:/Users/personal/Desktop/50_Startups.csv")

Startup.columns = "RD","Adm","MS","State","Profit"

#Exploratory Data Analysis

Startup.describe()

import matplotlib.pyplot as plt

# Graphical Representation of Box plot and Histogram

#RD

plt.bar(height = Startup.RD , x = np.arange(1,51,1))
plt.hist(Startup.RD)
plt.boxplot(Startup.RD)

#MS

plt.bar(height = Startup.MS, x = np.arange(1,51,1))
plt.hist(Startup.MS)
plt.boxplot(Startup.MS)

# Pair Plots

import seaborn as sns
sns.jointplot(x = Startup.RD ,y = Startup.Profit)
sns.jointplot(x = Startup.Adm, y = Startup.Profit)
sns.jointplot(x = Startup.MS, y = Startup.Profit)


# Creation of Dummy Variables

print(Startup.info())
Startup.boxplot('Profit','State')


# Checking of null values

cat_Startup = Startup.select_dtypes(include = ['object']).copy()
cat_Startup.head()
print(cat_Startup.isnull().values.sum()) 

print(cat_Startup['State'].value_counts())

#plot for State

State_count = cat_Startup['State'].value_counts()
sns.set(style = "darkgrid")
sns.barplot(State_count.index,State_count.values,alpha = 0.9)
plt.title('State influence on Profit')
plt.ylabel('count of States', fontsize=12)
plt.xlabel('State', fontsize=12)
plt.show()


cat_Startup_onehot_sklearn = cat_Startup.copy()
cat_Startup_onehot_sklearn

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_Startup_onehot_sklearn['State'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

print(lb_results_df.head())

# concate the dummy variable to the data sheet

Startup_df = pd.concat([Startup, lb_results_df], axis=1)
Startup_df
Startup_df = Startup_df.drop(['State'], axis=1)
Startup_df


# Scatter Plot

sns.pairplot(Startup_df.iloc[:, :])
                             
# Correlation matrix 
Startup_df.corr()


# MODEL BUILDING

model = sm.ols('Profit ~ RD + Adm + MS + State', data = Startup).fit()
model.summary()

# From the output the overall R^2 = 0.951 and State and Adm is still insignifcant


# Checking of Influential Values by ploting
import statsmodels.api as smf

smf.graphics.influence_plot(model)

# As the graph shows that 49 index show high influence ,by elimating the value creating the new model

Startup_new = Startup.drop(Startup.index[[49]])

# New Model Building without Outliers

model1 = sm.ols('Profit ~ RD + Adm + MS + State', data = Startup_new).fit()
model1.summary()

# By elimanating the outliers the R^2 value as been increased to 0.962 but still variables show insignificant

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_RD = sm.ols('RD ~ MS + Adm + State', data = Startup).fit().rsquared  
vif_RD = 1/(1 - rsq_RD) 
vif_RD


rsq_MS = sm.ols('MS ~ RD + State + Adm', data = Startup).fit().rsquared  
vif_MS = 1/(1 - rsq_MS)
vif_MS 


rsq_Adm = sm.ols('Adm ~ State + RD + MS', data = Startup).fit().rsquared  
vif_Adm = 1/(1 - rsq_Adm) 
vif_Adm # 1.77

# All the above variable VIF values are < 10,So there is no influation

# calculating vif with dummy variables

rsq_state1 = sm.ols('California ~ Adm + RD + MS', data = Startup_df).fit().rsquared  
vif_state1= 1/(1 - rsq_state1) 
vif_state1 # 1.03

rsq_state2 = sm.ols('Florida ~ Adm + RD + MS', data = Startup_df).fit().rsquared  
vif_state2= 1/(1 - rsq_state2) 
vif_state2 # 1.05

rsq_state3 = sm.ols('NewYork ~ Adm + RD + MS', data = Startup_df).fit().rsquared  
vif_state3= 1/(1 - rsq_state3) 
vif_state3 # 1.03


# this shows there is no collinearity between any of the inputs and making state 
# insignificant

# Final Model

model.final = sm.ols('Profit ~ RD + Adm + MS', data = Startup).fit()
model.final.summary()

# still Admin is insignificant

model.final1 = sm.ols('Profit ~ RD + MS', data = Startup).fit()
model.final1.summary()

# Now the MS significance is also improved and r square is 0.95


# Prediction
pred = model.final1.predict(Startup)

# Q-Q plot
res = model.final1.resid
smf.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

smf.graphics.influence_plot(model.final1)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Startup_train, Startup_test = train_test_split(Startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = sm.ols("Profit ~ RD + MS ", data = Startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(Startup_test)

# test residual values 
test_resid = test_pred - Startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 6275


# train_data prediction
train_pred = model_train.predict(Startup_train)

# train residual values 
train_resid  = train_pred - Startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
 
# 9464

# Over fit need to go with regularization

