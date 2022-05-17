# Input Variables (x) = R.D Spend , Administration , Marketing Spend , State
# Output Variable(y) = Profit

# Importing the dataset
Startup <- read.csv(file.choose())
colnames(Startup) <- c("RD","Admin","MS","State","Profit") # RD = R.D.Spend , Admin = Administrartion & MS = Marketing Spend
View(Startup)

# Creating dummy variables for State

install.packages("dummies")
library(dummies)

str(Startup)
Startup$State <- as.factor(Startup$State)
Startup$State = as.numeric(Startup$State)
str(Startup)
View(Startup)
attach(Startup)

# Normal distribution
qqnorm(RD , main = "R.D.Spend")
qqline(RD, main = "R.D.Spend")

qqnorm(Admin , main = "Administrartion Spend")
qqline(Admin , main = "Administrartion Spend")

qqnorm(MS , main = "Marketing Spend")
qqline(MS, main = "Marketing Spend")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Startup)

# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Boxplot Representation

boxplot(Startup$RD, col = "dodgerblue4",main = "R.D.Spend")
boxplot(Startup$Admin, col = "dodgerblue4",main = "Administration")
boxplot(Startup$MS, col = "dodgerblue4",main = "Marketing Spend")
boxplot(Startup$Profit, col = "red", horizontal = T,main = "Profit")

# Histogram Representation

hist(Startup$RD,col = "blue", main = "R.D.Spend" )
hist(Startup$Admin,col = "blue",main = "Administration")
hist(Startup$MS,col = "blue", main = "Marketing Spend")
hist(Startup$Profit,col = "red", main = "Profit")

# Scatter plot
plot(RD,Profit) # Plot relation ships between each X with Y
plot(Admin, Profit)
plot(MS,Profit)
#plot(State,Profit) #this shows error as it is a categorical variable

# Or make a combined plot
#pairs(Start_up)   #  doesnt work as there is a categorical variable
#Scatter plot for all pairs of variables
plot(Startup)

cor(RD, Profit) #0.973
cor(Startup) # correlation matrix
#Seems Strong coorelation between profit and R.D.Spend

# The Linear Model of interest
model.Startup <- lm(Profit ~ RD + Admin + MS + State, data = Startup) # lm(Y ~ X)
summary(model.Startup)
# p value shows insignificant for Admin,MS,State
#Over all R square is 0.9507 and p is less than 0.05

model.Admin <- lm(Profit ~ Admin)
summary(model.Admin) 

# p value is 0.1622 which is greater than 0.05 
#and r square value is also less 0.04029 which makes this insignificant

model.MS <- lm(Profit ~ MS)
summary(model.MS)

# Overall p values are significant so we can consider MS

model.State <- lm(Profit ~ State)
summary(model.State) 

#P values making state insignificant

model.AMS <- lm(Profit ~ Admin + MS + State )
summary(model.AMS) 


#### Scatter plot matrix with Correlations inserted in graph
# install.packages("GGally")
library(GGally)
ggpairs(Startup)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(Startup) #no strong correlatoin between the inputs

cor2pcor(cor(Startup)) # RD and Profit has strong correaltion

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.Startup) # Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.Startup, id.n = 5 ,main = "Influential Observations") # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.Startup, id.n = 3) # Index Plots of the influence measures
influencePlot(model.Startup, id.n = 3 , main = "Influential Observations") # A user friendly representation of the above
#50  and 49 inputs are influential observations

# Regression after deleting the 50th observation
model.Startup1 <- lm(Profit ~ RD + Admin + MS + State, data = Startup[-50, ])
summary(model.Startup1)
# still admin and state seems to be insignificant
model.Startup1 <- lm(Profit ~ RD + Admin + MS + State, data = Startup[-49, ])
summary(model.Startup1)
View(Startup)

#still no use

### Variance Inflation Factors
vif(model.Startup)  # VIF is > 10 => collinearity

#There is no VIF greater than 10  so there is no collinearity problem in this model

# Regression model to check R^2 on Independent variales
VIFRD <- lm(RD ~ Admin + MS + State) # state insignificant
VIFAdmin <- lm(Admin ~ RD + MS + State)# state insignificant
VIFMS <- lm(MS ~ RD + Admin + State)#state insignificant
#VIFState <- lm(State ~ RD + Admin + MS)

summary(VIFRD)
summary(VIFAdmin)
summary(VIFMS)
#summary(VIFState)

# VIF of RDS
1/(1-0.59) # <10 no multi collinearity double confirmed

#### Added Variable Plots ######
avPlots(model.Startup, id.n = 2, id.cex = 0.8, col = "red")

#As per the entire above analysis , state variable results in insignificant

# Linear Model without State
model.final <- lm(Profit ~ RD + Admin + MS, data = Startup)
summary(model.final)

# Linear model without State and influential observation
model.final1 <- lm(Profit ~ RD + Admin + MS, data = Startup[-50, ])
summary(model.final1)


# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")
# admin is insignificant

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

# Data Partitioning
n <- nrow(Startup)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- Startup[-train, ]

# Model Training
model <- lm(Profit ~ RD + Admin + MS, Startup[train, ])
summary(model)
# r square is 0.946 and overall p values is good but Admin is still insignificant

model <- lm(Profit ~ RD + MS, Startup[train, ])
summary(model)

pred <- predict(model, newdata = test)
actual <- test$Profit
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.Startup)




