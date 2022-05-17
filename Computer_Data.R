# Input Variables (x) = speed,hd,ram,screen,cd,multi,premium,ads,trend
# Output Variable(y) = Sales_Price

# Importing the dataset
Computer_Data <- read.csv(file.choose())
View(Computer_Data)

# Removing Uncessary columns
Computer_Data <- Computer_Data[ , 2:11]
View(Computer_Data)

# Creating dummy variables for State

install.packages("dummies")
library(dummies)

str(Computer_Data)
Computer_Data$cd <- as.factor(Computer_Data$cd)
Computer_Data$cd = as.numeric(Computer_Data$cd)
Computer_Data$multi <- as.factor(Computer_Data$multi)
Computer_Data$multi = as.numeric(Computer_Data$multi)
Computer_Data$premium <- as.factor(Computer_Data$premium)
Computer_Data$premium= as.numeric(Computer_Data$premium)
str(Computer_Data)
View(Computer_Data)
attach(Computer_Data)

# Normal distribution
qqnorm(speed)
qqline(speed)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Computer_Data)


# Scatter plot
# Or make a combined plot
#Scatter plot for all pairs of variables
plot(Computer_Data)

# correlation matrix
cor(Computer_Data) 


# The Linear Model of interest
model.Computer_Data <- lm(price ~ ., data = Computer_Data) # lm(Y ~ X)
summary(model.Computer_Data)
# p value shows significant for all Input variables
#Over all R square is 0.77 and p is less than 0.05

# install.packages("GGally")
library(GGally)
ggpairs(Computer_Data)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(Computer_Data) #no strong correlatoin between the inputs

cor2pcor(cor(Computer_Data)) # RD and Profit has strong correaltion

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.Computer_Data) # Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.Computer_Data, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

w# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.Computer_Data, id.n = 3) # Index Plots of the influence measures
influencePlot(model.Computer_Data, id.n = 3 ,main = "Influential Observations") # A user friendly representation of the above
#1441 and 1701 inputs are influential observations

# Regression after deleting the 50th observation
model.Computer_Data1 <- model.Computer_Data <- lm(price ~ ., data = Computer_Data [-1441, ])
summary(model.Computer_Data1)
model.Computer_Data1 <- model.Computer_Data <- lm(price ~ ., data = Computer_Data [-1701, ])
summary(model.Computer_Data1)
View(Computer_Data)



#### Added Variable Plots ######
avPlots(model.Computer_Data, id.n = 2, id.cex = 0.8, col = "red")


# Final Model
model.final <- lm(price ~ ., data= Computer_Data)
summary(model.final)


# Evaluation Model Assumptions
plot(model.final)
plot(model.final$fitted.values, model.final1$residuals)

qqnorm(model.final$residuals)
qqline(model.final$residuals)

# Data Partitioning
n <- nrow(Computer_Data)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- Computer_Data[-train, ]

# Model Training
model <- lm(price ~ ., Computer_Data[train, ])
summary(model)
# r square is 0.77 and overall p values is also good to go


pred <- predict(model, newdata = test)
actual <- test$price
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.Computer_Data)


# As there is no much variation in test and train RMSE,the model is right fit


