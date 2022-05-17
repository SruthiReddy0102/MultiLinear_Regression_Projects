# Input Variables (x) = other variables
# Output Variable(y) = Price

# Importing the dataset
ToyotaCorolla <- read.csv(file.choose())
View(ToyotaCorolla)

# Removing Uncessary columns
ToyotaCorolla <- ToyotaCorolla[ , 2:38]
View(ToyotaCorolla)

# Creating dummy variables for Model,Fuel_Type,Color

install.packages("dummies")
library(dummies)

str(ToyotaCorolla)
ToyotaCorolla$Model <- as.factor(ToyotaCorolla$Model)
ToyotaCorolla$Model = as.numeric(ToyotaCorolla$Model)
ToyotaCorolla$Fuel_Type <- as.factor(ToyotaCorolla$Fuel_Type)
ToyotaCorolla$Fuel_Type = as.numeric(ToyotaCorolla$Fuel_Type)
ToyotaCorolla$Color <- as.factor(ToyotaCorolla$Color)
ToyotaCorolla$Color = as.numeric(ToyotaCorolla$Color)
str(ToyotaCorolla)
View(ToyotaCorolla)
attach(ToyotaCorolla)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(ToyotaCorolla)

# correlation matrix
cor(ToyotaCorolla)

# 

# The Linear Model of interest
model.ToyotoCorolla <- lm(Price ~ ., data = ToyotaCorolla) # lm(Y ~ X)
summary(model.ToyotoCorolla)

# p value shows significant except for few Input variables
#Over all R square is 0.909 and p is less than 0.05

#### Scatter plot matrix with Correlations inserted in graph
# install.packages("GGally")
library(GGally)
ggpairs(ToyotaCorolla)


### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(ToyotaCorolla) #no strong correlatoin between the inputs

cor2pcor(cor(ToyotaCorolla))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.ToyotoCorolla) # Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.ToyotoCorolla, id.n = 5 ,main = "influential obseravations") # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.ToyotoCorolla, id.n = 5) # Index Plots of the influence measures
influencePlot(model.ToyotoCorolla, id.n = 5) # A user friendly representation of the above
#139  and 222 inputs are influential observations

# Regression after deleting the 139th observation
model.ToyotoCorolla1 <- model.ToyotoCorolla <- lm(Price ~ ., data = ToyotaCorolla[-139, ])
summary(model.ToyotoCorolla1)

# Regression after deleting the 222nd observation
model.ToyotoCorolla1 <- model.ToyotoCorolla <- lm(Price ~ ., data = ToyotaCorolla[-222, ])
summary(model.ToyotoCorolla1)
View(ToyotaCorolla)

### Variance Inflation Factors
vif(model.ToyotoCorolla)  # VIF is > 10 => collinearity

alias(Price~.,data = ToyotaCorolla)
ToyotaCorolla.alias <- lm(Price ~ . - Mfg_Year - Cylinders, data = ToyotaCorolla)

summary(ToyotaCorolla.alias)

vif(ToyotaCorolla.alias)
ToyotaCorolla.alias <- lm(Price ~ . - Radio - Radio_cassette, data = ToyotaCorolla)
summary(ToyotaCorolla.alias)


#As per the entire above analysis , Mfg year , Cylinder ,Radio , Radio Cassettee variable results in insignificant

# Linear Model without State
model.final <- lm(Price ~ ., data = ToyotaCorolla)
summary(model.final)


# Evaluation Model Assumptions
plot(model.final)
plot(model.final$fitted.values, model.final$residuals)

qqnorm(model.final$residuals)
qqline(model.final$residuals)

# Data Partitioning
n <- nrow(ToyotaCorolla)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- ToyotaCorolla[-train, ]

# Model Training
model <- lm(Price ~ ., ToyotaCorolla[train, ])
summary(model)
# Over all R^2 value is 0.91

pred <- predict(model, newdata = test)
actual <- test$Price
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.ToyotoCorolla)


# Still it is overfit need to do regularization.


