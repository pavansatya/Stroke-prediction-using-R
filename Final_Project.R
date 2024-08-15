### Install and load required packages
setwd("~/Desktop/Regression_R_project")
rm(list=ls())
library(readr)
library(corrplot)
library(ggplot2)
library(nnet)
library(e1071)
library(forcats)
library(rsample)
library(MASS)
library(caret)
library(mixlm)

set.seed(6554)
### Load the data
data <- read_csv("healthcare-dataset-stroke-data.csv")

# Convert character into numeric type
data$bmi <- as.numeric(data$bmi)

# Check for missing values and replacing with mean value
colSums(is.na(data))
mean_value <- mean(data$bmi, na.rm = TRUE)
data$bmi[is.na(data$bmi)] <- mean_value
values <- data$bmi
breaks <- c(0, 18,5, 25, 35, 100)
categories <- cut(values, breaks = breaks, labels = c("Under weight", "Ideal", "Over weight", "Obese", "Highly obese"), include.lowest = TRUE)
data$bmi <- factor(categories) 

# remove id column
data <- subset(data, select = -c(id))
summary(data)

# Boxplot
boxplot(data$age, data$avg_glucose_level, data$bmi,
        main = "Boxplot for Numerical Variables",
        names = c("Age", "Average glucode level", "BMI"))

# Example histogram for numerical variable
ggplot(data, aes(x = avg_glucose_level)) + geom_histogram()

# Example boxplot for comparing numerical variable across categories
ggplot(data, aes(x = smoking_status, y = avg_glucose_level)) + geom_boxplot()


# Categorising age values into 9 types 
values <- data$age
breaks <- seq(0, 90, by = 10)
categories <- cut(values, breaks = breaks, labels = FALSE)
data$age <- factor(categories) 


# Categorising avg_glucose_level into 3 types
values <- data$avg_glucose_level
breaks <- c(0, 80, 160, 275)
categories <- cut(values, breaks = breaks, labels = c("Low", "Medium", "High"), include.lowest = TRUE)
data$avg_glucose_level <- factor(categories) 


# Categorical variable to numerical variable 
data$gender <- factor(data$gender, levels=c("Male", "Female", "Other"))
data$ever_married <- factor(data$ever_married, levels=c('Yes', 'No'))
data$work_type <- factor(data$work_type, levels=c("Private", "Self-employed", "Govt_job", "children", "Never_worked"))
data$Residence_type <- factor(data$Residence_type, levels=c('Rural', 'Urban'))
data$smoking_status <- factor(data$smoking_status, levels=c('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

data <- as.data.frame(sapply(data, function(x) if(is.factor(x)) x else as.numeric(as.character(x))))
summary(data)

#(n = dim(data)[1])
smp_size <- floor(0.80 * nrow(data))
index <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[index, ]
test <- data[-index, ]

#index = sample(1:n, 4088) #80% of training and 20% of test data
#train = data[index,] 
#test = data[-index,]
dim(train)
dim(test)

# Correlation matrix
cor_matrix <- cor(data)
corrplot(cor_matrix, type="upper") 

#Data Exploration and Features Selection

fullmodel <- glm(stroke ~ ., data = train, family=binomial(link='logit'))
summary(fullmodel)

#Model building - 1
glm_model <- glm(stroke ~ age + hypertension + heart_disease + avg_glucose_level, data = train, family=binomial(link='logit'))
summary(glm_model)

#Evaluation
pHatLog <- predict(glm_model, train,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Training Set of Model-1 = %f",100 * mean(train$stroke == yHatLog))

yTP2 <- ifelse(yHatLog == 1 & train$stroke == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & train$stroke == 0, 1, 0)
sprintf("Sensitivity for Training Set of Model-1 = %f",100*(sum(yTP2)/sum(train$stroke==1)))
sprintf("Specificity for Training Set of Model-1 = %f",100*(sum(yTN2)/sum(train$stroke==0)))

pHatLog <- predict(glm_model, test,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Testing Set of Model-1 = %f",100 * mean(test$stroke == yHatLog))

yTP2 <- ifelse(yHatLog == 1 & test$stroke == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & test$stroke == 0, 1, 0)
sprintf("Sensitivity for Testing Set of Model-1 = %f",100*(sum(yTP2)/sum(test$stroke==1)))
sprintf("Specificity for Testing Set of Model-1 = %f",100*(sum(yTN2)/sum(test$stroke==0)))

# ROC & AUC curve

library(pROC)
predictions <- predict(glm_model, type = "response", newdata = test)
roc_curve <- roc(test$stroke, predictions)
plot(roc_curve, 
     col = "blue", 
     main = "ROC Curve",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")

# Add diagonal line (random classifier)
abline(a = 0, b = 1, lty = 2, col = "red")
auc_value <- round(auc(roc_curve), 2)
legend("bottomright", 
       legend = paste("AUC =", auc_value),
       col = "blue", 
       lwd = 1, 
       bty = "n")

# Model building - 2
# Naive Bayes model
nb_model <- naiveBayes(stroke ~ age + hypertension + heart_disease + avg_glucose_level, data=train)
summary(nb_model)

train_predictions <- predict(nb_model, train)
test_predictions <- predict(nb_model, test)
train_conf_matrix <- table(Predicted = train_predictions, Actual = train$stroke)
test_conf_matrix <- table(Predicted = test_predictions, Actual = test$stroke)


# Calculate accuracy, precision & recall
train_accuracy = sum(diag(train_conf_matrix))/sum(train_conf_matrix)
train_recall = train_conf_matrix[2,2] / (train_conf_matrix[2,2] + train_conf_matrix[1,2])
train_precision = train_conf_matrix[2,2] / (train_conf_matrix[2,2] + train_conf_matrix[2,1])

print(paste("Naive Bayes Train Accuracy:", train_accuracy))
print(paste("Naive Bayes Train Recall:", train_recall))
print(paste("Naive Bayes Train Precision:", train_precision))

test_accuracy = sum(diag(test_conf_matrix))/sum(test_conf_matrix)
test_recall = test_conf_matrix[2,2] / (test_conf_matrix[2,2] + test_conf_matrix[1,2])
test_precision = test_conf_matrix[2,2] / (test_conf_matrix[2,2] + test_conf_matrix[2,1])

print(paste("Naive Bayes Test Accuracy:", test_accuracy))
print(paste("Naive Bayes Test Recall:", test_recall))
print(paste("Naive Bayes Test Precision:", test_precision))

# Plotting
metrics = c('Accuracy', 'Recall', 'Precision')

data <- data.frame(
  Type = rep(c("Training", "Testing"), each = 3),
  Metric = rep(metrics, times = 2),
  Value = c(train_accuracy, train_recall, train_precision, test_accuracy, test_recall, test_precision))

ggplot(data, aes(x = Metric, y = Value, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_fill_brewer(palette = "Paired") +  
  labs(title = "Naive Bayes evaluation metrics",
       x = "Metric",
       y = "Value",
       fill = "Dataset Type") +
  theme_minimal() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Evaluate model performance for Naive Bayes
confusionMatrix(test_predictions, as.factor(test$stroke))
