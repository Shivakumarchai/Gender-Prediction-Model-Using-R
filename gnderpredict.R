install.packages("class")        
install.packages("e1071")        
install.packages("rpart")        
install.packages("caret")        

library(class)
library(e1071)
library(rpart)
library(caret)

# Loading the dataset
data1 <- read.csv("D:/B.tech 7 Semester/Predictive analysis int 234/gendertest.csv",stringsAsFactors = FALSE)
# categorical columns changed to factors
View(data1)
str(data1)
data1$Gender <- as.factor(data1$Gender)
data1$caring <- as.factor(data1$caring)
data1$craving <- as.factor(data1$craving)
data1$morning.sickness <- as.factor(data1$morning.sickness)
data1$describe.pregnancy <- as.factor(data1$describe.pregnancy)

# splitting it and training data as 80% and testing data as 20%
# at 80% testing it's giving 50% accuracy for both algorithm but it was varying at all execution
set.seed(123)  # to generate same number of samples
index <- createDataPartition(data1$Gender, p = 0.8, list = FALSE)  


train_data <- data1[index, ]  # training phase
test_data <- data1[-index, ]  # testing phase

# applying Naive Bayes algorithm here ,the necessary packages are installed previously
nb_model <- naiveBayes(Gender ~ ., data = train_data)
nb_predictions <- predict(nb_model, test_data)
nb_predictions
# accuracy check for Naive Bayes
nb_conf_matrix <- confusionMatrix(nb_predictions, test_data$Gender)
print(nb_conf_matrix)

# Applying Decision Tree algorithm
tree_model <- rpart(Gender ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, test_data, type = "class")

# accuracy check for Decision Tree algorithm
tree_conf_matrix <- confusionMatrix(tree_predictions, test_data$Gender)
print(tree_conf_matrix)


# printing the accuracy at one place for both the algorithm

cat("Naive Bayes Accuracy: ", nb_conf_matrix$overall['Accuracy'], "\n")
cat("Decision Tree Accuracy: ", tree_conf_matrix$overall['Accuracy'], "\n")


new_data<-data.frame(caring="High",craving.sweets="I'm craving sweets.",morning.sickness="I had severe morning sickness.",
                     describe.pregnancy="I look better than I ever have.",craving="fruits and veggies")
new_predictions<-predict(tree_model,new_data,type = "prob")
new_predictions

new_predictions_df <- as.data.frame(new_predictions)

# Find the gender with the highest probability
max_index <- which.max(new_predictions_df)

# Print the corresponding gender based on the highest probability
predicted_gender <- colnames(new_predictions_df)[max_index]
cat("The predicted Gender for the new_data is:", predicted_gender)
