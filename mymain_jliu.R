library(glmnet)
library(data.table)
library(pROC)
library(ROCR)
library(tictoc)

# Start timing
tic("Running Script")

################
#read the file
################

train = fread("train.csv", 
              stringsAsFactors = FALSE, 
              header = TRUE)

train <- as.data.frame(train)

train_x <- as.matrix(train[, 4:ncol(train)])  # from the 4th columns are the 1536- embeddings
#train_y <- as.factor(train$sentiment)
train_y <- train$sentiment

#####################################
# Train a binary classification model 
# and, find the best hyperparameter
#####################################

# Step 2: Set up cross-validation with AUC as the evaluation metric

# library(glmnetUtils)
# 
# # Run Elastic Net with multiple alpha values
# cv_model <- cva.glmnet(
#   x = train_x,
#   y = train_y,
#   family = "binomial",
#   alpha = seq(0, 1, by = 0.2),  # Automatically loops through alpha values
#   standardize = FALSE,
#   type.measure = "auc"
# )


# alpha_values <- 0.25
# cv_results <- list()
# 
# # Perform cross-validation for each alpha value
# set.seed(123)  # For reproducibility
# cv_model <- cv.glmnet(
#   x = train_x,
#   y = train_y,
#   family = "binomial",
#   alpha = alpha_values,          # Elastic Net mixing parameter
#   standardize = FALSE,  
#   type.measure = "auc",  # Use AUC for binary classification
#   lambda = exp(seq(-11.5, -13.5, length.out = 50)),
#   nfolds=5
# )
# 
# print(cv_model$lambda.min)
# cv_model$lambda
# best_alpha <- alpha_values
# best_lambda <- cv_model$lambda.min
# best_lambda
# 
# # # Extract the best alpha and corresponding model
# # best_alpha <- cv_model$best.alpha
# # best_lambda <- cv_model$modlist[[best_alpha]]$lambda.min
# 
# final_model <- glmnet(
#   x = train_x,
#   y = train_y,
#   family = "binomial",
#   alpha = best_alpha, 
#   lambda = best_lambda           
# )
# summary(final_model)
library(xgboost)

# grid <- expand.grid(
#   #gamma = c(1,5,10),          # Values for gamma
#   alpha = c(0,0.5,1)       # Values for max_depth
# )
# 
# results <- data.frame(
#   gamma = numeric(),
#   max_depth = numeric(),
#   auc = numeric()
# )
# 
# for (i in 1:nrow(grid)) {
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "auc",            # Use AUC as the evaluation metric
  eta = 0.1,                      # Learning rate
  max_depth = 8,                  # Maximum depth of a tree
  subsample = 0.8,                # Subsample ratio of the training data
  colsample_bytree = 0.7,
  alpha= 0.2,
    #grid$alpha[i],
  lambda = 1
)


dtrain <- xgb.DMatrix(data = train_x, label = as.numeric(train_y))

# # Step 2: Cross-Validation to Find Best Parameters
# set.seed(123) 
# cv_results <- xgb.cv(
#   params = params,
#   data = dtrain,
#   nfold = 5,
#   nrounds = 100,
#   early_stopping_rounds = 10,
#   verbose = 0
# )

# # Extract the best number of rounds based on AUC
# best_nrounds <- cv_results$best_iteration
# print(best_nrounds)
# Record the best AUC
# best_auc <- max(cv_results$evaluation_log$test_auc_mean)
# 
# # Save the results
# results <- rbind(results, c(grid$alpha[i],best_auc))
# }
# colnames(results) <- c("alpha","auc")

# Print the best combination of parameters
# best_params <- results[which.max(results$auc),]

# final_params <- list(
#   objective = "binary:logistic",
#   eval_metric = "auc",
#   eta = 0.1,
#   max_depth = 8,        
#   subsample = 0.8,               
#   colsample_bytree = 0.7,
#   alpha = best_params["alpha"],
#   lambda=5
# )

# Step 3: Train the Final Model
final_model <- xgboost(
  params = params,
    #final_params,
  data = dtrain,
  nrounds = 800, 
  verbose = 0            
)
xgb.save(final_model, "final_model.model")
#####################################
# Load test data, and
# Compute prediction
#####################################

test = fread("test.csv", 
             stringsAsFactors = FALSE, 
             header = TRUE)

test <- as.data.frame(test)

test_x <- as.matrix(test[, 3:ncol(test)])
dtest <- xgb.DMatrix(data = test_x)
pred <- predict(final_model, newdata = dtest)

# pred = predict(final_model,
#                s = best_lambda,
#                newx = test_x,
#                type='response')

#####################################
# Store prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################

# Prepare the output data frame
output = cbind(test$id, pred)
colnames(output) = c("id", "prob")


# Write the predictions to a file
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep = '\t')

# End timing
toc()