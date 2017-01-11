#You'll need the randomForest library from CRAN.
library(randomForest)
library(ROCR)
library(e1071)
library(C50)

# The data should be provided for the latest version of the dataset.
emply_all <- read.csv("lawsocemplytwitter.csv")

# Display data structure. The ID and Name variables refer to the potential 
# employee, the ConnectedFirm variables identify the employer. The rest of 
# the variables are automatic features potentially connecting the two, apart 
# from MatchesRoster, which is the outcome variable, which is semi-automatic,
# with all matches and partial matches to a name on the employer's public roster
# manually checked (but non-matches not checked).
# - Onsite          : whether the profile's name matches a name on the roster page of the employer's website. 'Maybe' is a partial name match.
# - HasFirmName     : the employee profile mentions a version of the firm's name in its self-descriptions
# - Mentioned       : the employee mentions a version of the firm's name in its twitter status history
# - MentionedBy     : the employer mentions the employee's name in its twitter history
# - IsFollowed      : the employer follows the employee's profile
# - IsFollowing     : the employee follows the employer's profile
# - FollowedMatches : the number of twitter users following both employee and employer
# - FollowingMatches: the number of twitter users both employee and employer are following
str(emply_all)

# Matches Roster is the outcome variable.
# You can see that there is a very low base rate of users connected to an employer 
# being its employees (as far as we can tell, but my impression from reviewing possible
# matches is that there's a tendency for these accounts to be 'in the industry', rather 
# than odd people who we can't confirm because they're not on the list). 
table(emply_all$MatchesRoster)

# Remove the dependent variable and the identifiers (as these would permit the tree to just distinguish per-account, which wouldn't be useful).
emply_use <- emply_all[,c('OnSite','HasFirmName','Mentioned','MentionedBy','IsFollowed','IsFollowing','FollowedMatches','FollowingMatches', 'MatchesRoster')]

#We construct a cost matrix which penalises false positives more harshly than false negatives.
costs <- matrix(c(0,1,4,0),nrow=2)

#Simple overview is just to throw all the data into the tree. We don't have many examples to work with.
emply_model <- randomForest(as.factor(MatchesRoster) ~ OnSite + HasFirmName + Mentioned + MentionedBy + IsFollowed + IsFollowing + FollowedMatches + FollowingMatches, data=emply_use, importance=T, ntree=2000)

#The summary reveals the rules the decision tree was able to build to divide the data so it corresponds
#with the output, and the resultant misclassification. 
#print(summary(emply_model))

#F-score is just a function of precision and recall. 
#we'll use the traditional balanced version for now.
fscore <- function(pre, rec){
 return (2*((pre*rec)/(pre+rec)))
}

# We can construct precision and recall from the predictions
display_precision <- function(model, data, truth){
  p <- predict(model, data)
  positive_pred <- sum(p == 'YES')
  print(paste("Predicts", positive_pred))
  positive_know <- sum(truth == 'YES')
  true_positives <- sum(p == 'YES' & truth == 'YES')

  precision <- true_positives / positive_pred 
  recall <- true_positives / positive_know 

  f1 <- fscore(precision,recall)

  return(c(precision, recall, f1))
}

print("All Data")
display_precision(emply_model, emply_use, emply_all$MatchesRoster)


# Just looking at the decision tree's performance on its training data is likely to overfit.
# We are probably doing this anyway, because the number of positive examples is very small,
# so chance combinations of characteristics may unduly influence the tree-building.
# However, it's worth dividing the tree into a training and testing set to see how the tree 
# thus built will perform on seen examples. 

#Randomise the dataset so as to avoid selecting e.g. one company as test data.
set.seed(12345)
emply_order <- order(runif(length(emply_use$OnSite)))

emply_use_rand <- emply_use[emply_order,]

tenth <- 375
max <- 3752
results <- c()

for (i in 0:9) {
  
  lower <- i*tenth+1
  upper <- i*tenth+tenth

  #90:10 train:test 
  emply_use_train <- emply_use_rand[c(1:lower,upper:max),]

  emply_use_test  <- emply_use_rand[lower:upper,]

  #Train a new decision tree
  emply_model <- randomForest(as.factor(MatchesRoster) ~ OnSite + HasFirmName + Mentioned + MentionedBy + IsFollowed + IsFollowing + FollowedMatches + FollowingMatches, data=emply_use_train, importance=T, ntree=2000)
  # The new model is slightly less complex, which i a sign
  # that the other was overfitting. However, it agrees about 
  # the most important attributes, as you would expect.

  print("Train/Test")
  #See how it does on the test data
  results <- cbind(results,display_precision(emply_model, emply_use_test, emply_use_test$MatchesRoster))
}

#results <- data.frame(t(results))
#png('../combined/employepr.png')
#plot(results$X2, results$X1, xlab='recall', ylab='precision', ylim=c(0,1), xlim=c(0,1))
#dev.off()
averages <- sapply(results, function(x){ mean(na.omit(x[!is.nan(x)]))})

print(averages)

