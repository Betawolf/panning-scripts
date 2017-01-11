#The ROCR package for various plots and performance assessments
library(ROCR)
library(PRROC)

#The data from the attribute comparisons.
ident_all <- read.csv('identity.csv')

# Display the data structure. The origin and target variables identify the
# two profiles involved in the comparison. The block variable identifies the 
# block which produced this comparison -- within each block there will be at 
# least one true comparison between profiles and a number of negatives which
# were formed by selecting candidate profiles using a SNSs's name-based search.
# All comparisons should have an origin network of Google+, as this is how most
# connections are found (allowing comparisons across other networks results in
# an explosion of negative examples). 
#
# The outcome variable indicates whether the profiles should or should not be 
# matched. The remaining variables form the
# comparison vector -- the similarity measures between the various attributes of 
# the profile. These are:
# - exactnames     : the number of name components which match exactly between two profiles.
# - bestname       : the (inverted) levenshtein editdistance between the two best (most likely to be real) name strings.
# - timecomparison : the proportion of time activity peaks and troughs which occur in the same window (1/6th of a day)
# - avatars        : pixel similarity between the two profiles' avatars.
# - friends        : the proportion of friends with the same names (editdistance measure at threshold 0.8).
# - linkactivity   : the proportion of identical web links found in profile content
# - stylometry     : euclidean distance between proportions of function words
# - geography      : proportion of locations which are 'near' each other (within 10k or substring of text). 
str(ident_all)

# You can see that only 1% of the data is matches.
# This reflects the challenging base rates -- searches for a person's name
# often return many matching profiles.
table(ident_all$outcome)

# As we are building a classifier, we need to test a number of thresholds on these numbers.
# Typically in record linkage we would have 'matched' 'not-matched' and 'possible match',
# the third being for human review, but the final class needs to be omitted here, so we're
# just building a binary classifier (i.e. one threshold). 

# The below function reports precision, recall and accuracy for a threshold.
accuracy <- function(level){
  predict <- ifelse(lmpred > level,1,0)
  true <- sum(predict & ident_test$outcome == 1)
  pre <- true / sum(predict == 1)
  rec <- true / sum(ident_test$outcome == 1)
  f1 <- 2 * ((pre * rec)/(pre + rec))
  return(c(level, pre, rec, f1))
}

# For modelling purposes, we need only the comparison vector and outcome.
ident <- ident_all[,c('exactnames','bestname','timeactivity','avatars','friends','linkactivity','stylometry','geography','outcome')]
trues <- ident[ident$outcome == 1,]
falses <- ident[ident$outcome == 0,]

max_f <- length(falses$outcome)
tenth_f <- floor(max_f/10)
max_t <- length(trues$outcome)
tenth_t <- floor(max_t/10)

auc_results <- c()
etc_results <- c()

cols = c('orangered','orchid','palegreen','paleturquoise', 'palevioletred', 'royalblue', 'seagreen','sienna','tan')
fired = 0

preds <- c()
truth <- c()
png('roc.png')
#Split into training and test data: 90:10. 
for (i in 0:9) {

  lower_f <- i*tenth_f+1
  upper_f <- i*tenth_f+tenth_f

  lower_t <- i*tenth_t+1
  upper_t <- i*tenth_t+tenth_t

  ident_test <- rbind( trues[lower_t:upper_t,], falses[lower_f:upper_f,])
  ident_train <- rbind( trues[c(1:lower_t,upper_t:max_t),], falses[c(1:lower_f,upper_f:max_f),])

  # Build a model, including interactions for all of the features (as we expect them to support each other).
  ident_model <- glm(outcome ~ .^2, data=ident_train, family='binomial')

  # Make numeric predictions based on the linear model.
  lmpred <- predict(ident_model, ident_test, type='response')
  preds <- c(preds, lmpred)
  truth <- c(truth, ident_test$outcome)

  #A summary of the predictions shows the range of values
  print(summary(lmpred))

  rcr_pred <- prediction(lmpred, ident_test$outcome)

  #Likely levels based on the range reported in the summary, steps of 0.1
  levels <- seq(round(min(lmpred),1), round(max(lmpred),1), 0.1)
#  levels <- seq(0.0, 0.3, 0.01)

  #Gather the report for each level into a data frame.
  res <- data.frame(t(sapply(levels,accuracy)))
  names(res) <- c('level','precision','recall','f1score')

  print(res)
  bl = 0.3
  print("Best threshold:")
  print(na.omit(res[res$level+0.01 > bl & res$level-0.01 < bl,]))
  etc_results <- cbind(etc_results, as.list(head(res[res$level+0.01 > bl & res$level-0.01 < bl,],1)))

  rcr_perf <- performance(rcr_pred, 'tpr','fpr')
  if (fired == 0){
    plot(rcr_perf, col=cols[i], lwd=2)
    fired = 1
  }
  else {
    plot(rcr_perf, col=cols[i], add=T, lwd=2)
  }
  # Calculate the AUC by comparing score values.
  pos.scores <- lmpred[which(ident_test$outcome == 1)]
  neg.scores <- lmpred[which(ident_test$outcome == 0)]
  auc_approx <- mean(sample(pos.scores,10000,replace=T) > sample(neg.scores,10000,replace=T))
  auc_results <- c(auc_results, auc_approx) 
  print(paste("AUC :",auc_approx))
}
overall_pred <- prediction(preds, truth)
overall_perf <- performance(overall_pred, 'tpr','fpr')
plot(overall_perf, col='black', add=T, lwd=3, lty=2)
dev.off()

auprcs <- c()
png('pr.png')
#Split into training and test data: 90:10. 
for (i in 0:9) {

  lower_f <- i*tenth_f+1
  upper_f <- i*tenth_f+tenth_f

  lower_t <- i*tenth_t+1
  upper_t <- i*tenth_t+tenth_t

  ident_test <- rbind( trues[lower_t:upper_t,], falses[lower_f:upper_f,])
  ident_train <- rbind( trues[c(1:lower_t,upper_t:max_t),], falses[c(1:lower_f,upper_f:max_f),])

  # Build a model, including interactions for all of the features (as we expect them to support each other).
  ident_model <- glm(outcome ~ .^2, data=ident_train, family='binomial')

  print(summary(ident_model))

  # Make numeric predictions based on the linear model.
  lmpred <- predict(ident_model, ident_test, type='response')


  #A summary of the predictions shows the range of values
#  print(summary(lmpred))

  rcr_pred <- prediction(lmpred, ident_test$outcome)

  #Build PRROC performance object from predict output.
  truth <- rcr_pred@labels[[1]]
  posvals <- rcr_pred@predictions[[1]][truth == 1]
  negvals <- rcr_pred@predictions[[1]][truth == 0]
  perf_obj <- pr.curve(scores.class0=posvals, scores.class1=negvals, curve=T)

  if (i == 0){
    plot(perf_obj, legend=F, color=i, lty=1, auc.main=F, main='')
  }
  else {
    plot(perf_obj, legend=F, color=i, add=T, lty=1, auc.main=F)
  }

  # Calculate the AUPRC 
  print(paste("AUPRC :",perf_obj$auc.integral))
  auprcs <- c(auprcs, perf_obj$auc.integral)
}
overall_perf <- performance(overall_pred, 'prec','rec')
plot(overall_perf, color=1, add=T, lwd=3, lty=2)
dev.off()


etc_results <- data.frame(t(etc_results))
print(paste('precision:',mean(unlist(etc_results$precision))))
print(paste('recall:',mean(unlist(etc_results$recall))))
print(paste('f1score:',mean(unlist(etc_results$f1score))))


print(paste("Average AUROC :",mean(auc_results)))
print(paste("Average AUPRC :",mean(auprcs)))

