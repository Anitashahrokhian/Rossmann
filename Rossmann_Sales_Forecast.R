

#-------------------------------------------------------------------------------
#load packages
#-------------------------------------------------------------------------------
library(readr)
library(randomForest)
library(xgboost)
library(dplyr)
library(lubridate)
library(gridExtra)
library(grid)
library(ggplot2)
library(patchwork)
library(ggpubr)

#-------------------------------------------------------------------------------
# Load Train, Test, and Store data 
#-------------------------------------------------------------------------------
trainDat=read.csv("W21_train.csv", header = T)
storeDat =read.csv("W21_store_info.csv", header = T)

#merge store information and train and test
train_total=left_join(trainDat,storeDat, by = "Store")
train_total=train_total %>% filter(Sales > 0, Open == 1)

#check if we have missing values
train_total[is.na(train_total)]=0


#-------------------------------------------------------------------------------
# Decompose date on training data
#-------------------------------------------------------------------------------
train_total$month=as.integer(month(ymd(train_total$Date)))
train_total$year=as.integer(year(ymd(train_total$Date)))
train_total$day=as.integer(day(ymd(train_total$Date)))

#Remove the date column (after decomposing) and also StateHoliday(many zeros)
train_total=train_total[,-c(3,8)]

set.seed(42)  # for reproducibility

# Split the data to make preditcion
n <- nrow(train_total)
index <- sample(1:n, size = 0.8 * n)
train <- train_total[index, ]
test <- train_total[-index, ]




#-------------------------------------------------------------------------------
#pick the variables 
#-------------------------------------------------------------------------------

Variables=names(train)[c(1,2,6:9,11:13)]
Variables

#change the categorical variables to integer
for (i in Variables) {
  if (class(train[[i]])=="character") {
    levels <- unique(c(train[[i]], test[[i]]))
    train[[i]]=as.integer(factor(train[[i]], levels=levels))
    test[[i]]=as.integer(factor(test[[i]],  levels=levels))
  }
}

#-------------------------------------------------------------------------------
#Data Visualization
#-------------------------------------------------------------------------------


#Average sale 
Dayofweek_mean_sales = aggregate(trainDat$Sales, by=list(trainDat$DayOfWeek), FUN=mean)
data_dayofweek = data.frame(DayOfWeek=c("1", "2", "3","4","5","6","7"),
                            Sales=Dayofweek_mean_sales$x)
# Enhanced ggplot
plot_dayofweek1 <- ggplot(data_dayofweek, aes(x = DayOfWeek, y = Sales)) +
  geom_bar(stat = "identity", fill = "#FF69B4", width = 0.7) +  # soft pink tone
  geom_text(aes(label = round(Sales, 0)), 
            vjust = -0.5, size = 4, color = "black", fontface = "bold") +
  labs(
    title = "Average Sales by Day of the Week",
    x = "Day of the Week",
    y = "Average Sales"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 12),
    panel.grid.major.y = element_line(color = "grey80"),
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, max(data_dayofweek$Sales) * 1.15)



# boxplot of sales 
Data_box=data.frame(DayOfWeek=as.factor(trainDat$DayOfWeek),Sales=trainDat$Sales)
plot_dayofweek2 <- ggplot(Data_box, aes(x = DayOfWeek, y = Sales, fill = DayOfWeek)) +
  geom_boxplot(notch = TRUE, outlier.shape = 16, outlier.alpha = 0.3) +
  scale_fill_manual(values = rep("#FF69B4", 7)) +  # consistent pink fill
  labs(
    title = "Sales Distribution by Day of the Week",
    x = "Day of the Week",
    y = "Daily Sales"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "none",  # remove legend for DayOfWeek
    panel.grid.major.y = element_line(color = "grey80"),
    panel.grid.major.x = element_blank()
  )

#1
plot_dayofweek1+plot_dayofweek2



# Histogram of raw sales
plot_sale <- ggplot(trainDat, aes(x = Sales)) +
  geom_histogram(color = "darkblue", fill = "#FF69B4", bins = 50) +
  labs(title = "Histogram of Sales",
       x = "Sales",
       y = "Count") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

# Histogram of log-transformed sales
plot_logsale <- ggplot(trainDat, aes(x = log(Sales))) +
  geom_histogram(color = "darkblue", fill = "#FF69B4", bins = 50) +
  labs(title = "Histogram of Log(Sales)",
       x = "log(Sales)",
       y = "Count") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

# 2
plot_sale + plot_logsale

# Q-Q for raw Sales
qq_sales <- ggqqplot(train$Sales, color = "#FF69B4", size = 1) +
  stat_qq_line(color = "darkblue", size = 1) +  # Baseline color here
  labs(title = "Q-Q Plot of Sales") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Q-Q for log(Sales)
qq_logsales <- ggqqplot(log(train$Sales), color = "#FF69B4", size = 1) +
  stat_qq_line(color = "darkblue", size = 1) +
  labs(title = "Q-Q Plot of Log(Sales)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

#3
qq_sales + qq_logsales



rmpse=function(preds, xgtrain) {
  labels <- getinfo(xgtrain, "label")
  elab <- exp(as.numeric(labels))
  epreds <- exp(as.numeric(preds))
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}



#-------------------------------------------------------------------------------
#model Linear Regression
#-------------------------------------------------------------------------------

#Fit model on 80% training split
lm_model <- lm(log(Sales + 1) ~ Store + DayOfWeek + Promo + SchoolHoliday + 
                 StoreType + CompetitionDistance + month + year + day, 
               data = train)

#Predict on validation set (excluding the Sales column)
valid_features <- test[, !(names(test) %in% c("Sales"))]
preds_regression <- as.numeric(exp(predict(lm_model, newdata = valid_features)) - 1)





#true values
actual_sales <- test$Sales

rmpse_regression <- sqrt(mean(((preds_regression / actual_sales) - 1)^2))

cat("Linear Regression RMSPE:", round(rmpse_regression, 4), "\n")


#-------------------------------------------------------------------------------
#Random Forest
#-------------------------------------------------------------------------------
##parameters
mtry=5
ntree=50
samplesize=nrow(train)
##Random Forest
RandomForest =randomForest(train[,Variables], 
                           log(train$Sales+1),
                           mtry=mtry,
                           ntree=ntree,
                           sampsize=samplesize,
                           do.trace=FALSE)


importance(RandomForest, type = 1)

#prediction
preds_randomforest<- as.numeric(exp(predict(RandomForest, newdata = valid_features)) - 1)


rmpse_randomforest <- sqrt(mean(((preds_randomforest / actual_sales) - 1)^2))

cat("Random Forest RMSPE:", round(rmpse_randomforest, 4), "\n")

#-------------------------------------------------------------------------------
#Xgboost
#-------------------------------------------------------------------------------
###spliting training data  20% in test and 80% training
train_integer=train[,Variables]
###set.seed(2)
Index=sample(1:nrow(train), 0.8 * (nrow(train)))
###Matrix from 
Mat_val=xgb.DMatrix(data=data.matrix(train_integer[Index,]),label=log(train$Sales+1)[Index])
Mat_train=xgb.DMatrix(data=data.matrix(train_integer[-Index,]),label=log(train$Sales+1)[-Index])
watchlist=list(val=Mat_val,train=Mat_train)
###Parameters
eta                 = 0.02
max_depth           = 10
subsample           = 0.9
colsample_bytree    = 0.7
###parameter for the best tune
Parameter = list(objective= "reg:linear", 
                 booster = "gbtree",
                 eta                 = eta,
                 max_depth           = max_depth,
                 subsample           = subsample,
                 colsample_bytree    = colsample_bytree)
###
nrounds=500
###estimate
XGBoost = xgb.train(params              = Parameter, 
                    data                = Mat_train, 
                    nrounds             = nrounds, 
                    verbose             = 0,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=rmpse)


preds_XGBoost<- exp(predict(XGBoost,  data.matrix(test[,Variables]))) - 1


rmpse_XGBoost <- sqrt(mean(((preds_XGBoost / actual_sales) - 1)^2))

cat("XGBoost RMSPE:", round(rmpse_XGBoost, 4), "\n")
