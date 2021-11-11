## Project part 3 a

## Project
RNGkind(sample.kind = "Rounding")
## load file


admission = read.csv("C:\\Users\\thera\\OneDrive\\Desktop\\Fall 2020\\MATH 4323\\MATH 4323\\Project\\Admission_Predict.csv")

head(admission)

dim(admission)

attach(admission)

### 3) Data analysis

#### a) Preprocessing/cleaning data

# plots 
par(mfrow=c(2,4))
plot(Chance.of.Admit ~ GRE.Score, col= 'blue')
plot(Chance.of.Admit ~ TOEFL.Score, col= 'blue')
plot(Chance.of.Admit ~ University.Rating, col= 'blue')
plot(Chance.of.Admit ~ SOP, col= 'blue')
plot(Chance.of.Admit ~ LOR, col= 'blue')
plot(Chance.of.Admit ~ CGPA, col= 'blue')
plot(Chance.of.Admit ~ Research, col= 'blue')

# add new variable admitmedian01
admission$admitmedian01 <- ifelse(admission$Chance.of.Admit > median(admission$Chance.of.Admit), 1, 0)
class(admission$admitmedian01)

admission$admitmedian01 <- as.factor(admission$admitmedian01)
class(admission$admitmedian01)

# deleting chance.of.admit and serial.no, and Research columns
admission$Chance.of.Admit <- NULL
admission$Serial.No. <- NULL
admission$Research <- NULL

head(admission)
dim(admission)
attach(admission)



# plots
par(mfrow=c(2,3))
plot(GRE.Score ~ admitmedian01)
plot(TOEFL.Score ~ admitmedian01)
plot(University.Rating ~ admitmedian01)
plot(SOP ~ admitmedian01)
plot(LOR ~ admitmedian01)
plot(CGPA ~ admitmedian01)
#plot(Research ~ admitmedian01)

## scaling
set.seed(1)

admission.scaled = data.frame(scale(admission[-7]))

admission.scaled <- cbind(admission.scaled, admitmedian01 = admission$admitmedian01)


#10 fold cross validation for knn

library(caret)
library(e1071)
set.seed(1)
index <- sample(400, 320)
train_df <- admission.scaled[index, ]
test_df <- admission.scaled[-index, ]


ctrlspecs <- trainControl(method="cv", number=10)
set.seed(1)
model1 <- train(admitmedian01~GRE.Score+TOEFL.Score+University.Rating+SOP + LOR+CGPA, 
                data = train_df,
                method = "knn",  tuneGrid   = expand.grid(k = 1:30),
                trControl  = ctrlspecs,
                metric     = "Accuracy")
model1
knn.10cross.predictions <- predict(model1, newdata = test_df)
test.err <- mean(knn.10cross.predictions != test_df$admitmedian01)
test.err


# 10 fold cross validation for SVM polynomial
set.seed(1)
train=sample(400,320)
set.seed(1)
tune.polyn <- tune(svm,
                   admitmedian01~ ., 
                   data = admission.scaled[train,],
                   kernel="polynomial",
                   ranges = list(cost = c(0.001,0.01, 0.1, 1, 5, 10, 100),
                                 degree = c(2,3,4)))
summary(tune.polyn)
print(tune.polyn$best.parameters)#optimal cost value
print(tune.polyn$best.performance)#optimal model's cross-validation error

mean(predict(tune.polyn$best.model, newdata=admission.scaled[-train,]) != admission.scaled$admitmedian01[-train])

#10 fold cross validation for svm radial

set.seed(1)
train=sample(400,320)
set.seed(1)
tune.obj.radial <- tune(svm,
                        admitmedian01 ~ ., 
                        data = admission.scaled[train,],
                        kernel="radial",
                        ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100),gamma=c(0.1, 0.2, 0.3, 0.4,0.5,0.7, 1,2,3,4)))


tune.obj.radial
summary(tune.obj.radial)

print(tune.obj.radial)
print(tune.obj.radial$best.parameters)#optimal cost value
print(tune.obj.radial$best.performance)#optimal model's cross-validation error
mean(predict(tune.obj.radial$best.model) != admission.scaled$admitmedian01[train])
mean(predict(tune.obj.radial$best.model, newdata=admission.scaled[-train,]) != admission.scaled$admitmedian01[-train])


#10 fold cross validation for svm linear
set.seed(1)
train <- sample(400,320)
set.seed(1)
tune.lin <- tune(svm,
                 admitmedian01~ ., 
                 data = admission.scaled[train,],
                 kernel="linear",
                 ranges = list(cost = c(0.001,0.01, 0.1, 1, 5, 10, 100)))

tune.lin
summary(tune.lin)
print(tune.lin$best.parameters)
print(tune.lin$best.performance)

mean(predict(tune.lin$best.model, newdata=admission.scaled[-train,]) != admission.scaled$admitmedian01[-train])

#best model is svm polynomial with cost = 100 and degree = 3

#Fit best model on full data set and show summaries of svm object and support vectors and also show plots
set.seed(1)
svm.polyn <- svm(admitmedian01~ ., 
                   data = admission.scaled,
                   kernel="polynomial", cost =100, degree = 3)
summary(svm.polyn)

svm.pred.poly <- predict(svm.polyn, newdata = admission.scaled)
mean(svm.pred.poly != admission.scaled$admitmedian01)


mean(predict(svm.polyn, newdata=admission.scaled) != admission.scaled$admitmedian01)






head(admission.scaled)
attach(admission.scaled)
dev.off()

plot(svm.polyn, admission.scaled, GRE.Score ~ CGPA)
plot(svm.polyn, admission.scaled, TOEFL.Score ~ CGPA)
plot(svm.polyn, admission.scaled, University.Rating ~ CGPA)
plot(svm.polyn, admission.scaled, GRE.Score ~ SOP)
plot(svm.polyn, admission.scaled, CGPA ~ LOR)
plot(svm.polyn, admission.scaled, University.Rating ~ SOP)
plot(svm.polyn, admission.scaled, TOEFL.Score ~ LOR)
plot(svm.polyn, admission.scaled, GRE.Score ~ TOEFL.Score)
plot(svm.polyn, admission.scaled, SOP ~ LOR)

