#### Purpose: GLM and NN using Keras
#### Author: Ronald Richman
#### License: MIT

# Load packages - uncomment the first line to install the CASdatasets package

#install.packages("CASdatasets",repos="http://dutangc.free.fr/pub/RRepos/", type="source")

library(CASdatasets)
library(data.table)
require(dplyr)
data(freMTPL2freq)

freMTPL2freq=freMTPL2freq %>% data.table

### correct for data errors following Noll, Salzmann and Wuthrich

freMTPL2freq[Exposure>1, Exposure:= 1]
freMTPL2freq[ClaimNb>=4, ClaimNb:= 4]

### prepare data for GLM

freMTPL2freq[,AreaGLM:=as.integer(Area)]
freMTPL2freq[,VehPowerGLM := as.factor(pmin(VehPower,9))]

VehAgeGLM=cbind(c(0:110),c(1,rep(2,10),rep(3,100)))
freMTPL2freq[,VehAgeGLM:=as.factor(VehAgeGLM[VehAge+1,2])]
freMTPL2freq[,VehAgeGLM := relevel(VehAgeGLM,ref="2")]

DrivAgeGLM=cbind(c(18:100),c(rep(1,21-18),
                             rep(2,26-21),
                             rep(3,31-26),
                             rep(4,41-31),
                             rep(5,51-41),
                             rep(6,71-51),
                             rep(7,101-71)))

freMTPL2freq[,DrivAgeGLM:=as.factor(DrivAgeGLM[DrivAge-17,2])]
freMTPL2freq[,DrivAgeGLM := relevel(DrivAgeGLM,ref="5")]

freMTPL2freq[,BonusMalusGLM :=as.integer(pmin(BonusMalus,150))]
freMTPL2freq[,DensityGLM:=as.numeric(log(Density))]
freMTPL2freq[,Region := relevel(Region,ref="R24")]

set.seed(100)
ll = sample (c (1: nrow ( freMTPL2freq )), round (0.9* nrow ( freMTPL2freq )), replace = FALSE )
learn = freMTPL2freq [ll ,]
test = freMTPL2freq [ setdiff (c (1: nrow (freMTPL2freq )), ll ),]

### Fit GLMs

null_model = glm ( formula = ClaimNb ~1 , family = poisson (), data = learn , offset = log ( Exposure ))

d.glm1 = glm ( formula = ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM +
                 BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region +
                 AreaGLM , family = poisson (), data = learn , offset = log ( Exposure ))


learn$fit <- fitted (d.glm1)
test$fit <- predict (d.glm1 , newdata =test , type ="response" )
in_sample <- 2*( sum ( learn$fit )- sum ( learn$ClaimNb )
                 + sum ( log (( learn$ClaimNb / learn$fit )^( learn$ClaimNb ))))

average_in_sample <- in_sample / nrow ( learn )

out_of_sample <- 2*( sum ( test$fit )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$fit )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test )
average_out_of_sample

results = data.table(Model = "GLM", OutOfSample = average_out_of_sample)

### Fit GLM using Keras
require(keras)

### Build one-hot encoded matrix
require(caret)

f<-ClaimNb+DensityGLM+BonusMalusGLM + AreaGLM+
  Exposure~(VehPowerGLM+VehAgeGLM+DrivAgeGLM+VehBrand+VehGas+Region)

dummies <- dummyVars(f, data = learn)
one_hot_learn = predict(dummies, newdata = learn) %>% data.table()
one_hot_test = predict(dummies, newdata = test) %>% data.table()

### Scale data to rande [0,1] for ease of fitting NN

scale_min_max = function(dat,dat_test)  {
  min_dat = min(dat)
  max_dat = max(dat)
  dat_scaled=(dat-min_dat)/(max_dat-min_dat)
  dat_scaled_test = (dat_test-min_dat)/(max_dat-min_dat)
  return(list(dat_scaled, dat_scaled_test))
}

density_glm = scale_min_max(learn$DensityGLM, test$DensityGLM)
bonus_malus_glm = scale_min_max(learn$BonusMalusGLM, test$BonusMalusGLM)
area_glm = scale_min_max(learn$AreaGLM, test$AreaGLM)

num_inputs     =data.table( DensityGLM=density_glm[[1]], 
                                BonusMalusGLM=bonus_malus_glm[[1]], 
                                AreaGLM=area_glm[[1]])

num_inputs_test     =data.table(DensityGLM=density_glm[[2]], 
                                BonusMalusGLM=    bonus_malus_glm[[2]], 
                                AreaGLM=  area_glm[[2]])


num_inputs_train = cbind(num_inputs,one_hot_learn) %>% as.matrix
num_inputs_test = cbind(num_inputs_test,one_hot_test) %>% as.matrix

x = list(NumInputs=num_inputs_train, Exposure = learn$Exposure)
x_test = list(NumInputs=num_inputs_test, Exposure = test$Exposure)

y = list(N = learn$ClaimNb)
y_test=list(N = test$ClaimNb)

############### Poisson Rate Regression
NumInputs <- layer_input(shape = c(54), dtype = 'float32', name = 'NumInputs')
Exposure <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

dense = NumInputs %>%  
  layer_dense(units = 1, activation = 'sigmoid')
  
N <- layer_multiply(list(dense,Exposure), name = 'N')

model <- keras_model(
  inputs = c(NumInputs,Exposure), 
  outputs = c(N))

model %>% compile(
  optimizer = "adadelta",
  loss = "poisson_dev")

fit = model %>% fit(
  x = x,
  y = y, 
  epochs = 10,
  batch_size = 256,verbose = 1, shuffle = T)

learn$GLM = model %>% predict(x)
test$GLM = model %>% predict(x_test)


out_of_sample <- 2*( sum ( test$GLM )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$GLM )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test)

results = rbind(results,
                data.table(Model = "GLM_Keras", OutOfSample = average_out_of_sample))

############### Shallow NN
NumInputs <- layer_input(shape = c(54), dtype = 'float32', name = 'NumInputs')
Exposure <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

dense = NumInputs %>%
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

N <- layer_multiply(list(dense,Exposure), name = 'N')

model <- keras_model(
  inputs = c(NumInputs,Exposure), 
  outputs = c(N))

model %>% compile(
  optimizer = "adadelta",
  loss = "poisson_dev")

fit = model %>% fit(
  x = x,
  y = y, 
  epochs = 10,
  batch_size = 256,verbose = 1, shuffle = T)

learn$NN_shallow = model %>% predict(x)
test$NN_shallow = model %>% predict(x_test)

out_of_sample <- 2*( sum ( test$NN_shallow )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$NN_shallow )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test)

average_out_of_sample


results = rbind(results,
                data.table(Model = "NN_shallow_Keras", OutOfSample = average_out_of_sample))

### NN - no feature engineering
f<-ClaimNb+Density+BonusMalus + AreaGLM+VehPower+ VehAge+ DrivAge+
  Exposure~(VehBrand+VehGas+Region)

dummies <- dummyVars(f, data = learn)
one_hot_learn = predict(dummies, newdata = learn) %>% data.table()
one_hot_test = predict(dummies, newdata = test) %>% data.table()

density = scale_min_max(learn$Density, test$Density)
bonus_malus = scale_min_max((learn$BonusMalus), (test$BonusMalus))
area_glm = scale_min_max(learn$AreaGLM, test$AreaGLM)
vehPower = scale_min_max(learn$VehPower, test$VehPower)
vehAge = scale_min_max((learn$VehAge), (test$VehAge))
drivAge = scale_min_max(learn$DrivAge, test$DrivAge)

num_inputs     =data.table( Density=density[[1]], 
                            BonusMalus=bonus_malus[[1]], 
                            AreaGLM=area_glm[[1]], 
                            VehPower = vehPower[[1]],
                            VehAge = vehAge[[1]],
                            DrivAge = drivAge[[1]])

num_inputs_test     =data.table(Density=density[[2]], 
                                BonusMalus=    bonus_malus[[2]], 
                                AreaGLM=  area_glm[[2]], 
                                VehPower = vehPower[[2]],
                                VehAge = vehAge[[2]],
                                DrivAge = drivAge[[2]])

num_inputs_train = cbind(num_inputs,one_hot_learn) %>% as.matrix
num_inputs_test = cbind(num_inputs_test,one_hot_test) %>% as.matrix

x = list(NumInputs=num_inputs_train, Exposure = learn$Exposure)
x_test = list(NumInputs=num_inputs_test, Exposure = test$Exposure)

y = list(N = learn$ClaimNb)
y_test=list(N = test$ClaimNb)

############### Poisson Rate Regression
NumInputs <- layer_input(shape = c(41), dtype = 'float32', name = 'NumInputs')
Exposure <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

dense = NumInputs %>%  
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.1) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(0.1) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

N <- layer_multiply(list(dense,Exposure), name = 'N')

model <- keras_model(
  inputs = c(NumInputs,Exposure), 
  outputs = c(N))

model %>% compile(
  optimizer = "adadelta",
  loss = "poisson_dev")

fit = model %>% fit(
  x = x,
  y = y, validation_data = list(x_test, y_test) ,
  epochs = 15,
  batch_size = 256,verbose = 1, shuffle = T)

learn$NN_no_FE = model %>% predict(x)
test$NN_no_FE = model %>% predict(x_test)

out_of_sample <- 2*( sum ( test$NN_no_FE )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$NN_no_FE )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test )

average_out_of_sample

results = rbind(results,
                data.table(Model = "NN_no_FE", OutOfSample = average_out_of_sample))

results %>% fwrite("c:/r/NL_Pricing_GLM.csv")