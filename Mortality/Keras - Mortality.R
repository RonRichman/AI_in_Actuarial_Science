#### Purpose: Fit autoencoders to HMD data
#### Author: Ronald Richman
#### License: MIT
#### Data: The data were sourced from the HMD by downloading the relevant text files

require(data.table)
require(dplyr)
require(ggplot2)
require(data.table)
require(reshape2)
require(HMDHFDplus)
require(gnm)

### Get England/Wales data from StMoMo
dat = HMDHFDplus::readHMD("c:/r/Mx_1x1.txt") %>% data.table

### transform data
dat[,logmx:=log(Male)]

scale_min_max = function(dat,dat_test)  {
  min_dat = min(dat)
  max_dat = max(dat)
  dat_scaled=(dat-min_dat)/(max_dat-min_dat)
  dat_scaled_test = (dat_test-min_dat)/(max_dat-min_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, min = min_dat, max=max_dat))
}

scale_z = function(dat,dat_test)  {
  mean_dat = mean(dat)
  sd_dat = sd(dat)
  dat_scaled=(dat-mean_dat)/(sd_dat)
  dat_scaled_test = (dat_test-mean_dat)/(sd_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, mean_dat = mean_dat, sd_dat=sd_dat))
}

dat = dat[Year>1949 & Age<100]
train = dat[Year < 2000]
test = dat[Year >= 2000]

scaled = scale_min_max(train$logmx, test$logmx)

train$mx_scale = scaled$train
test$mx_scale = scaled$test

train_rates = train %>% dcast.data.table(Year~Age, value.var = "mx_scale")

### Lee-Carter Baseline - Fit on Raw Rates
fit = gnm(Male~ as.factor(Age) + Mult(as.factor(Age), as.factor(Year), inst = 1) -1,family = poisson(link = "log"),data=train)

train[,pred_LC:=predict(fit, type="response")]

coefs = data.table(names = fit %>% coef %>% names, coef=fit %>% coef)
coefs[,row:=.I]
coefs[row %in% c(1:100),var:="ax"]
coefs[row %in% c(101:200),var:="bx"]
coefs[row %in% c(201:250),var:="k"]

ax =coefs[var == "ax"]$coef
bx =coefs[var == "bx"]$coef
k =coefs[var == "k"]$coef

c1 = mean(k)
c2 = sum(bx)
ax = ax+c1*bx
bx = bx/c2
k = (k-c1)*c2

forecast_k  =k %>% forecast::forecast(17)
k_forecast = forecast_k[[2]]

fitted = (ax+(bx)%*%t(k))
fitted_test = (ax+(bx)%*%t(k_forecast)) %>% melt
test$pred_LC =   fitted_test$value %>% exp
results = data.table(model = "Lee-Carter", MSE_OutOfSample = 
                       test[,sum((Male - pred_LC)^2)])

test[Year==2016]%>% ggplot(aes(x=Age, y = log(pred_LC)))+ geom_point(size = 0.5, alpha=0.5)+facet_wrap(~Year)+geom_line(aes(x=Age, y=log(Male)))

##### Simple auto encoder
require(keras)

x = list(NumInputs=as.matrix(train_rates[,c(-1),with=F]), 
         Rates= as.matrix(train_rates[,c(-1),with=F]))

NumInputs <- layer_input(shape = c(100), dtype = 'float32', name = 'NumInputs')

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1")


Rates = encode %>%  
  layer_dense(units = 100, activation = 'sigmoid', name = "Rates") 

model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Rates))

adam = optimizer_adam(lr = 0.001)

model %>% compile(
  optimizer = adam,
  loss = "mape")

lr_schedule = function(n,lr) ifelse(n<20,0.01,ifelse(n<40000,0.001,0.0001))
lr_sched = callback_learning_rate_scheduler(schedule = lr_schedule)

# fit = model %>% fit(
#   x = x,
#   y = x, 
#   epochs = 50000,
#   batch_size = 8,verbose = 1, shuffle = T, callbacks = list(lr_sched))
# 
# model %>% save_model_hdf5("c:/r/greedy_auto_encode1.mod")
model = load_model_hdf5("c:/r/greedy_auto_encode1.mod")

### Step 2

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1") %>% 
  layer_dense(units = 2, activation = 'linear', name = "encode")

Rates = encode %>%  
  layer_dense(units =4, activation = 'tanh', name = "DecodeRates1") %>% 
  layer_dense(units = 100, activation = 'sigmoid', name = "Rates") 

model_stage2 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Rates))

adam = optimizer_adam(lr = 0.001)

set_weights(get_layer(model_stage2, "EncodeRates1"), get_weights(get_layer(model, "EncodeRates1")))
set_weights(get_layer(model_stage2, "Rates"), get_weights(get_layer(model, "Rates")))

layer = get_layer(model_stage2, "EncodeRates1")
layer$trainable = FALSE

layer = get_layer(model_stage2, "Rates")
layer$trainable = FALSE

lr_schedule = function(n,lr) ifelse(n<20,0.01,ifelse(n<40000,0.001,0.0001))
lr_sched = callback_learning_rate_scheduler(schedule = lr_schedule)

model_stage2 %>% compile(
  optimizer = adam,
  loss = "mape")
# 
# fit = model_stage2 %>% fit(x = x,
#   y = x, 
#   epochs = 50000,
#   batch_size = 8,verbose = 1, shuffle = T, callbacks = list(lr_sched))
# 
# model_stage2 %>% save_model_hdf5("c:/r/greedy_auto_encode2.mod")
model_stage2 = load_model_hdf5("c:/r/greedy_auto_encode2.mod")

### Step 3

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1") %>% 
  layer_dense(units = 2, activation = 'linear', name = "encode")

Rates = encode %>%  
  layer_dense(units =4, activation = 'tanh', name = "DecodeRates1") %>% 
  layer_dense(units = 100, activation = 'sigmoid', name = "Rates") 

model_stage3 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Rates))

adam = optimizer_adam(lr = 0.001)

set_weights(get_layer(model_stage3, "EncodeRates1"), get_weights(get_layer(model_stage2, "EncodeRates1")))
set_weights(get_layer(model_stage3, "encode"), get_weights(get_layer(model_stage2, "encode")))
set_weights(get_layer(model_stage3, "DecodeRates1"), get_weights(get_layer(model_stage2, "DecodeRates1")))
set_weights(get_layer(model_stage3, "Rates"), get_weights(get_layer(model_stage2, "Rates")))


lr_schedule = function(n,lr) ifelse(n<20,0.01,ifelse(n<40000,0.001,0.0001))
lr_sched = callback_learning_rate_scheduler(schedule = lr_schedule)


model_stage3 %>% compile(
  optimizer = adam,
  loss = "mse")

# fit = model_stage3 %>% fit(x = x,
#                            y = x,
#                            epochs = 50000,
#                            batch_size = 8,verbose = 1, shuffle = T, callbacks = list(lr_sched))
# 
# model_stage3 %>% save_model_hdf5("c:/r/greedy_auto_encode3.mod")

model_stage3=load_model_hdf5("c:/r/greedy_auto_encode3.mod")

model = model_stage3

require(stringr)

pred_train = model %>% predict(x$NumInputs) %>% data.table
pred_train[,Year:=.I]
pred_train = pred_train %>% melt(id.vars="Year") %>% data.table()
pred_train[,Age:=as.integer(str_sub(variable, 2))-1]
pred_train[,mx_autoencode:=value]
pred_train %>% setkey(Year,Age)

train$mx_autoencode = pred_train$mx_autoencode
train[,mx_autoencode:=exp(mx_autoencode*(scaled$max-scaled$min)+scaled$min)]
train[,.(Autoencode = sum((Male-mx_autoencode)^2)/.N, LC = sum((Male-pred_LC)^2)/.N)]
train_metrics = train[,.(Autoencode = sum((Male-mx_autoencode)^2), LC = sum((Male-pred_LC)^2)), keyby = Year]
train_metrics[, delta := Autoencode-LC]
train_metrics %>% ggplot(aes(x=Year, y=delta))+ geom_point()

### get codes

codes_model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(encode)) %>% compile(
    optimizer = adam,
    loss = "mape")

codes =  codes_model %>% predict(x$NumInputs) %>% data.table
codes[,Year:= train[,unique(Year)]]

require(forecast)

ts_mat = as.ts(codes[,c(1:2),with=F])

dim1_forecasts = ts_mat[,1] %>% forecast::rwf(h = 17,drift=T) %>% forecast(17)
dim2_forecasts = ts_mat[,2] %>% forecast::rwf(h=17,drift=T) %>% forecast(17)

codes_forecast=data.table(V1 = as.double(dim1_forecasts$mean),
                          V2 = as.double(dim2_forecasts$mean),
                          Year=2000:2016)

codes = rbind(codes, codes_forecast)

### forecasts

CodeInputs <- layer_input(shape = c(2), dtype = 'float32', name = 'CodeInputs')

DecodedRates = CodeInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "DecodeRates1") %>% 
  layer_dense(units = 100, activation = 'sigmoid', name = "DecodedRates")

decode_model <- keras_model(
  inputs = c(CodeInputs), 
  outputs = c(DecodedRates)) %>% compile(
    optimizer = adam,
    loss = "mape")

set_weights(get_layer(decode_model, "DecodeRates1"), get_weights(get_layer(model, "DecodeRates1")))
set_weights(get_layer(decode_model, "DecodedRates"), get_weights(get_layer(model, "Rates")))

x_codes = list(encode = codes[,c(1,2),with=F] %>% as.matrix)

codes =  decode_model %>% predict(x_codes$encode) %>% data.table

codes[,Year:=.I]
codes = codes %>% melt(id.vars="Year") %>% data.table()
codes[,Age:=as.integer(str_sub(variable, 2))-1]
codes[,Year:=Year+1949]
codes[,mx_autoencode:=value]
codes[,mx_nn:=exp(mx_autoencode*(scaled$max-scaled$min)+scaled$min)]

codes %>% setkey(Year,Age)

test$mx_autoencode = codes[Year>=2000]$mx_autoencode
test[,mx_autoencode:=exp(mx_autoencode*(scaled$max-scaled$min)+scaled$min)]
test[,sum((Male-mx_autoencode)^2)]
test[,sum((Male - pred_LC)^2)]

results= rbind(results, data.table(model = "Autoencoder", MSE_OutOfSample=test[,sum((Male-mx_autoencode)^2)]))

test[Year==2016]%>% ggplot(aes(x=Age, y = log(pred_LC)))+ geom_point(size = 0.5, alpha=0.5)+facet_wrap(~Year)+geom_line(aes(x=Age, y=log(Male)))

#### Regression

train_reg = train[,c(1,2,4,8),with=F]
test_reg = test[,c(1,2,4,8),with=F]

year_scale = scale_min_max(train_reg$Year,test_reg$Year)

#train

train_reg$Year = year_scale[[1]]
test_reg$Year = year_scale[[2]]

Year       =  train_reg$Year
Age        =  train_reg$Age 

x = list(Year      = Year,
         Age = Age)

y = (main_output= train_reg$mx_scale)

#test

Year       =  test_reg$Year
Age        =  test_reg$Age 

x_test = list(Year      = Year,
         Age = Age)

y_test = (main_output= test_reg$mx_scale)

############### Build embedding layers
Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year')
Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
Age_embed = Age %>% 
  layer_embedding(input_dim = 100, output_dim = 5,input_length = 1, name = 'Age_embed') %>%
  keras::layer_flatten()

main_output <- layer_concatenate(list(Year, 
                                      Age_embed)) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(0.1) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(0.1) %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

model <- keras_model(
  inputs = c(Year,Age), 
  outputs = c(main_output))

adam = optimizer_adam(lr=0.001)
lr_callback = callback_reduce_lr_on_plateau(factor=.80, patience = 50, verbose=1, cooldown = 100, min_lr = 0.0001)
model_callback = callback_model_checkpoint(filepath = "c:/r/best_mx_reg_callback.mod", verbose = 1,save_best_only = TRUE)

model %>% compile(
  optimizer = adam,
  loss = "mse")

### run to end

fit = model %>% fit(
  x = x,
  y = y, 
  epochs = 1500,
  batch_size = 16,verbose = 1, shuffle = T, validation_split = 0.1, callbacks = list(lr_callback,model_callback))


#model %>%  save_model_hdf5("c:/r/best_mx_reg.mod")

model = load_model_hdf5("c:/r/best_mx_reg.mod")

test$mx_deep_reg = model %>% predict(x_test)
test[,mx_deep_reg:=exp(mx_deep_reg*(scaled$max-scaled$min)+scaled$min)]
test[,.(lc = sum((Male-pred_LC)^2),
        autoencode_forecast= sum((Male-mx_autoencode)^2),
        Deep_reg = sum((Male-mx_deep_reg)^2))
     ]

results = rbind(results, data.table(model = "Deep_reg", MSE_OutOfSample =
                                      test[,sum((Male-mx_deep_reg)^2)]))

results %>% fwrite("c:/r/mort_model.csv")

gbrtenw = fread("c:/r/gbrtenw.csv")

Y2016 = test[Year==2016]
Y2016$mx_deep_reg_all = gbrtenw$mx_deep_reg_full

Y2016[,.(lc = sum((Male-pred_LC)^2),
        autoencode_forecast= sum((Male-mx_autoencode)^2),
        Deep_reg = sum((Male-mx_deep_reg)^2),
       Deep_reg_all = sum((Male-mx_deep_reg_all)^2))
     ]


Y2016[,c(2,4,9:12),with=F] %>% melt(id.vars="Age") %>% 
  ggplot(aes(x=Age, y=log(value)))+geom_line(aes(group = variable,
                                                 colour=variable,
                                                 linetype=variable))+
  geom_point(aes(colour=variable,
                shape=variable))+
  theme_pubr()

ggsave("c:/r/mx2016.wmf", device = "wmf")