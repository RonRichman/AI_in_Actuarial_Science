#### Purpose: NN using Keras
#### Author: Ronald Richman
#### License: MIT

# Load packages - uncomment the first line to install the CASdatasets package

#install.packages("CASdatasets",repos="http://dutangc.free.fr/pub/RRepos/", type="source")
library(CASdatasets)
library(data.table)
require(dplyr)
require(Hmisc)
require(caret)
data(freMTPL2freq)

freMTPL2freq=freMTPL2freq %>% data.table

### correct for data errors

freMTPL2freq[Exposure>1, Exposure:= 1]
freMTPL2freq[ClaimNb>=4, ClaimNb:= 4]

### prepare data for Embeddings
freMTPL2freq[,VehPowerNN    :=as.integer(as.factor(VehPower))-1]
freMTPL2freq[,VehAgeNN      :=as.integer(as.factor(cut2(VehAge,m=10000)))-1]
freMTPL2freq[,DrivAgeNN     :=as.integer(as.factor(cut2(DrivAge,m=10000)))-1]
freMTPL2freq[,BonusMalusNN  :=as.integer(as.factor(cut2(BonusMalus,m=10000)))-1]
freMTPL2freq[,DensityNN     :=as.integer(as.factor(cut2(Density,m=10000)))-1]
freMTPL2freq[,VehGasNN      :=as.integer(as.factor(VehGas))-1]
freMTPL2freq[,RegionNN      :=as.integer(as.factor(Region))-1]
freMTPL2freq[,VehBrandNN    :=as.integer(as.factor(VehBrand))-1]
freMTPL2freq[,AreaNN        :=as.integer(as.factor(Area))-1]
freMTPL2freq[,ExposureNN    :=as.integer(as.factor(cut2(Exposure,m=10000)))-1]

### prepare dataframe for embeddings

veh_power_cats  = freMTPL2freq[,c("VehPowerNN","VehPower"),with=F] %>% unique %>% setkey(VehPowerNN)
veh_age_cats    = freMTPL2freq[,.(VehAge = mean(VehAge)), keyby = VehAgeNN]
driv_age_cats   = freMTPL2freq[,.(Drivage = mean(DrivAge)), keyby = DrivAgeNN]
bon_mal_cats    = freMTPL2freq[,.(BonusMalus = mean(BonusMalus)), keyby = BonusMalusNN]
density_cats    = freMTPL2freq[,.(Density = mean(Density)), keyby = DensityNN]
veh_gas_cats    = freMTPL2freq[,c("VehGasNN","VehGas"),with=F] %>% unique %>% setkey(VehGasNN)
region_cats     = freMTPL2freq[,c("RegionNN","Region"),with=F] %>% unique %>% setkey(RegionNN)
veh_brand_cats  = freMTPL2freq[,c("VehBrandNN","VehBrand"),with=F] %>% unique %>% setkey(VehBrandNN)
area_cats       = freMTPL2freq[,c("AreaNN","Area"),with=F] %>% unique %>% setkey(AreaNN)
exposure_cats   =  freMTPL2freq[,.(Exposure = mean(Exposure)), keyby = ExposureNN]

veh_age_dim = veh_age_cats[,.N]
driv_age_dim = driv_age_cats[,.N]
bon_mal_dim = bon_mal_cats[,.N]
density_dim = density_cats[,.N]
exposure_dim = exposure_cats[,.N]

freMTPL2freq[,id_cats:=interaction(BonusMalusNN,DrivAgeNN )]

set.seed(100)
splits = createDataPartition(freMTPL2freq$id_cats,p=0.9)

learn = freMTPL2freq [c(splits$Resample1) ,]
test = freMTPL2freq [ setdiff (c (1: nrow (freMTPL2freq )), splits$Resample1 ),]

### Check

learn[,.N]+test[,.N]-freMTPL2freq[,.N]

### Fit NN using Keras
require(keras)

### Munge data into correct format for embeddings - Train set
veh_gas       =  learn$VehGasNN
veh_power     =  learn$VehPowerNN  
veh_age       =  learn$VehAgeNN
driv_age      =  learn$DrivAgeNN
veh_brand     =  learn$VehBrandNN     
region        =  learn$RegionNN   
bon_mal       =  learn$BonusMalusNN      
density       =  learn$DensityNN  
area          =  learn$AreaNN
exposure      =  learn$ExposureNN

x = list(Exposure      = learn$Exposure,
         VehGas        = veh_gas,
         VehPower      = veh_power,
         VehAge        = veh_age,
         DrivAge       = driv_age,
         VehBrand      = veh_brand,
         Region        = region,
         Area          = area,
         BonMal        = bon_mal, 
         Density       = density  ,
         ExposureNN = exposure)

y = list(N = learn$ClaimNb)

### Munge data into correct format for embeddings - test set
veh_gas       =  test$VehGasNN
veh_power     =  test$VehPowerNN  
veh_age       =  test$VehAgeNN
driv_age      =  test$DrivAgeNN
veh_brand     =  test$VehBrandNN     
region        =  test$RegionNN   
bon_mal       =  test$BonusMalusNN      
density       =  test$DensityNN  
area          =  test$AreaNN
exposure      =  test$ExposureNN

x_test = list(Exposure = test$Exposure,
              VehGas=veh_gas,
              VehPower=veh_power,
              VehAge=veh_age,
              DrivAge=driv_age,
              VehBrand=veh_brand,
              Region=region,
              Area=area,
              BonMal = bon_mal, 
              Density = density,ExposureNN = exposure)

y_test = list(N = test$ClaimNb)

############### Build embedding layers
Exposure <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

VehGas <- layer_input(shape = c(1), dtype = 'int32', name = 'VehGas')
VehGas_embed = VehGas %>% 
  layer_embedding(input_dim = 2, output_dim = 1,input_length = 1, name = 'VehGas_embed') %>%
  keras::layer_flatten()

VehPower <- layer_input(shape = c(1), dtype = 'int32', name = 'VehPower')
VehPower_embed = VehPower %>% 
  layer_embedding(input_dim = 12, output_dim = 2,input_length = 1, name = 'VehPower_embed') %>%
  keras::layer_flatten()

VehAge <- layer_input(shape = c(1), dtype = 'int32', name = 'VehAge')
VehAge_embed = VehAge %>% 
  layer_embedding(input_dim = veh_age_dim, output_dim = trunc(veh_age_dim/4),input_length = 1, name = 'VehAge_embed') %>%
  keras::layer_flatten()

DrivAge <- layer_input(shape = c(1), dtype = 'int32', name = 'DrivAge')
DrivAge_embed = DrivAge %>% 
  layer_embedding(input_dim = driv_age_dim, output_dim = trunc(driv_age_dim/4),input_length = 1, name = 'DrivAge_embed') %>%
  keras::layer_flatten()

VehBrand <- layer_input(shape = c(1), dtype = 'int32', name = 'VehBrand')
VehBrand_embed = VehBrand %>% 
  layer_embedding(input_dim = 11, output_dim = 5,input_length = 1, name = 'VehBrand_embed') %>%
  keras::layer_flatten()

Region <- layer_input(shape = c(1), dtype = 'int32', name = 'Region')
Region_embed = Region %>% 
  layer_embedding(input_dim = 22, output_dim = 5,input_length = 1, name = 'Region_embed') %>%
  keras::layer_flatten()

Area <- layer_input(shape = c(1), dtype = 'int32', name = 'Area')
Area_embed = Area %>% 
  layer_embedding(input_dim = 6, output_dim = 2,input_length = 1, name = 'Area_embed') %>%
  keras::layer_flatten()

BonMal <- layer_input(shape = c(1), dtype = 'int32', name = 'BonMal')
BonMal_embed = BonMal %>% 
  layer_embedding(input_dim = bon_mal_dim, output_dim = trunc(bon_mal_dim/4),input_length = 1, name = 'BonMal_embed') %>%
  keras::layer_flatten()

Density <- layer_input(shape = c(1), dtype = 'int32', name = 'Density')
Density_embed = Density %>% 
  layer_embedding(input_dim = density_dim, output_dim = trunc(density_dim/4) ,input_length = 1, name = 'Density_embed') %>%
  keras::layer_flatten()

ExposureNN <- layer_input(shape = c(1), dtype = 'int32', name = 'ExposureNN')
ExposureNN_embed = ExposureNN %>% 
  layer_embedding(input_dim = exposure_dim, output_dim = trunc(exposure_dim/4),input_length = 1) %>%
  keras::layer_flatten()

middle_layer <- layer_concatenate(list(VehGas_embed, 
                                      VehBrand_embed,
                                      VehPower_embed,
                                      VehAge_embed,
                                      DrivAge_embed,
                                      VehBrand_embed,
                                      Region_embed,Area_embed, BonMal_embed, Density_embed,ExposureNN_embed)) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dense(units = 32, activation = 'relu')%>%  
  layer_dropout(0.2, name='features')


  main_output = middle_layer  %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

  exposure_modified = main_output %>% 
    layer_dense(units = 4, activation = 'relu') %>% 
    layer_dropout(0.05) %>% 
    layer_dense(units = 4, activation = 'relu')%>%  
    layer_dropout(0.05) %>% 
    layer_dense(units = 1, activation = k_exp)
  exposure_modified= layer_multiply(list(Exposure,exposure_modified), name = "exposure_modified")

  N <- layer_multiply(list(main_output,exposure_modified), name = 'N')
  
model <- keras_model(
  inputs = c(Exposure,VehGas, VehPower, VehAge,DrivAge,VehBrand,Region,Area,BonMal, Density,ExposureNN), 
  outputs = c(N))

model_exp <- keras_model(
  inputs = c(VehGas, VehPower, VehAge,DrivAge,VehBrand,Region,Area,BonMal, Density,ExposureNN,Exposure), 
  outputs = c(exposure_modified))

model %>% compile(
  optimizer = "adadelta",
  loss = list("poisson_dev"))

model_exp %>% compile(
  optimizer = "adadelta",
  loss = list("poisson_dev"))

fit = model %>% fit(
  x = x,
  y = y, validation_data = list(x_test,y_test),
  epochs = 20,
  batch_size = 256,verbose = 1, shuffle = T)

#model %>% save_model_weights_hdf5("c:/r/embed_model_weights_exp.mod")

model = load_model_weights_hdf5(model, "c:/r/embed_model_weights_exp.mod")

learn$NN_embed_exp = model %>% predict(x)
test$NN_embed_exp = model %>% predict(x_test)

learn$learned_exp = model_exp %>% predict(x)
test$learned_exp = model_exp %>% predict(x_test)


out_of_sample <- 2*( sum ( test$NN_embed_exp )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$NN_embed_exp )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test )

average_out_of_sample


results = rbind(results,
                data.table(Model = "NN_learned_embed", OutOfSample = average_out_of_sample))

results %>% fwrite("c:/r/nnembed_exp.csv")

test[,`Change in Exposure`:=learned_exp/Exposure]

test[,.(`Learned Exposure` = mean(learned_exp)), keyby=.(ClaimNb,Exposure)] %>% 
  ggplot(aes(x=Exposure,y=`Learned Exposure`))+
  geom_line(aes(group = ClaimNb,linetype=as.factor(ClaimNb), colour=as.factor(ClaimNb )))+
  theme_pubr()
ggsave("c:/r/learned_exp.wmf", device = "wmf", width = 10,height = 10)
