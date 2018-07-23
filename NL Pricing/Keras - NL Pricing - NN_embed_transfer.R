#### Purpose: GLM and NN using Keras
#### Author: Ronald Richman
#### License: MIT

# Load packages - uncomment the first line to install the CASdatasets package


#install.packages("CASdatasets",repos="http://dutangc.free.fr/pub/RRepos/", type="source")
require(CASdatasets)
require(data.table)
require(dplyr)
require(Hmisc)
require(caret)
require(ggpubr)
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

freMTPL2freq[,id_cats:=interaction(BonusMalusNN,DrivAgeNN )]

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

veh_age_dim = veh_age_cats[,.N]
driv_age_dim = driv_age_cats[,.N]
bon_mal_dim = bon_mal_cats[,.N]
density_dim = density_cats[,.N]

embedding_dat = list(VehPower_embed = veh_power_cats,
                     VehAge_embed = veh_age_cats  ,
                     DrivAge_embed = driv_age_cats ,
                     BonMal_embed = bon_mal_cats  ,
                     Density_embed = density_cats  ,
                     VehGas_embed =  veh_gas_cats  ,
                     Region_embed = region_cats   ,
                     VehBrand_embed = veh_brand_cats,
                     Area_embed = area_cats      )

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

x = list(Exposure      = learn$Exposure,
         VehGas        = veh_gas,
         VehPower      = veh_power,
         VehAge        = veh_age,
         DrivAge       = driv_age,
         VehBrand      = veh_brand,
         Region        = region,
         Area          =area,
         BonMal        = bon_mal, 
         Density       = density 
         )

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

x_test = list(
  
         Exposure = test$Exposure,
         VehGas=veh_gas,
         VehPower=veh_power,
         VehAge=veh_age,
         DrivAge=driv_age,
         VehBrand=veh_brand,
         Region=region,
         Area=area,
         BonMal = bon_mal, 
         Density = density)

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
  layer_embedding(input_dim = density_dim, output_dim = trunc(density_dim/4),input_length = 1, name = 'Density_embed') %>%
  keras::layer_flatten()


main_output <- layer_concatenate(list(VehGas_embed, 
                                      VehBrand_embed,
                                      VehPower_embed,
                                      VehAge_embed,
                                      DrivAge_embed,
                                      VehBrand_embed,
                                      Region_embed,Area_embed, BonMal_embed, Density_embed
                                      )) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(0.1) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(0.1) %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

N <- layer_multiply(list(main_output,Exposure), name = 'N')

model <- keras_model(
  inputs = c(Exposure,VehGas, VehPower, VehAge,DrivAge,VehBrand,Region,Area,BonMal, Density), 
  outputs = c(N))

model %>% compile(
  optimizer = "adadelta",
  loss = "poisson_dev")

fit = model %>% fit(
  x = x,
  y = y, validation_data = list(x_test,y_test),
  epochs = 30,
  batch_size = 256,verbose = 1, shuffle = T)

#model %>% save_model_hdf5(c:/r/embed_model.mod")

#model %>% save_model_weights_hdf5("c:/r/embed_model_weights.mod")
model = load_model_hdf5("c:/r/embed_model.mod")

learn$NN_embed = model %>% predict(x)
test$NN_embed = model %>% predict(x_test)

out_of_sample <- 2*( sum ( test$NN_embed )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$NN_embed )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test)

average_out_of_sample

results = data.table(Model = "NN_embed", OutOfSample = average_out_of_sample)


################### Visualize embeddings

get_embedding_values = function(layer_name){
embedding = model %>% get_layer(layer_name) %>% get_weights()
temp = embedding[[1]] %>% data.table()
temp%>% setnames(names(temp),paste0(layer_name,"dim",seq(1:length(names(temp)))))
temp}

for (name in names(embedding_dat)) {
  print(name)
  embed_dims = get_embedding_values(name)
  embedding_dat[[name]] = cbind(embedding_dat[[name]],embed_dims)
}

require(Rtsne)

set.seed(123)
tsne = Rtsne(embedding_dat$DrivAge_embed[,c(-1,-2),with=F], perplexity = 5)
embed_to_plot = cbind(embedding_dat$DrivAge_embed[,c(2),with=F],
                      data.table(dim1 = tsne$Y[,1], dim2 = tsne$Y[,2]))

embed_to_plot %>% ggplot(aes(x=dim1,y=dim2))+ 
  geom_text(aes(label = trunc(Drivage), colour = Drivage))+ theme_pubr()
ggsave("c:/r/age_cluster.wmf", device = "wmf")

embed_to_plot %>% melt(id.vars="Drivage") %>% 
  ggplot(aes(x=Drivage,y=value))+ geom_point(aes(shape=variable, colour=variable))+
  geom_line(aes(group = variable, linetype = variable, colour=variable))+
  theme_pubr()

ggsave("c:/r/age_dims.wmf", device = "wmf")

### transfer learning

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

learned_feature = embedding_dat$DrivAge_embed
learned_feature$Drivage = NULL
learned_feature %>% setkey(DrivAgeNN)
freMTPL2freq %>% setkey(DrivAgeNN)
freMTPL2freq = freMTPL2freq %>% merge(learned_feature, all.x=T)

learn = freMTPL2freq [c(splits$Resample1) ,]
test = freMTPL2freq [ setdiff (c (1: nrow (freMTPL2freq )), splits$Resample1 ),]

d.glm1 = glm ( formula = ClaimNb ~ VehPowerGLM + VehAgeGLM + 
                 BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region +
                 AreaGLM +
                 DrivAge_embeddim1 + DrivAge_embeddim2+ DrivAge_embeddim3+
                 DrivAge_embeddim4+ DrivAge_embeddim5 +DrivAge_embeddim6+ DrivAge_embeddim7+
                 DrivAge_embeddim8+ DrivAge_embeddim9+ DrivAge_embeddim10+
                 DrivAge_embeddim11 +DrivAge_embeddim12, family = poisson (), data = learn , offset = log ( Exposure ))


learn$GLM_embed <- fitted (d.glm1)
test$GLM_embed <- predict (d.glm1 , newdata =test , type ="response" )

out_of_sample <- 2*( sum ( test$GLM_embed )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$GLM_embed )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test )
average_out_of_sample


results = rbind(results,
                data.table(Model = "GLM_embed", OutOfSample = average_out_of_sample))

### Compare GLM to other models

d.glm1 = glm ( formula = ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM +
                 BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region +
                 AreaGLM , family = poisson (), data = learn , offset = log ( Exposure ))

test$GLM <- predict (d.glm1 , newdata =test , type ="response" )


veh_gas       =  test$VehGasNN
veh_power     =  test$VehPowerNN  
veh_age       =  test$VehAgeNN
driv_age      =  test$DrivAgeNN
veh_brand     =  test$VehBrandNN     
region        =  test$RegionNN   
bon_mal       =  test$BonusMalusNN      
density       =  test$DensityNN  
area          =  test$AreaNN

x_test = list(
  
  Exposure = test$Exposure,
  VehGas=veh_gas,
  VehPower=veh_power,
  VehAge=veh_age,
  DrivAge=driv_age,
  VehBrand=veh_brand,
  Region=region,
  Area=area,
  BonMal = bon_mal, 
  Density = density)

test$NN_embed = model %>% predict(x_test)


out_of_sample <- 2*( sum ( test$NN_embed )- sum ( test$ClaimNb )
                     + sum ( log (( test$ClaimNb / test$NN_embed )^( test$ClaimNb ))))

average_out_of_sample <- out_of_sample / nrow ( test )
average_out_of_sample

freq_to_plot = 
  test[,.(data = mean(ClaimNb),GLM = mean(GLM), GLM_embed = mean(GLM_embed), NN_embed = mean(NN_embed)), keyby = (DrivAge)]

freq_to_plot=freq_to_plot %>% melt(id.vars=c("DrivAge","data"))
freq_to_plot[DrivAge<80] %>% ggplot(aes(x=DrivAge,y=value))+
  geom_line(aes(group = variable, linetype = variable,colour=variable))+
  geom_point(aes(x=DrivAge, y = data))+theme_pubr()

ggsave("c:/r/ageplot.wmf", device = "wmf")