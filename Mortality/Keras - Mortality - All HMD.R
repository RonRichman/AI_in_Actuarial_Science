#### Purpose: Fit deep regression network to HMD data
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

# dir = "C:/R/death_rates"
# 
# all_files = list.files(dir)
# 
# all_mort = list()
# i = 0
# for (file in all_files){
#   i=i+1
#   dat = HMDHFDplus::readHMD(paste0(dir,"/",file)) %>% data.table
#   dat[,Country :=str_replace(file, ".Mx_1x1.txt", "")]
#   all_mort[[i]] = dat
# }
# 
# all_mort = rbindlist(all_mort)
# all_mort %>% fwrite("c:/r/allmx.csv")
# 

#### setup test/train
all_mort= fread("c:/r/allmx.csv")
all_mort = all_mort[Year>1949 & Age<100]
all_mort = all_mort[,c(1,2,3,4,7),with=F]
all_mort = all_mort %>% melt(id.vars=c("Year" ,"Age", "Country")) %>% data.table

all_mort[,mx:=value]
all_mort[,logmx:=log(value)]
all_mort$value = NULL

all_mort[,Sex:=variable]
all_mort$variable = NULL

all_mort[,Country_fact:=as.integer(as.factor(Country))-1]
all_mort[,Sex_fact:=as.integer(as.factor(Sex))-1]

all_mort=all_mort[!is.na(logmx) & mx>0]

all_mort=all_mort[Country != "HRV"]

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


train = all_mort[Year < 2000]
test = all_mort[Year >= 2000]

scaled = scale_min_max(train$logmx, test$logmx)

train$mx_scale = scaled$train
test$mx_scale = scaled$test

#### Regression

train_reg = train[,c(1:2,7:9),with=F]
test_reg = test[,c(1:2,7:9),with=F]

year_scale = scale_min_max(train_reg$Year,test_reg$Year)

train_reg$Year = year_scale[[1]]
test_reg$Year = year_scale[[2]]


#train
x = list(Year      = train_reg$Year,
         Age = train_reg$Age, Country = train_reg$Country_fact, Sex=train_reg$Sex_fact)

y = (main_output= train_reg$mx_scale)

#test

x_test = list(Year      = test_reg$Year,
         Age = test_reg$Age, Country = test_reg$Country_fact, Sex=test_reg$Sex_fact)

y_test = (main_output= test_reg$mx_scale)

require(keras)
############### Build embedding layers
Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year')
Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
Country <- layer_input(shape = c(1), dtype = 'int32', name = 'Country')
Sex <- layer_input(shape = c(1), dtype = 'int32', name = 'Sex')

Age_embed = Age %>% 
  layer_embedding(input_dim = 100, output_dim = 20,input_length = 1, name = 'Age_embed') %>%
  keras::layer_flatten()


Sex_embed = Sex %>% 
  layer_embedding(input_dim = 2, output_dim = 1,input_length = 1, name = 'Sex_embed') %>%
  keras::layer_flatten()


Country_embed = Country %>% 
  layer_embedding(input_dim = 41, output_dim = 10,input_length = 1, name = 'Country_embed') %>%
  keras::layer_flatten()


main_output <- layer_concatenate(list(Year,Age_embed,Sex_embed,Country_embed
                                      )) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

model <- keras_model(
  inputs = c(Year,Age,Country,Sex), 
  outputs = c(main_output))

adam = optimizer_adam(lr=0.0005)
lr_callback = callback_reduce_lr_on_plateau(factor=.80, patience = 5, verbose=1, cooldown = 5, min_lr = 0.00005)
model_callback = callback_model_checkpoint(filepath = "c:/r/best_mx_reg_allhmd.mod", verbose = 1,save_best_only = TRUE)

model %>% compile(
  optimizer = adam,
  loss = "mse")

fit = model %>% fit(
  x = x,
  y = y, 
  epochs = 300,
  batch_size = 4096/6,verbose = 1, shuffle = T, validation_split = 0.1, callbacks = list(lr_callback,model_callback))

model = load_model_hdf5("c:/r/best_mx_reg_allhmd.mod")

test$mx_deep_reg_full = model %>% predict(x_test)
test[,mx_deep_reg_full:=exp(mx_deep_reg_full*(scaled$max-scaled$min)+scaled$min)]
test[,.(        Deep_reg = sum((mx-mx_deep_reg_full)^2)), keyby = .(Country,Sex)
     ] %>% fwrite("all_country.csv")

### visualize embedding

get_embedding_values = function(layer_name){
  embedding = model %>% get_layer(layer_name) %>% get_weights()
  temp = embedding[[1]] %>% data.table()
  temp%>% setnames(names(temp),paste0(layer_name,"dim",seq(1:length(names(temp)))))
  temp}

embedding_dat = list()
embed_dims = get_embedding_values("Age_embed")
embedding_dat[["Age_embed"]] = cbind(x$Age %>% unique,embed_dims)

embed_dims = get_embedding_values("Country_embed")
embedding_dat[["Country_embed"]] = cbind(train$Country %>% unique,embed_dims)

require(Rtsne)
set.seed(123)
tsne = Rtsne(embedding_dat$Age_embed[,c(-1),with=F], perplexity = 10)
pca = embedding_dat$Age_embed[,c(-1),with=F] %>% princomp()

embed_to_plot = cbind(embedding_dat$Age_embed[,c(1),with=F],
                      data.table(dim1 = pca$scores[,1], dim2 = pca$scores[,2]))

embed_to_plot[,Age:=V1]
embed_to_plot$V1 = NULL

embed_to_plot %>% melt(id.vars="Age") %>% 
  ggplot(aes(x=Age,y=value))+ geom_point(aes(shape=variable, colour=variable))+
  geom_line(aes(group = variable, linetype = variable, colour=variable))+
  theme_pubr()

ggsave("c:/r/age_dims_mort.wmf", device = "wmf")

tsne = Rtsne(embedding_dat$Country_embed[,c(-1),with=F], perplexity = 10)
pca = embedding_dat$Country_embed[,c(-1),with=F] %>% princomp()

embed_to_plot = cbind(embedding_dat$Country_embed[,c(1),with=F],
                      data.table(dim1 = tsne$Y[,1], dim2 = tsne$Y[,2]))

embed_to_plot[1:40] %>% ggplot(aes(x=dim1,y=dim2))+ 
  geom_text(aes(label = V1))+
  theme_pubr()


test[Country == "GBRTENW"][Year==2016][Sex == "Male"] %>% fwrite("c:/r/gbrtenw.csv")