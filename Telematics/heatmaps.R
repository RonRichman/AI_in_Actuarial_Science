#### Purpose: Autoencode heatmap using Keras
#### Author: Ronald Richman
#### License: MIT
#### Data: The data was generated generated using the simulation machine generously 
####       provided by Wuthrich(2018)


require(data.table)
require(dplyr)
require(ggplot2)
require(reshape2)

### Read in a matrix of heatmaps 
heat_maps = fread("c:/R/heatmaps.csv")
heat_maps_mat = as.matrix(heat_maps)

require(keras)

### Greedy unsupervised pre-training

x = list(NumInputs=heat_maps_mat,Density=heat_maps_mat)

NumInputs <- layer_input(shape = c(400), dtype = 'float32', name = 'NumInputs')

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1")


Density = encode %>%  
  layer_dense(units = 400, activation = 'sigmoid', name = "Density") 

model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

adam = optimizer_adam(lr = 0.001)

model %>% compile(
  optimizer = adam,
  loss = "binary_crossentropy")

fit = model %>% fit(
  x = x,
  y = x, 
  epochs = 50,
  batch_size = 32,verbose = 1, shuffle = T)

#model %>% save_model_hdf5("c:/r/greedy_auto_encode1_HM.mod")
model = load_model_hdf5("c:/r/greedy_auto_encode1_HM.mod")

### Step 2

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1") %>% 
  layer_dense(units = 2, activation = 'linear', name = "encode")

Density = encode %>%  
  layer_dense(units =4, activation = 'tanh', name = "DecodeRates1") %>% 
  layer_dense(units = 400, activation = 'sigmoid', name = "Density") 

model_stage2 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

adam = optimizer_adam(lr = 0.001)

set_weights(get_layer(model_stage2, "EncodeRates1"), get_weights(get_layer(model, "EncodeRates1")))
set_weights(get_layer(model_stage2, "Density"), get_weights(get_layer(model, "Density")))

layer = get_layer(model_stage2, "EncodeRates1")
layer$trainable = FALSE

layer = get_layer(model_stage2, "Density")
layer$trainable = FALSE

model_stage2 %>% compile(
  optimizer = adam,
  loss = "binary_crossentropy")

fit = model_stage2 %>% fit(x = x,
                           y = x, 
                           epochs = 20,
                           batch_size = 8,verbose = 1, shuffle = T)

#model_stage2 %>% save_model_hdf5("c:/r/greedy_auto_encode2_HM.mod")
model_stage2 = load_model_hdf5("c:/r/greedy_auto_encode2_HM.mod")


### Step 3

encode = NumInputs %>%  
  layer_dense(units = 4, activation = 'tanh', name = "EncodeRates1") %>% 
  layer_dense(units = 2, activation = 'linear', name = "encode")

Density = encode %>%  
  layer_dense(units =4, activation = 'tanh', name = "DecodeRates1") %>% 
  layer_dense(units = 400, activation = 'sigmoid', name = "Density") 

model_stage3 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

set_weights(get_layer(model_stage3, "EncodeRates1"), get_weights(get_layer(model_stage2, "EncodeRates1")))
set_weights(get_layer(model_stage3, "encode"), get_weights(get_layer(model_stage2, "encode")))
set_weights(get_layer(model_stage3, "DecodeRates1"), get_weights(get_layer(model_stage2, "DecodeRates1")))
set_weights(get_layer(model_stage3, "Density"), get_weights(get_layer(model_stage2, "Density")))

model_stage3 %>% compile(
  optimizer = "adadelta",
  loss = "binary_crossentropy")

fit = model_stage3 %>% fit(x = x,
                           y = x, 
                           epochs = 10,
                           batch_size = 8,verbose = 1, shuffle = T)

model = model_stage3
#model %>% save_model_hdf5("c:/r/vaheatmap_autoencode.mod")

model = load_model_hdf5("c:/r/vaheatmap_autoencode.mod")

codes_model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(encode)) %>% compile(
    optimizer = adam,
    loss = "binary_crossentropy")

codes =  codes_model %>% predict(x$NumInputs) %>% data.table

require(Hmisc)

codes[,V1_Grp := cut2(V1*10000,g=3)]
codes[,V2_Grp := cut2(V2*10000,g=3)]
codes[,col := as.integer(as.factor(paste0(V1_Grp,V2_Grp)))]

require(ggplot2)
require(ggpubr)

codes %>% setnames(names(codes), c("dim1", "dim2", "dim1_grp", "dim2_grp", "col"))
codes[order(dim1,dim2)] %>% 
  ggplot(aes(x=dim1,y=dim2)) + geom_point(aes(colour=as.factor(col)))+
  theme_pubr()+scale_color_discrete(name = "Group")
ggsave("c:/r/auto_encode_va.wmf", device = "wmf")

codes[,id:=.I]
codes %>% setkey(id)

heat_maps=heat_maps %>% data.table()
heat_maps[,id:=.I]
heat_maps %>% setkey(id)

heat_maps_frame=heat_maps %>% merge(codes)

heat_maps_melt = heat_maps_frame %>% melt(id.vars=c("id","dim1_grp", "dim2_grp", "col", "dim1","dim2"))
summary = heat_maps_melt[,mean(value), keyby = .(col, variable)]

N_types = summary[,unique(col)] %>% max

summary$v = rep(sapply(1:20,function(x) rep(x,20)) %>% as.vector(),N_types)
summary$a = rep(rep(seq(1,20),20),N_types)


sum_plot = ggplot(summary) + 
  aes(x = v, y = a, z = V1, fill = V1) + 
  geom_tile() + 
  coord_equal() +
  geom_contour(color = "white", alpha = 1) + 
  scale_fill_distiller(palette="Spectral", na.value="white", name="Density") + 
  theme_bw()+ facet_wrap(~col)+theme_pubr()

ggpar(sum_plot,legend="right")

ggsave("c:/r/auto_encode_va_mean.wmf", device = "wmf", height = 10, width =15 )

### individual maps

maps = heat_maps_melt[,c("id", "col"),with=F] %>% group_by(col) %>% sample_n(5)

summary = heat_maps_melt[id %in% maps$id] %>% setkey(col,id)
code_id = summary[,.GRP, by=.(col,id)][,id_graph := 1:5, by=col]%>% setkey(col,id)
summary = summary %>% merge(code_id)

N_types = summary[,unique(id)] %>% length

summary$v = rep(sapply(1:20,function(x) rep(x,20)) %>% as.vector(),N_types)
summary$a = rep(rep(seq(1,20),20),N_types)

ggplot(summary) + 
  aes(x = v, y = a, z = value, fill = value) + 
  geom_tile() + 
  coord_equal() +
  geom_contour(color = "white", alpha = 0.5) + 
  scale_fill_distiller(palette="Spectral", na.value="white") + 
  theme_bw()+ facet_grid(col~id_graph)
