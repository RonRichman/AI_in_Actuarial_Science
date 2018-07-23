#### Purpose: Autoencode heatmap convolutionally using Keras
#### Author: Ronald Richman
#### License: MIT
#### Data: The data was generated generated using the simulation machine generously 
####       provided by Wuthrich(2018)

require(data.table)
require(dplyr)
require(ggplot2)
require(reshape2)

### Read in a matrix of heatmpas generated using the simulation machine provided by Wuthrich(2018)

heat_maps = fread("c:/r/heatmaps.csv")

heat_maps_mat = as.matrix(heat_maps)

### Greedy unsupervised pretraining

require(keras)

### read in heatmaps as an array of images

x = array_reshape(heat_maps_mat, c(nrow(heat_maps_mat), 20, 20, 1))
input_shape <- c(20, 20, 1)

NumInputs <- layer_input(shape = c(20,20,1), dtype = 'float32', name = 'NumInputs')

encode = NumInputs %>%  
  layer_conv_2d(filters = 8, kernel_size = c(10,10), activation = 'tanh', name="Conv1")
#%>% 
 # layer_conv_2d(filters = 4, kernel_size = c(10,10), activation = 'relu', name= "Conv2") %>% 
  #layer_conv_2d(filters = 2, kernel_size = c(2,2), activation = 'relu', name='encode')
  
Density = encode %>%  
  #layer_conv_2d_transpose(filters = 2, kernel_size = c(2,2), activation = 'relu', name = "Deconv3") %>% 
  #layer_conv_2d_transpose(filters = 4, kernel_size = c(10,10), activation = 'relu', name = "Deconv2") %>% 
  layer_conv_2d_transpose(filters = 8, kernel_size = c(10,10), activation = 'tanh', name = "Deconv1") %>% 
  layer_conv_2d_transpose(filters = 1, kernel_size = c(1,1), activation = 'sigmoid', name = "decode")

model_stage1 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

model_stage1 %>% compile(
  optimizer = "adadelta",
  loss = "mape")

fit = model_stage1 %>% fit(
  x = x,
  y = x, 
  epochs = 15,
  batch_size = 32,verbose = 1, shuffle = T)

#model_stage1 %>% save_model_hdf5("c:/r/greedy_auto_encodeconv1.mod")
model_stage1 = load_model_hdf5("c:/r/greedy_auto_encodeconv1.mod")

### Step 2
encode = NumInputs %>%  
  layer_conv_2d(filters = 8, kernel_size = c(10,10), activation = 'tanh', name="Conv1") %>% 
 layer_conv_2d(filters = 4, kernel_size = c(10,10), activation = 'tanh', name= "Conv2") 
#%>% 
#layer_conv_2d(filters = 2, kernel_size = c(2,2), activation = 'relu', name='encode')

Density = encode %>%  
  #layer_conv_2d_transpose(filters = 2, kernel_size = c(2,2), activation = 'relu', name = "Deconv3") %>% 
  layer_conv_2d_transpose(filters = 4, kernel_size = c(10,10), activation = 'tanh', name = "Deconv2") %>% 
  layer_conv_2d_transpose(filters = 8, kernel_size = c(10,10), activation = 'tanh', name = "Deconv1") %>% 
  layer_conv_2d_transpose(filters = 1, kernel_size = c(1,1), activation = 'sigmoid', name = "decode")

model_stage2 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

set_weights(get_layer(model_stage2, "Conv1"), get_weights(get_layer(model_stage1, "Conv1")))
set_weights(get_layer(model_stage2, "decode"), get_weights(get_layer(model_stage1, "decode")))

layer = get_layer(model_stage2, "Conv1")
layer$trainable = FALSE

layer = get_layer(model_stage2, "decode")
layer$trainable = FALSE

model_stage2 %>% compile(
  optimizer = "adadelta",
  loss = "mape")

fit = model_stage2 %>% fit(
  x = x,
  y = x, 
  epochs = 15,
  batch_size = 32,verbose = 1, shuffle = T)

model_stage2 %>% save_model_hdf5("c:/r/greedy_auto_encodeconv2.mod")
model_stage2 = load_model_hdf5("c:/r/greedy_auto_encodeconv2.mod")

### Step 3
encode = NumInputs %>%  
  layer_conv_2d(filters = 8, kernel_size = c(10,10), activation = 'tanh', name="Conv1") %>% 
  layer_conv_2d(filters = 4, kernel_size = c(10,10), activation = 'tanh', name= "Conv2") %>% 
  layer_conv_2d(filters = 2, kernel_size = c(2,2), activation = 'tanh', name='encode')

Density = encode %>%  
  layer_conv_2d_transpose(filters = 2, kernel_size = c(2,2), activation = 'tanh', name = "Deconv3") %>% 
  layer_conv_2d_transpose(filters = 4, kernel_size = c(10,10), activation = 'tanh', name = "Deconv2") %>% 
  layer_conv_2d_transpose(filters = 8, kernel_size = c(10,10), activation = 'tanh', name = "Deconv1") %>% 
  layer_conv_2d_transpose(filters = 1, kernel_size = c(1,1), activation = 'sigmoid', name = "decode")

model_stage3 <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

set_weights(get_layer(model_stage3, "Conv1"), get_weights(get_layer(model_stage2, "Conv1")))
set_weights(get_layer(model_stage3, "Deconv1"), get_weights(get_layer(model_stage2, "Deconv1")))
set_weights(get_layer(model_stage3, "Conv2"), get_weights(get_layer(model_stage2, "Conv2")))
set_weights(get_layer(model_stage3, "decode"), get_weights(get_layer(model_stage2, "decode")))


layer = get_layer(model_stage3, "Conv1")
layer$trainable = FALSE

layer = get_layer(model_stage3, "Conv2")
layer$trainable = FALSE

layer = get_layer(model_stage3, "Deconv1")
layer$trainable = FALSE

layer = get_layer(model_stage3, "decode")
layer$trainable = FALSE

model_stage3 %>% compile(
  optimizer = "adadelta",
  loss = "mape")

fit = model_stage3 %>% fit(
  x = x,
  y = x, 
  epochs = 15,
  batch_size = 32,verbose = 1, shuffle = T)

model_stage3 %>% save_model_hdf5("c:/r/greedy_auto_encodeconv3.mod")
model_stage3 = load_model_hdf5("c:/r/greedy_auto_encodeconv3.mod")

### stage 4

encode = NumInputs %>%  
  layer_conv_2d(filters = 8, kernel_size = c(10,10), activation = 'tanh', name="Conv1") %>% 
  layer_conv_2d(filters = 4, kernel_size = c(10,10), activation = 'tanh', name= "Conv2") %>% 
  layer_conv_2d(filters = 2, kernel_size = c(2,2), activation = 'tanh', name='encode')

Density = encode %>%  
  layer_conv_2d_transpose(filters = 2, kernel_size = c(2,2), activation = 'tanh', name = "Deconv3") %>% 
  layer_conv_2d_transpose(filters = 4, kernel_size = c(10,10), activation = 'tanh', name = "Deconv2") %>% 
  layer_conv_2d_transpose(filters = 8, kernel_size = c(10,10), activation = 'tanh', name = "Deconv1") %>% 
  layer_conv_2d_transpose(filters = 1, kernel_size = c(1,1), activation = 'sigmoid', name = "decode")

model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(Density))

set_weights(get_layer(model, "Conv1"), get_weights(get_layer(model_stage3, "Conv1")))
set_weights(get_layer(model, "Deconv1"), get_weights(get_layer(model_stage3, "Deconv1")))
set_weights(get_layer(model, "Conv2"), get_weights(get_layer(model_stage3, "Conv2")))
set_weights(get_layer(model, "Deconv2"), get_weights(get_layer(model_stage3, "Deconv2")))
set_weights(get_layer(model, "encode"), get_weights(get_layer(model_stage3, "encode")))
set_weights(get_layer(model, "Deconv3"), get_weights(get_layer(model_stage3, "Deconv3")))
set_weights(get_layer(model, "decode"), get_weights(get_layer(model_stage3, "decode")))

model %>% compile(
  optimizer = "adadelta",
  loss = "binary_crossentropy")

fit = model %>% fit(
  x = x,
  y = x, 
  epochs = 10,
  batch_size = 32,verbose = 1, shuffle = T)

#model %>% save_model_hdf5("c:/r/greedy_auto_encodeconv.mod")
model= load_model_hdf5("c:/r/greedy_auto_encodeconv.mod")

codes_model <- keras_model(
  inputs = c(NumInputs), 
  outputs = c(encode)) %>% compile(
    optimizer = "adadelta",
    loss = "binary_crossentropy")

#predmaps = model %>% predict(x)

codes =  codes_model %>% predict(x)
codes = data.table(codes[,,,1], codes[,,,2])

require(Hmisc)

codes[,V1_Grp := cut2(V1,m=3000)]
codes[,V2_Grp := cut2(V2,m=3000)]
codes[,col := as.integer(as.factor(paste0(V1_Grp,V2_Grp)))]


require(ggplot2)


codes %>% setnames(names(codes), c("dim1", "dim2", "dim1_grp", "dim2_grp", "col"))
codes[order(dim1,dim2)] %>% 
  ggplot(aes(x=dim1,y=dim2)) + geom_point(aes(colour=as.factor(col)))+
  theme_pubr()+scale_color_discrete(name = "Group")
ggsave("c:/r/auto_encode_va_conv.wmf", device = "wmf")

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

ggsave("c:/r/auto_encode_va_mean_conv.wmf", device = "wmf", height = 10, width =15 )

###

maps = heat_maps_melt[,c("id", "col"),with=F] %>% group_by(col) %>% sample_n(5)

summary = heat_maps_melt[id %in% maps$id] %>% setkey(col,id)
code_id = summary[,.GRP, by=.(col,id)][,id_graph := 1:10, by=col]%>% setkey(col,id)
summary = summary %>% merge(code_id)

N_types = summary[,unique(id)] %>% length

summary$v = rep(sapply(1:20,function(x) rep(x,20)) %>% as.vector(),N_types)
summary$a = rep(rep(seq(1,20),20),N_types)

ggplot(summary) + 
  aes(x = v, y = a, z = value, fill = value) + 
  geom_tile() + 
  coord_equal() +
  geom_contour(color = "white", alpha = 1) + 
  scale_fill_distiller(palette="Spectral", na.value="white", name="Density") + 
  theme_bw()+ facet_grid(id_graph~col)+theme_pubr()

ggsave("c:/r/auto_encode_va_mean_conv_sample.wmf", device = "wmf")



