import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers.core.dense import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.applications import vgg19
from keras import Model
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

# read inputs
target_images = []
input_images = []
for file in os.listdir("target/"):
    if(file == ".ipynb_checkpoints"):
        continue
    print(file)
    target_img = cv2.imread("target/"+file)
    # target_img = np.expand_dims(target_img,0)
    target_img = tf.convert_to_tensor(target_img)
    target_img = tf.cast(target_img,tf.float32)
    target_images.append(target_img)
    
    input_img = cv2.imread("input/"+file)
    # input_img = np.expand_dims(input_img,0)
    input_img = tf.convert_to_tensor(input_img)
    input_img = tf.cast(input_img,tf.float32)
    input_images.append(input_img)    

inputs = tf.convert_to_tensor(input_images)
targets = tf.convert_to_tensor(target_images)


### LOSS RELATED
def vgg_layers(layer_names):
        """ Creates a VGG model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on ImageNet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs,style_targets,content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([((tf.norm((style_outputs[name]-style_targets[name]),ord=1))/(style_targets[name].shape[2]**2))
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.norm((content_outputs[name]-content_targets[name]),ord=1) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

content_layers = ['block1_conv2',
                 'block2_conv2',
                 'block3_conv2',
                 'block4_conv2',
                 'block5_conv2',] 

style_layers = ['block1_conv2',
                 'block2_conv2',
                 'block3_conv2',
                 'block4_conv2',
                 'block5_conv2',]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

extractor = StyleContentModel(style_layers, content_layers)
content_weight=0.05
style_weight=120

class Encoder(keras.layers.Layer):
    def __init__(self,name="encoder", **kwargs):
        super(Encoder,self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters = 64,kernel_size=3,activation="relu",strides=1,padding="same")
        self.bn1 = BatchNormalization()

        self.conv2_1 = Conv2D(filters=128,kernel_size=3,activation = "relu",strides=2,padding="same")
        self.bn2 = BatchNormalization()
        self.conv2_2 = Conv2D(filters=128,kernel_size=3,activation = "relu",strides=1,padding="same")
        self.bn3 = BatchNormalization()

        self.conv3_1 = Conv2D(filters=256,kernel_size=3,activation = "relu",strides=2,padding="same")
        self.bn4 = BatchNormalization()
        self.conv3_2 = Conv2D(filters=256,kernel_size=3,activation="relu",strides=1,padding="same")
        self.bn5 = BatchNormalization()

        self.conv4_1 = Conv2D(filters=512,kernel_size=3,activation="relu",strides=2,padding="same")
        self.bn6 = BatchNormalization()
        self.conv4_2 = Conv2D(filters=512,kernel_size=3,activation="relu",strides=1,padding="same")
        self.bn7 = BatchNormalization()

        self.conv5_1 = Conv2D(filters=1024,kernel_size=3,activation="relu",strides=2,padding="same")
        self.bn8 = BatchNormalization()
        self.conv5_2 = Conv2D(filters=1024,kernel_size=3,activation="relu",strides=1,padding="same")
        self.bn9 = BatchNormalization()
        
        
    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.conv2_2(x)
        x = self.bn3(x)
        x = self.conv3_1(x)
        x = self.bn4(x)
        x = self.conv3_2(x)
        convolution_3_2 = self.bn5(x)
        y = self.conv4_1(convolution_3_2)
        y = self.bn6(y)
        y = self.conv4_2(y)
        convolution_4_2 = self.bn7(y)
        z = self.conv5_1(convolution_4_2)
        z = self.bn8(z)
        z = self.conv5_2(z)
        convolution_5_2 = self.bn9(z)
        
        return convolution_3_2,convolution_4_2,convolution_5_2 
        # Done Encoder
        
# @tf.keras.utils.register_keras_serializable()
class Decoder(keras.layers.Layer):
    def __init__(self,name="dencoder", **kwargs):
        super(Decoder,self).__init__(name=name, **kwargs)
        # self.transpose_conv_block_5 = transpose_conv_block_5
        # self.transpose_conv_block_4 = transpose_conv_block_4
        # self.transpose_conv_block_3 = transpose_conv_block_3
        
        self.bi_upsample_1 = UpSampling2D(size=(2,2),interpolation = 'bilinear')
        self.conv6 = Conv2D(filters = 512,kernel_size = 3,activation="relu",strides=1,padding="same")
        self.bi_upsample_2 = UpSampling2D(size=(2,2),interpolation = 'bilinear')
        self.conv7 = Conv2D(filters = 256,kernel_size = 3,activation="relu",strides=1,padding="same")
        self.bi_upsample_3 = UpSampling2D(size=(2,2),interpolation = 'bilinear')
        self.conv8 = Conv2D(filters = 128,kernel_size = 3,activation="relu",strides=1,padding="same")
        self.bi_upsample_4 = UpSampling2D(size=(2,2),interpolation = 'bilinear')
        self.conv9 = Conv2D(filters = 64,kernel_size = 3,activation="relu",strides=1,padding="same")
        self.conv10 = Conv2D(filters = 3,kernel_size = 3,activation="relu",strides=1,padding="same")
                
    
    def call(self,transpose_conv_block_5,transpose_conv_block_4,transpose_conv_block_3):
        x = self.bi_upsample_1(transpose_conv_block_5)
        x = self.conv6(x)
        sum1 = tf.math.add(x,transpose_conv_block_4)
        y = self.bi_upsample_2(sum1)
        y = self.conv7(y)
        sum2 = tf.math.add(y,transpose_conv_block_3)
        z = self.bi_upsample_3(sum2)
        z = self.conv8(z)
        z = self.bi_upsample_4(z)
        z = self.conv9(z)
        z = self.conv10(z)
        
        return z
    
# @tf.keras.utils.register_keras_serializable()
class a_squared_layer(keras.layers.Layer):
    # def __init__(self,encoded_feature,name="encoder", **kwargs)::
    #     super(a_squared_layer,self).__init__(name=name, **kwargs):
    #     self.encoded_feature = encoded_feature
        
    def call(self,encoded_feature):

        feature_squared = tf.math.square(encoded_feature)
        # feature_squared = np.square(encoded_feature)

        target_shape = [feature_squared.shape[0],feature_squared.shape[1]*2,feature_squared.shape[2]*2,feature_squared.shape[3]]
        # target_shape = [1,feature_squared.shape[1]*2,feature_squared.shape[2]*2,feature_squared.shape[3]]

        padded_feature = tf.keras.layers.ZeroPadding2D(padding=int(feature_squared.shape[1]/2))(feature_squared)
        
        
#         padded_feature = tf.zeros(target_shape)
        
#         print(feature_squared.shape)
#         print("target_shape",target_shape)
#         print("tr",tr.shape)
#         print("chec",tf.equal(tr[:,int(feature_squared.shape[1]/2):(int(feature_squared.shape[1]/2) + int(feature_squared.shape[1])),int(feature_squared.shape[2]/2):(int(feature_squared.shape[2]/2) + int(feature_squared.shape[2])),:],feature_squared))
#         print("chec2",tf.equal(feature_squared,feature_squared))
        
#         h_stack_value = tf.zeros([feature_squared.shape[0],int(feature_squared.shape[1]),int(feature_squared.shape[2]/2),feature_squared.shape[3]])
#         print(tf.concat([feature_squared,h_stack_value],axis=1).shape)
                                    

#         padded_feature[:,int(feature_squared.shape[1]/2):(int(feature_squared.shape[1]/2) + int(feature_squared.shape[1])),int(feature_squared.shape[2]/2):(int(feature_squared.shape[2]/2) + int(feature_squared.shape[2])),:] = feature_squared

        padded_feature = tf.convert_to_tensor(padded_feature,dtype=tf.float32)

        filter_feature = tf.zeros(feature_squared.shape[1:]) + 1

        filter_feature = tf.expand_dims(filter_feature,axis=3)

        a_squared = tf.nn.conv2d(input=padded_feature,filters=filter_feature,strides=1,padding="VALID")        

        return a_squared        


# @tf.keras.utils.register_keras_serializable()
class b_squared_layer(keras.layers.Layer):
    # def __init__(self,encoded_feature):
    #     super(b_squared_layer,self).__init__()
    #     self.encoded_feature = encoded_feature
        
    def call(self,encoded_feature):
        
        filter_feature_encoded = encoded_feature[0,:]
        
        feature = encoded_feature
        
        target_shape = [feature.shape[0],feature.shape[1]*2,feature.shape[2]*2,feature.shape[3]]                

        filter_feature_encoded = tf.expand_dims(filter_feature_encoded,3)

        filter_feature_encoded_squared = tf.square(filter_feature_encoded)

        padded_feature_b = np.zeros(target_shape)

        padded_feature_b[:,int(feature.shape[1]/2):(int(feature.shape[1]/2) + int(feature.shape[1])),int(feature.shape[2]/2):(int(feature.shape[2]/2) + int(feature.shape[2])),:] = 1

        padded_feature_b = tf.convert_to_tensor(padded_feature_b,dtype=tf.float32)

        b_squared = tf.nn.conv2d(input = padded_feature_b,filters=filter_feature_encoded_squared,strides=1,padding="VALID")
             
        return b_squared
    
# @tf.keras.utils.register_keras_serializable()
class ab_layer(keras.layers.Layer):
    # def __init__(self,encoded_feature):
    #     super(ab_layer,self).__init__()
    #     self.encoded_feature = encoded_feature
        
    def call(self,encoded_feature):
        feature = encoded_feature
        
        filter_feature_encoded = encoded_feature[0,:]  
        
        filter_feature_encoded = tf.expand_dims(filter_feature_encoded,3)        
        
        target_shape = [feature.shape[0],feature.shape[1]*2,feature.shape[2]*2,feature.shape[3]]                
        
        padded_feature_ab = tf.keras.layers.ZeroPadding2D(padding=int(feature.shape[1]/2))(feature)
        
#         padded_feature_ab = np.zeros(target_shape)

#         padded_feature_ab[:,int(feature.shape[1]/2):(int(feature.shape[1]/2) + int(feature.shape[1])),int(feature.shape[2]/2):(int(feature.shape[2]/2) + int(feature.shape[2])),:] = feature

        padded_feature_ab = tf.convert_to_tensor(padded_feature_ab,dtype=tf.float32)

        ab = tf.nn.conv2d(input = padded_feature_ab,filters=filter_feature_encoded,strides=1,padding="VALID")                     
        
        return ab
    

# @tf.keras.utils.register_keras_serializable()
class self_similarity_layer(keras.layers.Layer):
    def __init__(self,name="self_similarity", **kwargs):
        super(self_similarity_layer,self).__init__(name=name, **kwargs)
        self.a_squared_operation = a_squared_layer()
        self.b_squared_operation = b_squared_layer()
        self.ab_operation = ab_layer()
        self.branch1 = tf.function(Conv2D(filters=8,kernel_size = 3,strides=1,activation="relu",padding="same"))
        self.branch2 = tf.function(Conv2D(filters=1,kernel_size = 3,strides=1,padding="same"))
        
    def call(self,encoded_feature):
        #Calculation of a_squared
        a_squared = self.a_squared_operation(encoded_feature)
        #Calculation of b_squared    
        b_squared = self.b_squared_operation(encoded_feature)

        #calculate ab
        ab = self.ab_operation(encoded_feature)
        
        #calculation of self_similarity
        self_similarity_map = -(((a_squared + b_squared) - (2*ab))/a_squared)
        self_similarity_branch_conv1 = self.branch1(self_similarity_map)
        self_similarity_branch_conv2 = self.branch2(self_similarity_branch_conv1)
        # self_similarity_branch_conv1 = Conv2D(filters=8,kernel_size = 3,strides=1,activation="relu",padding="same")(self_similarity_map)
        # self_similarity_branch_conv2 = Conv2D(filters=1,kernel_size = 3,strides=1,padding="same")(self_similarity_branch_conv1)
        
        return self_similarity_branch_conv2        
    

# @tf.keras.utils.register_keras_serializable()
class transpose_convolution_layer(keras.layers.Layer):
    def __init__(self,num_channels = 256,name="transpose_convolution_layer", **kwargs):
        super(transpose_convolution_layer,self).__init__(name=name, **kwargs)
        self.compute_self_similarity_map = self_similarity_layer()
        self.num_filters = num_channels
        self.final_layer = tf.function(Conv2D(filters =self.num_filters,kernel_size =1,activation="relu",strides=1))
        self.branch_conv1 = tf.function(Conv2D(filters=self.num_filters,kernel_size=3,activation="relu",strides=1,padding="same"))
        self.branch_conv2 = tf.function(Conv2D(filters=self.num_filters,kernel_size=3,strides=1,padding="same"))
    
    def call(self,encoded_feature):
        self_similarity_map = self.compute_self_similarity_map(encoded_feature)
        layer_h = encoded_feature.shape[1]
        layer_c = encoded_feature.shape[3]
        batch_size = encoded_feature.shape[0]        
        
        output_shape_h = layer_h * 2
        output_shape_c = layer_c 
        
        layer_branch_conv1 = self.branch_conv1(encoded_feature)
        layer_branch_conv2 = self.branch_conv2(layer_branch_conv1)
        
        bias = GlobalAveragePooling2D()(layer_branch_conv2)
        transposed_conv_bias = tf.function(Dense(batch_size))(bias)   
        
        
        transposed_conv_weight = layer_branch_conv2[0,:]
        transposed_conv_weight = tf.expand_dims(transposed_conv_weight,axis=3)        
        
        transpose_output_shape = np.array([  batch_size, output_shape_h, output_shape_h, output_shape_c])
        transpose_output_shape = tf.convert_to_tensor(transpose_output_shape,dtype=tf.int32) 
        transpose_operation_output = tf.nn.conv2d_transpose(input = self_similarity_map,filters=transposed_conv_weight,output_shape=transpose_output_shape, strides=1,padding="VALID")
        transpose_operation_output = transposed_conv_bias + transpose_operation_output
        transpose_conv_output_1 = self.final_layer(transpose_operation_output)
         
        return transpose_conv_output_1
    
class unet_like(keras.Model):
    def __init__(self,name="autoencoder",**kwargs):
        super(unet_like, self).__init__(name=name, **kwargs)        
        self.encode = Encoder()
        self.decode = Decoder()
        
    def call(self,inputs):
#        print("inputs",inputs.shape)
        conv3_2,conv4_2,conv5_2 = self.encode(inputs)
        
        ##### LAYER 3_2 #####
        conv3_2_number_of_channels = conv3_2.shape[3]
        compute_transpose_conv_3_2 = transpose_convolution_layer(conv3_2_number_of_channels)
        transpose_conv_block_3 = compute_transpose_conv_3_2(conv3_2)
        
        ### LAYER 4_2 ###
        conv4_2_number_of_channels = conv4_2.shape[3]
        compute_transpose_conv_4_2 = transpose_convolution_layer(conv4_2_number_of_channels)
        transpose_conv_block_4 = compute_transpose_conv_4_2(conv4_2)
        
        ### LAYER 5_2 ###
        conv5_2_number_of_channels = conv5_2.shape[3]
        compute_transpose_conv_5_2 = transpose_convolution_layer(conv5_2_number_of_channels)
        transpose_conv_block_5 = compute_transpose_conv_5_2(conv5_2)
        
        output_image = self.decode(transpose_conv_block_5,transpose_conv_block_4,transpose_conv_block_3)
        
        return output_image
    
    
def my_loss_function(y_true,y_pred):    
    i_target = y_true
    i_output = y_pred
    outputs = extractor(i_output)

    style_targets = extractor(i_target)['style']
    content_targets = extractor(i_target)['content']
    loss = style_content_loss(outputs,style_targets,content_targets)
    
    return loss  

model = unet_like()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0000000001),loss = my_loss_function,metrics=['accuracy'])

model.fit(inputs,targets,epochs=50,batch_size = 1)

model.save("saved_model/model_tr1")
