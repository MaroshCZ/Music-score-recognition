import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

#The first Convolutional layer learns the low-level features of the input image, while the second Convolutional layer learns the high-level features.

#BatchNormalization layers after each Convolutional layer, we can improve the stability and speed of the network during training. 
# BatchNormalization helps to normalize the activations between layers, which can reduce the impact of the vanishing gradient problem 
# and improve the overall convergence speed and stability of the model.

#you can also add Dropout layers to the U-Net model to improve its regularization and prevent overfitting. Dropout is usually added after 
# the Convolutional layer, as it randomly drops out some of the activations to prevent the model from relying too heavily on any single feature or neuron.

# In general, a smaller stride size is preferred when the input images have important small-scale features that need to be preserved, 
# while a larger stride size can be used when the input images have fewer details or when there is a need to reduce the output resolution. 
# It's also worth noting that using larger strides can lead to faster computations and fewer parameters, which can be beneficial in some cases.


def EncoderMiniBlock(inputs,nfilters,activations,drop,max_pool):
    """
    """
    
    c= Conv2D(nfilters,(5,5), activation= activations,kernel_initializer='HeNormal', padding='same')(inputs)
    c= BatchNormalization()(c)
    c= Conv2D(nfilters,(5,5), activation= activations,kernel_initializer='HeNormal', padding='same')(c)
    c= BatchNormalization()(c)
    
    if drop > 0:
        c = Dropout(drop)(c)
        
    if max_pool:
        next_layer = MaxPooling2D((2,2))(c)
    else:
        next_layer = c
        
    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = c
    
    return next_layer, skip_connection
        
def DecoderMiniBlock(inputs,skip_layer,nfilters,activations,drop):
    """
    """
    c= Conv2DTranspose(nfilters,(5,5), strides=(2,2), activation= 'relu',kernel_initializer='HeNormal', padding='same')(inputs)
    c= BatchNormalization()(c)
    merge= concatenate([c,skip_layer])
    c= Conv2D(nfilters,(5,5), activation= 'relu',kernel_initializer='HeNormal', padding='same')(c)
    c= Conv2D(nfilters,(5,5), activation= 'relu',kernel_initializer='HeNormal', padding='same')(c)
    c= BatchNormalization()(c)
    
    if drop > 0:
        next_layer = Dropout(drop)(c)
    else:
        next_layer = c
    
    return next_layer
        
def u_net_model(input_size, nfilters,n_classes):
    """
    Definition of U-net network consisting of Encoder and Decoder
    """
    input= Input(shape=input_size)
    
    #EncoderMiniBlock(inputs,nfilters,activations,drop,max_pool)
    cblock1= EncoderMiniBlock(input, nfilters, activations='relu',drop= 0.1, max_pool=True)
    cblock2= EncoderMiniBlock(cblock1[0], 2*nfilters, activations='relu',drop= 0.1, max_pool=True)
    cblock3= EncoderMiniBlock(cblock2[0], 4*nfilters, activations='relu',drop= 0.1, max_pool=True)
    cblock4= EncoderMiniBlock(cblock3[0], 8*nfilters, activations='relu',drop= 0.1, max_pool=True)
    cblock5= EncoderMiniBlock(cblock4[0], 16*nfilters, activations='relu',drop= 0.1, max_pool=False)
    
    #DecoderMiniBlock(inputs,skip_layer,nfilters,activations,drop)
    ublock6= DecoderMiniBlock(cblock5[0],cblock4[1],8*nfilters,activations='relu',drop= 0.1)
    ublock7= DecoderMiniBlock(ublock6,cblock3[1],4*nfilters,activations='relu',drop= 0.1)
    ublock8= DecoderMiniBlock(ublock7,cblock2[1],2*nfilters,activations='relu',drop= 0.1)
    ublock9= DecoderMiniBlock(ublock8,cblock1[1],nfilters,activations='relu',drop= 0.1)
    
   
    conv10 = Conv2DTranspose(1,(5,5),activation= 'relu',kernel_initializer='HeNormal', padding='same')(ublock9)
    conv11 = Conv2D(1, (1,1), activation='tanh',kernel_initializer='HeNormal', padding='same')(conv10)
    
    #Define model
    model= Model(inputs= input,outputs= conv10)
    return model
        
        

#model= u_net_model((256,256,1),16,2) #u_net_model(input_size,nfilters,classes)
#model.summary()