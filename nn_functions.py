import numpy as np
import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D


def one_hot_encoding(Y_raw, num_classes):

    return np.eye(num_classes)[Y_raw.astype(int)]
    
def identitiy_block(X, f, filters, stage, block):

    #save input X as it needs to be added later on 
    X_pre = X

    F1, F2, F3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #First CONV layer
    X = Conv2D(F1, (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #Second CONV layer
    X = Conv2D(F2, (f,f), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #Third CONV layer, no activation
    X = Conv2D(F3, (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #Add X_pre
    X = Add()([X, X_pre])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):

    X_pre = X

    F1, F2, F3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #First CONV layer
    X = Conv2D(F1, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #Second CONV layer
    X = Conv2D(F2, kernel_size = (f,f), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #Third CONV layer, activation
    X = Conv2D(F3, kernel_size = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    #Add X_pre after convolving it
    X_pre = Conv2D(F3, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '1')(X_pre)
    X_pre = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_pre)
    
    X = Add()([X, X_pre])
    X = Activation('relu')(X)

    return X

def ResNet(input_shape = (32, 32, 3), classes = 10):

    #Define Input to Model to be of shape input_shape
    X_input = Input(input_shape)

    #Pad Input with 3 in dim 0, 1
    X = ZeroPadding2D((3,3))(X_input)

    #Stage 1: CONV2D --> BatchNorm --> ReLU --> MaxPool
    X = Conv2D(8, kernel_size = (7, 7), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), (2,2))(X)

    #Stage 2: CONV_Block --> ID_block (x2)
    X = convolutional_block(X, 3, [16, 16, 32], stage = 2, block = 'a', s = 1)
    X = identitiy_block(X, 3, [16, 16, 32], stage = 2, block = 'b')
    X = identitiy_block(X, 3, [16, 16, 32], stage = 2, block = 'c')

    #Stage 3: CONV_block --> ID_block (x3)
    X = convolutional_block(X, 3, [32, 32, 64], stage = 3, block = 'a', s = 2)
    X = identitiy_block(X, 3, [32, 32, 64], stage = 3, block = 'b')
    X = identitiy_block(X, 3, [32, 32, 64], stage = 3, block = 'c')
    X = identitiy_block(X, 3, [32, 32, 64], stage = 3, block = 'd')

    #Stage 4: CONV_block --> ID_block(x5)
    X = convolutional_block(X, 3, [64, 64, 128], stage = 4, block = 'a', s = 2)
    X = identitiy_block(X, 3, [64, 64, 128], stage = 4, block = 'b')
    X = identitiy_block(X, 3, [64, 64, 128], stage = 4, block = 'c')
    X = identitiy_block(X, 3, [64, 64, 128], stage = 4, block = 'd')
    X = identitiy_block(X, 3, [64, 64, 128], stage = 4, block = 'e')
    X = identitiy_block(X, 3, [64, 64, 128], stage = 4, block = 'f')

    #Stage 5: Avg_Pool --> FC --> Output
    X = AveragePooling2D((2,2))(X)

    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes))(X)

    model = keras.models.Model(inputs = X_input, outputs = X, name = 'ResNet')

    return model








    





