import os
import tensorflow as tf
prtnt(tf.test.is_gpu_available())
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
from keras.datasets import cifar100
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
(X_train,y_train) , (X_test,y_test) = cifar100.load_data()

def res_block():
    inp=Input(shape=(None,None,64))
    x = Conv2D(64,3,activation='relu',padding='same')(inp)
    x = Conv2D(64,3,activation='relu',padding='same')(x)
    x = Add()([x,inp])
    return Model(inp,Activation('relu')(x))

inp=Input(shape=(32,32,3))
x = Conv2D(64,3,activation='relu',padding='same')(inp)
x = res_block()(x)
x = res_block()(x)
y = Conv2D(64,3,activation='relu',padding='same')(inp)
y = res_block()(y)
y = res_block()(y)
x = Add()([x,y])
x = Activation('relu')(x)
y = res_block()(y)
y = res_block()(y)
x = res_block()(x)
x = res_block()(x)
x = Add()([x,y])
x = Activation('relu')(x)
x = res_block()(x)
x = res_block()(x)
x = Add()([x,y])
x = Flatten()(x)
x = Dense(100,activation='softmax')(x)
model = Model(inp,x)
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()
model.fit(X_train,to_categorical(y_train),validation_split=0.2,epochs=20)
