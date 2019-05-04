from keras.applications import DenseNet201
from keras.datasets import cifar100
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
(X_train,y_train) , (X_test,y_test) = cifar100.load_data()

from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)

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
