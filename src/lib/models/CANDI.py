import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.activations import relu

class CANDI(Model):
    def __init__(self):
        super(CANDI, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.layer_1 = Conv2D(64, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.dense_net_1 = self.dense_net(64, initializer)
        self.layer_2_5 = self.CANDI_block(64,3,initializer)
        self.dense_net_2 = self.dense_net(64, initializer)
        self.layer_6_9 = self.CANDI_block(64,3,initializer)
        self.dense_net_3 = self.dense_net(64, initializer)
        self.layer_10_13 = self.CANDI_block(64,3,initializer)
        self.dense_net_4 = self.dense_net(64, initializer)
        self.layer_14_17 = self.CANDI_block(64,3,initializer)
        self.dense_net_5 = self.dense_net(64, initializer)
        self.layer_18_21 = self.CANDI_block(64,3,initializer)
        self.layer_22 = Conv2D(1, 3, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        
    def CANDI_block(self, filters, kernel_size, initializer):
        result = tf.keras.Sequential()
        #1 - conv + bn + relu 
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        
        #2 - conv + bn + relu 
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())

        #3 - conv + bn + relu 
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        
        #4 conv
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))

        return result

    def dense_net(self, filters, initializer):
        result = tf.keras.Sequential()
        result.add(GlobalAveragePooling2D())
        result.add(Dense(filters, activation='relu', kernel_initializer=initializer))
        result.add(Dense(filters, activation='sigmoid', kernel_initializer=initializer))
        #result.add(Reshape((-1,1,1,filters)))
        return result   

    def call(self, input):
        
        layer1 = self.layer_1(input)

        #block 1
        layer_2_5 = self.layer_2_5(layer1)
        attention_1 = self.dense_net_1(layer_2_5)
        attention_1 = tf.reshape(attention_1, [-1,1,1,64])
        layer_2_5 = layer_2_5 * attention_1
        layer_2_5 = tf.keras.layers.Add()([layer_2_5, layer1])
       
        #block 2
        layer_6_9 = self.layer_6_9(layer_2_5)
        attention_2 = self.dense_net_2(layer_6_9)
        attention_2 = tf.reshape(attention_2, [-1,1,1,64])
        layer_6_9 = layer_6_9 * attention_2
        layer_6_9 = tf.keras.layers.Add()([layer_6_9, layer_2_5])   

        #block 3
        layer_10_13 = self.layer_10_13(layer_6_9)
        attention_3 = self.dense_net_3(layer_10_13)
        attention_3 = tf.reshape(attention_3, [-1,1,1,64])
        layer_10_13 = layer_10_13 * attention_3
        layer_10_13 = tf.keras.layers.Add()([layer_10_13, layer_6_9])  

        #block 4
        layer_14_17 = self.layer_14_17(layer_10_13)
        attention_4 = self.dense_net_4(layer_14_17)
        attention_4 = tf.reshape(attention_4, [-1,1,1,64])
        layer_14_17 = layer_14_17 * attention_4
        layer_14_17 = tf.keras.layers.Add()([layer_14_17, layer_10_13]) 

        #block 5
        layer_18_21 = self.layer_18_21(layer_14_17)
        attention_5 = self.dense_net_5(layer_18_21)
        attention_5 = tf.reshape(attention_5, [-1,1,1,64])
        layer_18_21 = layer_18_21 * attention_5
        layer_18_21 = tf.keras.layers.Add()([layer_18_21, layer_14_17]) 

        output = self.layer_22(layer_18_21)

        return output