import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense

class GRes_SE_ResNet(Model):
    def __init__(self):
        super(GRes_SE_ResNet, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        
        self.layer_1 = Conv2D(64, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.layer_2_3 = self.ResNet_block(64, 3, initializer)
        self.attention_1 = self.dense_net(64, initializer)
        self.layer_4_5 = self.ResNet_block(64, 3, initializer)
        self.attention_2 = self.dense_net(64, initializer)
        self.layer_6_7 = self.ResNet_block(64, 3, initializer)
        self.attention_3 = self.dense_net(64, initializer)
        self.layer_8_9 = self.ResNet_block(64, 3, initializer)
        self.attention_4 = self.dense_net(64, initializer)
        self.layer_10_11 = self.ResNet_block(64, 3, initializer)
        self.attention_5 = self.dense_net(64, initializer)
        self.layer_12_13 = self.ResNet_block(64, 3, initializer)
        self.attention_6 = self.dense_net(64, initializer)
        self.layer_14_15 = self.ResNet_block(64, 3, initializer)
        self.attention_7 = self.dense_net(64, initializer)
        self.layer_16_17 = self.ResNet_block(64, 3, initializer)
        self.attention_8 = self.dense_net(64, initializer)
        self.layer_18_19 = self.ResNet_block(64, 3, initializer)
        self.attention_9 = self.dense_net(64, initializer)
        self.layer_20 = Conv2D(1, 3, activation=None, strides=1, padding='same', kernel_initializer=initializer)

        self.relu_1 = ReLU()
        self.relu_2 = ReLU() 
        self.relu_3 = ReLU() 
        self.relu_4 = ReLU() 
        self.relu_5 = ReLU() 
        self.relu_6 = ReLU() 
        self.relu_7 = ReLU() 
        self.relu_8 = ReLU() 
        self.relu_9 = ReLU()  
        
    def ResNet_block(self, filters, kernel_size, initializer):

        result = tf.keras.Sequential() 
    
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())

        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
       
        return result


    def dense_net(self, filters, initializer):
        result = tf.keras.Sequential()
        result.add(GlobalAveragePooling2D())
        result.add(Dense(filters/2, activation='relu', kernel_initializer=initializer))
        result.add(Dense(filters, activation='sigmoid', kernel_initializer=initializer))
        return result  


    def call(self, input):
        
        x = self.layer_1(input)
        
        skip = x
        x = self.layer_2_3(x)
        attention_1 = self.attention_1(x)
        attention_1 = tf.reshape(attention_1, [-1,1,1,64])
        x = x * attention_1
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_1(x)

        skip = x
        x = self.layer_4_5(x)
        attention_2 = self.attention_2(x)
        attention_2 = tf.reshape(attention_2, [-1,1,1,64])
        x = x * attention_2
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_2(x)

        skip = x
        x = self.layer_6_7(x)
        attention_3 = self.attention_3(x)
        attention_3 = tf.reshape(attention_3, [-1,1,1,64])
        x = x * attention_3
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_3(x)
        
        skip = x
        x = self.layer_8_9(x)
        attention_4 = self.attention_4(x)
        attention_4 = tf.reshape(attention_4, [-1,1,1,64])
        x = x * attention_4
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_4(x)

        skip = x
        x = self.layer_10_11(x)
        attention_5 = self.attention_5(x)
        attention_5 = tf.reshape(attention_5, [-1,1,1,64])
        x = x * attention_5
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_5(x)

        skip = x
        x = self.layer_12_13(x)
        attention_6 = self.attention_6(x)
        attention_6 = tf.reshape(attention_6, [-1,1,1,64])
        x = x * attention_6
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_6(x)

        skip = x
        x = self.layer_14_15(x)
        attention_7 = self.attention_7(x)
        attention_7 = tf.reshape(attention_7, [-1,1,1,64])
        x = x * attention_7
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_7(x)

        skip = x
        x = self.layer_16_17(x)
        attention_8 = self.attention_8(x)
        attention_8 = tf.reshape(attention_8, [-1,1,1,64])
        x = x * attention_8
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_8(x)

        skip = x
        x = self.layer_18_19(x)
        attention_9 = self.attention_9(x)
        attention_9 = tf.reshape(attention_9, [-1,1,1,64])
        x = x * attention_9
        x = tf.keras.layers.Add()([skip, x])
        x = self.relu_9(x)        

        x = self.layer_20(x)


        return tf.keras.layers.Subtract()([input, x])   