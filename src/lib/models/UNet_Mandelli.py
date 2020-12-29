import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, LeakyReLU, Conv2DTranspose, MaxPool2D

class UNet_Mandelli(Model):
    def __init__(self):
        super(UNet_Mandelli, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)

        self.conv_1 = self.Conv2dBatchLayer(64,4, useBN=False)
        self.conv_2 = self.Conv2dBatchLayer(128,4)
        self.conv_3 = self.Conv2dBatchLayer(256,4)
        self.conv_4 = self.Conv2dBatchLayer(512,4)
        self.conv_5 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer=initializer)

        self.up_1 = self.Deconv2dBatchLayer(512, 4)
        self.up_2 = self.Deconv2dBatchLayer(256, 4)
        self.up_3 = self.Deconv2dBatchLayer(128, 4)
        self.up_4 = self.Deconv2dBatchLayer(64, 4)
        
        self.last_conv = self.Deconv2dBatchLayer(1, 4, useBN=False)

    def Conv2dBatchLayer(self, filters, kernel_size, useBN=True):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=2, padding='same', kernel_initializer=initializer))
        if useBN:
            result.add(BatchNormalization())
        
        result.add(LeakyReLU())
        return result
    
    def Deconv2dBatchLayer(self, filters, kernel_size, useBN=True):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(ReLU())
        result.add(Conv2DTranspose(filters, kernel_size, strides=2, padding='same', kernel_initializer=initializer))
        if useBN:
            result.add(BatchNormalization())
        return result

    def call(self, input):
        
        #224 -> 112
        conv1 = self.conv_1(input)

        #112 -> 56
        conv2 = self.conv_2(conv1)

        #56 -> 28
        conv3 = self.conv_3(conv2)

        #28 -> 14
        conv4 = self.conv_4(conv3)

        #14 -> 7
        conv5 = self.conv_5(conv4)

        #7 -> 14     
        up1 =  self.up_1(conv5)
        concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)
        
        #14 -> 28
        up2 =  self.up_2(concat1)
        concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)
        
        #28 -> 56
        up3 =  self.up_3(concat2)
        concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)
      
        #56 -> 112
        up4 =  self.up_4(concat3)
        concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)
       
        #112 -> 224 
        return self.last_conv(concat4)
        
