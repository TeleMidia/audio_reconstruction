import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D

class LinkNet_v2(Model):
    def __init__(self):
        super(LinkNet_v2, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.conv_1 = self.Conv2dBatchLayer(32,3)
        self.down_1 = self.Conv2dBatchLayer(32,3, strides_=2)
        self.conv_2 = self.Conv2dBatchLayer(64,3)
        self.down_2 = self.Conv2dBatchLayer(64,3, strides_=2)
        self.conv_3 = self.Conv2dBatchLayer(128,3)
        self.down_3 = self.Conv2dBatchLayer(128,3, strides_=2)
        self.conv_4 = self.Conv2dBatchLayer(256,3)
        self.down_4 = self.Conv2dBatchLayer(256,3, strides_=2)
        self.conv_5 = self.Conv2dBatchLayer(512,3)
        self.down_5 = self.Conv2dBatchLayer(512,3, strides_=2)
        self.conv_6 = self.Conv2dBatchLayer(1024,3)
        self.up_1 = self.Conv2dTransposeBatchLayer(1024, 512, 4, strides_=2)
        self.conv_7 = self.Conv2dBatchLayer(512,3)
        self.up_2 = self.Conv2dTransposeBatchLayer(512, 256, 4, strides_=2)
        self.conv_8 = self.Conv2dBatchLayer(256,3)
        self.up_3 = self.Conv2dTransposeBatchLayer(256, 128, 4, strides_=2)
        self.conv_9 = self.Conv2dBatchLayer(128,3)
        self.up_4 = self.Conv2dTransposeBatchLayer(128, 64, 4, strides_=2)
        self.conv_10 = self.Conv2dBatchLayer(64,3)
        self.up_5 = self.Conv2dTransposeBatchLayer(64, 32, 4, strides_=2)
        self.conv_11 = self.Conv2dBatchLayer(32, 3)
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size, strides_=1):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=strides_, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    
    def Conv2dTransposeBatchLayer(self, input_filters, output_filters, kernel_size, strides_=2):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(input_filters // 4, 1, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        result.add(Conv2DTranspose(input_filters // 4, kernel_size, strides=strides_, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        result.add(Conv2D(output_filters, 1, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    

    def call(self, input):
        conv1 = self.conv_1(input)
        down1 = self.down_1(conv1)

       
        conv2 = self.conv_2(down1)
        down2 = self.down_2(conv2)

       
        conv3 = self.conv_3(down2)
        down3 = self.down_3(conv3)

       
        conv4 = self.conv_4(down3)
        down4 = self.down_4(conv4)

       
        conv5 = self.conv_5(down4)
        down5 = self.down_5(conv5)

       
        conv6 = self.conv_6(down5)

        
        up1 =  self.up_1(conv6)
        add1 = tf.keras.layers.add([conv5, up1])
        conv7 = self.conv_7(add1)

        
        up2 =  self.up_2(conv7)
        add2 = tf.keras.layers.add([conv4, up2])
        conv8 = self.conv_8(add2)

       
        up3 =  self.up_3(conv8)
        add3 = tf.keras.layers.add([conv3, up3])
        conv9 = self.conv_9(add3)

       
        up4 =  self.up_4(conv9)
        add4 = tf.keras.layers.add([conv2, up4])
        conv10 = self.conv_10(add4)

       
        up5 =  self.up_5(conv10)
        add5 = tf.keras.layers.add([conv1, up5])
        conv11 = self.conv_11(add5)

        
        return self.last_conv(conv11)