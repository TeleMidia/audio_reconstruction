import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D, UpSampling2D
from tensorflow.keras.activations import relu

class FPN_RFB_TDM(Model):
    def __init__(self):
        super(FPN_RFB_TDM, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.upsample_2x = UpSampling2D(size=(2, 2), interpolation="nearest")

        self.conv_1 = self.Conv2dBatchLayer(32,3)
        self.conv_2 = self.Conv2dBatchLayer(64,3)
        self.conv_3 = self.Conv2dBatchLayer(128,3)
        self.conv_4 = self.Conv2dBatchLayer(256,3)
        self.conv_5 = self.Conv2dBatchLayer(512,3)

        self.side_conv_1 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_2 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_3 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_4 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)

        self.side_conv_4_1 = self.Conv2dBatchLayer_2x(32, 3)

        self.conv_7 = self.Conv2dBatchLayer(32,3)
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    
    def Conv2dBatchLayer_2x(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result


    def call(self, input):
        
        #224 -> 112
        conv1 = self.conv_1(input)
        down1 = self.max_pool(conv1)

        #112 -> 56
        conv2 = self.conv_2(down1)
        down2 = self.max_pool(conv2)

        #56 -> 28
        conv3 = self.conv_3(down2)
        down3 = self.max_pool(conv3)

        #28 -> 14
        conv4 = self.conv_4(down3)
        down4 = self.max_pool(conv4)

        #14 -> 7
        conv5 = self.conv_5(down4)
        down5 = self.max_pool(conv5)

        #7 -> 14
        side1 = self.side_conv_1(down5)
        up1 = self.upsample_2x(side1)

        #14 -> 28
        side2 = self.side_conv_2(down4)
        add1 = tf.keras.layers.concatenate([side2,  up1], axis=3)
        up2 = self.upsample_2x(add1)        

        #28 -> 56
        side3 = self.side_conv_3(down3)
        add2 = tf.keras.layers.concatenate([side3,  up2], axis=3)
        up3 = self.upsample_2x(add2)

        #56 -> 112
        side4 = self.side_conv_4(down2)
        add3 = tf.keras.layers.concatenate([side4,  up3], axis=3)
        up4 = self.upsample_2x(add3)

        #112 -> 224
        side_4_1 = self.side_conv_4_1(up4)
        up_f_2x = self.upsample_2x(side_4_1)
       
         
        conv7 = self.conv_7(up_f_2x)
       
        return self.last_conv(conv7)  