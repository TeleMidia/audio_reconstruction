import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D, UpSampling2D
from tensorflow.keras.activations import relu

class UNet_Sun(Model):
    def __init__(self):
        super(UNet_Sun, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.upsample = UpSampling2D(size=(2, 2))
        self.conv_1 = self.Conv2dBatchLayer(16,6)
        self.conv_2 = self.Conv2dBatchLayer(32,6)
        self.conv_3 = self.Conv2dBatchLayer(32,4)
        self.conv_4 = self.Conv2dBatchLayer(32,3)
        self.conv_5 = self.Conv2dBatchLayer(16,3)
        self.conv_6 = self.Conv2dBatchLayer(8,3)
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    

    def call(self, input):
        
        #200x200x1 -> 100x100x16
        conv1 = self.conv_1(input)
        down1 = self.max_pool(conv1)

        #100x100x16 -> 50x50x32
        conv2 = self.conv_2(down1)
        down2 = self.max_pool(conv2)

        # 50x50x32 -> 50x50x32
        conv3 = self.conv_3(down2)
        conv4 = self.conv_4(conv3)
       
        up1 = self.upsample(conv4)
        sum1 = tf.keras.layers.Add()([up1, conv2])
        conv5 = self.conv_5(sum1)

        up2 = self.upsample(conv5)
        sum2 = tf.keras.layers.Add()([up2, conv1])
        conv6 = self.conv_6(sum2)

        return self.last_conv(conv6)  