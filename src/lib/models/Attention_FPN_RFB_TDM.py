import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D, UpSampling2D
from tensorflow.keras.activations import relu

class Attention_FPN_RFB_TDM(Model):
    def __init__(self):
        super(Attention_FPN_RFB_TDM, self).__init__()

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
        self.x_conv2 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.g_conv2 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.conv_r2 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)

        self.side_conv_3 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.x_conv3 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.g_conv3 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.conv_r3 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)

        self.side_conv_4 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.x_conv4 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.g_conv4 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.conv_r4 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)

        self.side_conv_4_1 = self.Conv2dBatchLayer_2x(32, 3)
        
        self.conv_6 = self.Conv2dBatchLayer(32,3)
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
        
        #128x128x1 -> 64x64x32
        conv1 = self.conv_1(input)
        down1 = self.max_pool(conv1)

        #64x64x32 -> 32x32x64
        conv2 = self.conv_2(down1)
        down2 = self.max_pool(conv2)

        #32x32x64 -> 16x16x128
        conv3 = self.conv_3(down2)
        down3 = self.max_pool(conv3)

        #16x16x128 -> 8x8x256
        conv4 = self.conv_4(down3)
        down4 = self.max_pool(conv4)

        #8x8x256 -> 4x4x512
        conv5 = self.conv_5(down4)
        down5 = self.max_pool(conv5)

        #4x4x512 -> 8x8x256
        side1 = self.side_conv_1(down5)
        up1 = self.upsample_2x(side1)

        #8x8x256 -> 16x16x128
        side2 = self.side_conv_2(down4)
        
        x_conv2 = self.x_conv2(side2)
        g_conv2 = self.g_conv2(up1)
        Add_2 = tf.keras.layers.add([x_conv2, g_conv2])
        actv_r2 = tf.keras.activations.relu(Add_2)
        conv_r2 = self.conv_r2(actv_r2)
        actv_s2 = tf.keras.activations.sigmoid(conv_r2)
        mult_2 = tf.keras.layers.multiply([side2, actv_s2])

        add1 = tf.keras.layers.concatenate([mult_2,  up1], axis=3)
        up2 = self.upsample_2x(add1)        

        #16x16x128 -> 32x32x64
        side3 = self.side_conv_3(down3)

        x_conv3 = self.x_conv3(side3)
        g_conv3 = self.g_conv3(up2)
        Add_3 = tf.keras.layers.add([x_conv3, g_conv3])
        actv_r3 = tf.keras.activations.relu(Add_3)
        conv_r3 = self.conv_r3(actv_r3)
        actv_s3 = tf.keras.activations.sigmoid(conv_r3)
        mult_3 = tf.keras.layers.multiply([side3, actv_s3])

        add2 = tf.keras.layers.concatenate([mult_3,  up2], axis=3)
        up3 = self.upsample_2x(add2)

        #32x32x64 -> 64x64x32
        side4 = self.side_conv_4(down2)

        x_conv4 = self.x_conv4(side4)
        g_conv4 = self.g_conv4(up3)
        Add_4 = tf.keras.layers.add([x_conv4, g_conv4])
        actv_r4 = tf.keras.activations.relu(Add_4)
        conv_r4 = self.conv_r4(actv_r4)
        actv_s4 = tf.keras.activations.sigmoid(conv_r4)
        mult_4 = tf.keras.layers.multiply([side4, actv_s4])

        add3 = tf.keras.layers.concatenate([mult_4,  up3], axis=3)
        up4 = self.upsample_2x(add3)

        #64x64x32 -> 128x128x32
        side_4_1 = self.side_conv_4_1(up4)
        up_f_2x = self.upsample_2x(side_4_1)
        conv6 = self.conv_6(up_f_2x)
        
        #128x128x1
        return self.last_conv(conv6)