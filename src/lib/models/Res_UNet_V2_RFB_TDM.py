import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose
from tensorflow.keras.activations import relu

class Res_UNet_V2_RFB_TDM(Model):
    def __init__(self):
        super(Res_UNet_V2_RFB_TDM, self).__init__()

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
        self.up_1 = Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_7 = self.Conv2dBatchLayer(512,3)
        self.up_2 = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_8 = self.Conv2dBatchLayer(256,3)
        self.up_3 = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_9 = self.Conv2dBatchLayer(128,3)
        self.up_4 = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_10 = self.Conv2dBatchLayer(64,3)
        self.up_5 = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_11 = self.Conv2dBatchLayer(32, 3)

        self.side_conv_1 = Conv2D(512, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_2 = Conv2D(256, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_3 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_4 = Conv2D(64, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.side_conv_5 = Conv2D(32, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)

        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size, strides_=1):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=strides_, padding='same', kernel_initializer=initializer))
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

        
        #3x3x1024 -> 6x6x512
        up1 =  self.up_1(conv6)
        side1 = self.side_conv_1(conv5)
        fusion1 = tf.keras.layers.concatenate([side1, up1], axis=3)
        conv7 = self.conv_7(fusion1)

        #6x6x512 -> 12x12x246
        up2 =  self.up_2(conv7)
        side2 = self.side_conv_2(conv4)
        fusion2 = tf.keras.layers.concatenate([side2, up2], axis=3)
        conv8 = self.conv_8(fusion2)

        #12x12x246 -> 24x24x128
        up3 =  self.up_3(conv8)
        side3 = self.side_conv_3(conv3)
        fusion3 = tf.keras.layers.concatenate([side3, up3], axis=3)
        conv9 = self.conv_9(fusion3)

        #24x24x128 -> 48x48x64
        up4 =  self.up_4(conv9)
        side4 = self.side_conv_4(conv2)
        fusion4 = tf.keras.layers.concatenate([side4, up4], axis=3)
        conv10 = self.conv_10(fusion4)

        #48x48x64 -> 96x96x32
        up5 =  self.up_5(conv10)
        side5 = self.side_conv_5(conv1)
        fusion5 = tf.keras.layers.concatenate([side5, up5], axis=3)
        conv11 = self.conv_11(fusion5)

        last_conv = self.last_conv(conv11)

        return tf.keras.layers.Subtract()([input, last_conv])