import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D
from tensorflow.keras.activations import relu

class Dual_Res_UNet_v2(Model):
    def __init__(self):
        super(Dual_Res_UNet_v2, self).__init__()

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
        self.mid_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

        self.conv_12 = self.Conv2dBatchLayer(32,3)
        self.down_6 = self.Conv2dBatchLayer(32,3, strides_=2)
        self.conv_13 = self.Conv2dBatchLayer(64,3)
        self.down_7 = self.Conv2dBatchLayer(64,3, strides_=2)
        self.conv_14 = self.Conv2dBatchLayer(128,3)
        self.down_8 = self.Conv2dBatchLayer(128,3, strides_=2)
        self.conv_15 = self.Conv2dBatchLayer(256,3)
        self.down_9 = self.Conv2dBatchLayer(256,3, strides_=2)
        self.conv_16 = self.Conv2dBatchLayer(512,3)
        self.down_10 = self.Conv2dBatchLayer(512,3, strides_=2)
        self.conv_17 = self.Conv2dBatchLayer(1024,3)
        self.up_6 = Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_18 = self.Conv2dBatchLayer(512,3)
        self.up_7 = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_19 = self.Conv2dBatchLayer(256,3)
        self.up_8 = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_20 = self.Conv2dBatchLayer(128,3)
        self.up_9 = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_21 = self.Conv2dBatchLayer(64,3)
        self.up_10 = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_22 = self.Conv2dBatchLayer(32, 3)
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size, strides_=1):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=strides_, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    

    def call(self, input):
        
        ##RECONSTRUCTION##
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
        concat1 = tf.keras.layers.concatenate([conv5, up1], axis=3)
        conv7 = self.conv_7(concat1)

        up2 =  self.up_2(conv7)
        concat2 = tf.keras.layers.concatenate([conv4, up2], axis=3)
        conv8 = self.conv_8(concat2)

        up3 =  self.up_3(conv8)
        concat3 = tf.keras.layers.concatenate([conv3, up3], axis=3)
        conv9 = self.conv_9(concat3)

        up4 =  self.up_4(conv9)
        concat4 = tf.keras.layers.concatenate([conv2, up4], axis=3)
        conv10 = self.conv_10(concat4)

        up5 =  self.up_5(conv10)
        concat5 = tf.keras.layers.concatenate([conv1, up5], axis=3)
        conv11 = self.conv_11(concat5)

        mid_conv = self.mid_conv(conv11)
        mid_out = tf.keras.layers.Subtract()([input, mid_conv])
        
        ##REFINEMENT##
        conv12 = self.conv_12(mid_conv)
        down6 = self.down_6(conv12)

       
        conv13 = self.conv_13(down6)
        down7 = self.down_7(conv13)

       
        conv14 = self.conv_14(down7)
        down8 = self.down_8(conv14)

       
        conv15 = self.conv_15(down8)
        down9 = self.down_9(conv15)

       
        conv16 = self.conv_16(down9)
        down10 = self.down_10(conv16)

        conv17 = self.conv_17(down10)

        up6 =  self.up_6(conv17)
        concat6 = tf.keras.layers.concatenate([conv16, up6], axis=3)
        conv18 = self.conv_18(concat6)

        up7 =  self.up_7(conv18)
        concat7 = tf.keras.layers.concatenate([conv15, up7], axis=3)
        conv19 = self.conv_19(concat7)

        up8 =  self.up_8(conv19)
        concat8 = tf.keras.layers.concatenate([conv14, up8], axis=3)
        conv20 = self.conv_20(concat8)

        up9 =  self.up_9(conv20)
        concat9 = tf.keras.layers.concatenate([conv13, up9], axis=3)
        conv21 = self.conv_21(concat9)

        up10 =  self.up_10(conv21)
        concat10 = tf.keras.layers.concatenate([conv12, up10], axis=3)
        conv22 = self.conv_22(concat10)

        last_conv = self.last_conv(conv22)
        
        output = tf.keras.layers.Subtract()([input, last_conv])
        
        return [mid_out, output]