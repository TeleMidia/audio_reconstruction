import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D, Activation, Add, Multiply
from tensorflow.keras.activations import relu

class Res_Attention_UNet(Model):
    def __init__(self):
        super(Res_Attention_UNet, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.conv_1 = self.Conv2dBatchLayer(32,3)
        self.conv_2 = self.Conv2dBatchLayer(64,3)
        self.conv_3 = self.Conv2dBatchLayer(128,3)
        self.conv_4 = self.Conv2dBatchLayer(256,3)
        self.conv_5 = self.Conv2dBatchLayer(512,3)
        self.conv_6 = self.Conv2dBatchLayer(1024,3)
        self.up_1 = Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.x_conv1 = Conv2D(512, 1, strides=1, kernel_initializer=initializer)
        self.g_conv1 = Conv2D(512, 1, strides=1, kernel_initializer=initializer)
        self.conv_r1 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)
        self.conv_7 = self.Conv2dBatchLayer(512,3)
        self.up_2 = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.x_conv2 = Conv2D(256, 1, strides=1, kernel_initializer=initializer)
        self.g_conv2 = Conv2D(256, 1, strides=1, kernel_initializer=initializer)
        self.conv_r2 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)
        self.conv_8 = self.Conv2dBatchLayer(256,3)
        self.up_3 = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.x_conv3 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.g_conv3 = Conv2D(128, 1, strides=1, kernel_initializer=initializer)
        self.conv_r3 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)
        self.conv_9 = self.Conv2dBatchLayer(128,3)
        self.up_4 = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.x_conv4 = Conv2D(64, 1, strides=1, kernel_initializer=initializer)
        self.g_conv4 = Conv2D(64, 1, strides=1, kernel_initializer=initializer)
        self.conv_r4 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)
        self.conv_10 = self.Conv2dBatchLayer(64,3)
        self.up_5 = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.x_conv5 = Conv2D(32, 1, strides=1, kernel_initializer=initializer)
        self.g_conv5 = Conv2D(32, 1, strides=1, kernel_initializer=initializer)
        self.conv_r5 = Conv2D(1, 1, strides=1, kernel_initializer=initializer)
        self.conv_11 = self.Conv2dBatchLayer(32, 3)
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)


    def Conv2dBatchLayer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result

    def call(self, input):
        
        #96x96x3 -> 48x48x32
        conv1 = self.conv_1(input)
        down1 = self.max_pool(conv1)

        #48x48x32 -> 24x24x64
        conv2 = self.conv_2(down1)
        down2 = self.max_pool(conv2)

        #24x24x64 -> 12x12x128
        conv3 = self.conv_3(down2)
        down3 = self.max_pool(conv3)

        #12x12x128 -> 6x6x256
        conv4 = self.conv_4(down3)
        down4 = self.max_pool(conv4)

        #6x6x256 -> 3x3x512
        conv5 = self.conv_5(down4)
        down5 = self.max_pool(conv5)

        #3x3x512 -> 3x3x1024
        conv6 = self.conv_6(down5)

        #3x3x1024 -> 6x6x512
        up1 =  self.up_1(conv6)
        x_conv1 = self.x_conv1(conv5)
        g_conv1 = self.g_conv1(up1)
        Add_1 = tf.keras.layers.add([x_conv1, g_conv1])
        actv_r1 = tf.keras.activations.relu(Add_1)
        conv_r1 = self.conv_r1(actv_r1)
        actv_s1 = tf.keras.activations.sigmoid(conv_r1)
        mult_1 = tf.keras.layers.multiply([conv5, actv_s1])
        concat1 = tf.keras.layers.concatenate([mult_1, up1], axis=3)
        conv7 = self.conv_7(concat1)

        #6x6x512 -> 12x12x246
        up2 =  self.up_2(conv7)
        x_conv2 = self.x_conv2(conv4)
        g_conv2 = self.g_conv2(up2)
        Add_2 = tf.keras.layers.add([x_conv2, g_conv2])
        actv_r2 = tf.keras.activations.relu(Add_2)
        conv_r2 = self.conv_r2(actv_r2)
        actv_s2 = tf.keras.activations.sigmoid(conv_r2)
        mult_2 = tf.keras.layers.multiply([conv4, actv_s2])
        concat2 = tf.keras.layers.concatenate([mult_2, up2], axis=3)
        conv8 = self.conv_8(concat2)

        #12x12x246 -> 24x24x128
        up3 =  self.up_3(conv8)
        x_conv3 = self.x_conv3(conv3)
        g_conv3 = self.g_conv3(up3)
        Add_3 = tf.keras.layers.add([x_conv3, g_conv3])
        actv_r3 = tf.keras.activations.relu(Add_3)
        conv_r3 = self.conv_r3(actv_r3)
        actv_s3 = tf.keras.activations.sigmoid(conv_r3)
        mult_3 = tf.keras.layers.multiply([conv3, actv_s3])
        concat3 = tf.keras.layers.concatenate([mult_3, up3], axis=3)
        conv9 = self.conv_9(concat3)

        #24x24x128 -> 48x48x64
        up4 =  self.up_4(conv9)
        x_conv4 = self.x_conv4(conv2)
        g_conv4 = self.g_conv4(up4)
        Add_4 = tf.keras.layers.add([x_conv4, g_conv4])
        actv_r4 = tf.keras.activations.relu(Add_4)
        conv_r4 = self.conv_r4(actv_r4)
        actv_s4 = tf.keras.activations.sigmoid(conv_r4)
        mult_4 = tf.keras.layers.multiply([conv2, actv_s4])
        concat4 = tf.keras.layers.concatenate([mult_4, up4], axis=3)
        conv10 = self.conv_10(concat4)

        #48x48x64 -> 96x96x32
        up5 =  self.up_5(conv10)
        x_conv5 = self.x_conv5(conv1)
        g_conv5 = self.g_conv5(up5)
        Add_5 = tf.keras.layers.add([x_conv5, g_conv5])
        actv_r5 = tf.keras.activations.relu(Add_5)
        conv_r5 = self.conv_r5(actv_r5)
        actv_s5 = tf.keras.activations.sigmoid(conv_r5)
        mult_5 = tf.keras.layers.multiply([conv1, actv_s5])
        concat5 = tf.keras.layers.concatenate([mult_5, up5], axis=3)
        conv11 = self.conv_11(concat5)

        #96x96x32 -> 96x96x1
        last_conv = self.last_conv(conv11) 

        return tf.keras.layers.Subtract()([input, last_conv])