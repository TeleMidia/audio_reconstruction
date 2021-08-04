import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.activations import relu

class VAE_UNet_v2(Model):
    def __init__(self):
        super(VAE_UNet_v2, self).__init__()

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
        
        self.flatten = Flatten()
        self.dense_mean = Dense(800, name = 'mu')
        self.dense_var = Dense(800, name = 'log_var')
        self.samp_latent = Lambda(self.sampling, name='sampling_latent')
        self.out_latent = Dense(16384, name = 'sampling')
        self.rshp_latent = Reshape((4,4,1024))

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
        self.last_conv = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size, strides_=1):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=strides_, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    
    def sampling(self, args):
        mu, sigma = args
        std = tf.exp(0.5 * sigma)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def call(self, input):
        #128x128x1 ~ input

        #############   ENCODER   ##############
        conv1 = self.conv_1(input)  #128x128x32
        down1 = self.down_1(conv1)  #64x64x32
        
        conv2 = self.conv_2(down1)  #64x64x64
        down2 = self.down_2(conv2)  #32x32x64

        conv3 = self.conv_3(down2)  #32x32x128
        down3 = self.down_3(conv3)  #16x16x128

        conv4 = self.conv_4(down3)  #16x16x256
        down4 = self.down_4(conv4)  #8x8x256

        conv5 = self.conv_5(down4)  #8x8x512
        down5 = self.down_5(conv5)  #4x4x512

        
        conv6 = self.conv_6(down5)  #4x4x1024

        ############# lATENT BLOCK ##############
        latent = self.flatten(conv6)

        mean_mu = self.dense_mean(latent)
        log_var = self.dense_var(latent)

        samp_latent = self.samp_latent([mean_mu, log_var])

        out_latent =  self.out_latent(samp_latent)
        rshp_latent = self.rshp_latent(out_latent)

        #############   DECODER   ##############
        up1 =  self.up_1(rshp_latent) #8x8x512
        concat1 = tf.keras.layers.concatenate([conv5, up1], axis=3) #8x8x1024
        conv7 = self.conv_7(concat1) #8x8x512

        
        up2 =  self.up_2(conv7) #16x16x256
        concat2 = tf.keras.layers.concatenate([conv4, up2], axis=3) #16x16x512
        conv8 = self.conv_8(concat2) #16x16x256

       
        up3 =  self.up_3(conv8) #32x32x128
        concat3 = tf.keras.layers.concatenate([conv3, up3], axis=3) #32x32x256
        conv9 = self.conv_9(concat3) #32x32x128

       
        up4 =  self.up_4(conv9) #64x64x64
        concat4 = tf.keras.layers.concatenate([conv2, up4], axis=3) #64x64x128
        conv10 = self.conv_10(concat4) #64x64x64

       
        up5 =  self.up_5(conv10) #128x128x32
        concat5 = tf.keras.layers.concatenate([conv1, up5], axis=3) #128x128x64
        conv11 = self.conv_11(concat5) #128x128x32

        return self.last_conv(conv11)  #128x128x1


