import keras
from keras import layers, models
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, Activation
from keras.metrics import AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from keras import backend as K
from keras.models import Model
import numpy as np
from keras import backend as K

class Autoencoder:
    def __init__(self, input_shape, multiplier, latentDim):
        super(Autoencoder, self).__init__()
        
        input_layer = Input(shape=input_shape)
        x = Conv2D(int(16*multiplier), 4, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(32*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(64*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(64*multiplier), 4, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        volumeSize = K.int_shape(x)

        #Latent representation Encoder
        x = Flatten()(x)
        latent_enc = Dense(latentDim, activation='relu')(x)
        self.encoder = Model(input_layer, latent_enc, name="encoder")
        
        #Latent representation Decoder
        latentInputs = Input(shape=(latentDim,))
        latent_dec = Dense(np.prod(volumeSize[1:]))(latentInputs)
        latent_dec = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(latent_dec)

        x = Conv2DTranspose(int(64*multiplier), 4, strides=2, padding='same')(latent_dec)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
        outputs = Activation("relu")(x)
        volumeSize = K.int_shape(x)
                    
        self.decoder = Model(latentInputs, outputs, name="decoder")


        self.autoencoder = Model(input_layer, self.decoder(self.encoder(input_layer)),
            name="autoencoder")
    

    def compile_AE(self):
        self.autoencoder.compile(optimizer = 'adam', loss='mse',
                     metrics= ['mse', 
                               AUC(name="AUC"),
                               Precision(name="Precision"),
                               Recall(name='Recall'),
                               TruePositives(name="True Positives"),
                               TrueNegatives(name="True Negatives"),
                               FalsePositives(name="False Positives"),
                               FalseNegatives(name="False Negatives")])
        return
        
    def fit_AE(self, x_train, y_train, epochs = 1, batch_size = 32):
        return self.autoencoder.fit(x_train, y_train, epochs = epochs, batch_size=batch_size)
