import tensorflow.keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, Activation
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.losses as losses
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    #this helps autoencoder to learn different sample from same image
    #like mixing it up so when fed a different version of the same humerus it can still learn it right

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, 16))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class encoder_decoder(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,3), multiplier: int = 4, latent_size: int = 16):
        super(encoder_decoder, self).__init__()
        self.upae = upae
        self.multiplier = multiplier
        self.latent_size = latent_size

        self.encoder_layers = []
        self.decoder_layers = []

        # Encoder layers
        self.encoder_layers.append(Conv2D(int(16*multiplier), 4, strides=2, padding='same'))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(32*multiplier), 4, strides=2, padding='same'))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same'))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same'))
        self.encoder_layers.append(Activation('relu'))
        self.encoder_layers.append(BatchNormalization())

        if upae:
            self.encoder_layers.append(Flatten())
            self.encoder_layers.append(Dense(2048, activation='relu'))
            self.encoder_layers.append(Dense(latent_size))
        else:
            self.encoder_layers.append(Flatten())
            self.encoder_layers.append(Dense(2048, activation='relu'))
            self.encoder_layers.append(Dense(latent_size*2))

        # Decoder layers
        self.decoder_layers.append(Dense(2048, activation='relu'))
        self.decoder_layers.append(Dense(int(64 * multiplier) * 16 * 16))
        self.decoder_layers.append(Reshape((16, 16, int(64*multiplier))))
        self.decoder_layers.append(BatchNormalization())

        self.decoder_layers.append(Conv2DTranspose(int(64*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(3, 4, strides=2, padding='same'))
        self.decoder_layers.append(Activation("relu"))

        self.encoder = keras.Sequential(self.encoder_layers, name="encoder")
        self.decoder = keras.Sequential(self.decoder_layers, name="decoder")

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)


class VAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,3), multiplier: int = 4, latent_size: int = 16):
        super(VAE, self).__init__()
        self.encoder_decoder = encoder_decoder(upae=upae, input_shape=input_shape, 
                                               multiplier=multiplier, latent_size=latent_size)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    def call(self, inputs):
        return self.encoder_decoder(inputs)


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    #will run during fit()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            print("Vanilla Loss")
            encoder_output  = self.encoder_decoder.encoder(data)
            reconstruction = self.encoder_decoder.decoder(encoder_output)

            # Reconstructs shape to original size
            reconstruction = tf.image.resize(reconstruction, data.shape[1:3])

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            #getting mean squared error after making data type equal
            mse_loss = tf.reduce_mean(tf.square(tf.cast(data, tf.float32) - tf.cast(reconstruction, tf.float32)))
            total_loss = mse_loss
            

        #calculate gradients using back propagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "mse_loss": self.total_loss_tracker.result() , 
            "binary_crossentropy: ": self.reconstruction_loss_tracker.result()
        }
    

    
        #will run during model.predict()
    def predict(self, data, batch_size=32):
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        predictions = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch_data = data[start:end]

            encoder_output = self.encoder_decoder.encoder(batch_data)
            reconstruction = self.encoder_decoder.decoder(encoder_output)
            predictions.append(reconstruction)

        return np.concatenate(predictions, axis=0)

######################################################################

class UPAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,3), multiplier: int = 4, latent_size: int = 16, 
                 **kwargs):
        super().__init__(**kwargs)
        encoder_decoder.__init__(self, upae=upae, input_shape=input_shape, multiplier=multiplier, 
                                 latent_size=latent_size)
        self.recontruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name="accuracy")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.loss1_tracker = keras.metrics.Mean(name="loss1")
        self.loss2_tracker = keras.metrics.Mean(name="loss2")

    @property
    def metrics(self):
        return [
            self.recontruction_loss_tracker,
            self.accuracy_tracker,
            self.total_loss_tracker,
            self.loss1_tracker,
            self.loss2_tracker
        ]

    #will run during model.fit()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            print("UPAE Training")

            encoder_output  = self.encoder(data)
            reconstruction, z_mean, z_log_var = self.decoder(encoder_output)

            #to be used for training vs validation loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            rec_err = (tf.cast(z_mean, tf.float32) - tf.cast(data, tf.float32)) ** 2
            loss1 = K.mean(K.exp(-z_log_var)*rec_err)
            loss2 = K.mean(z_log_var)
            loss = loss1 + loss2

        #calculate gradients update the weights of the model during backpropagation
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.recontruction_loss_tracker.update_state(reconstruction_loss)
        self.accuracy_tracker.update_state(data, reconstruction)
        self.total_loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss1)
        self.loss2_tracker.update_state(loss2)

        #outputs every epoch
        return {
            "total_loss: ": self.total_loss_tracker.result(),
            "loss1: ": self.loss1_tracker.result(),
            "loss2: ": self.loss2_tracker.result(),
            "binary_crossentropy: ": self.recontruction_loss_tracker.result(),
            "accuracy: ": self.accuracy_tracker.result()
        }

    #will run during model.evaluate()
    def test_step(self, data):
        print("UPAE Validation")
     
        # return {
        #     "total_loss: ": self.total_loss_tracker.result(),
        #     "loss1: ": self.loss1_tracker.result(),
        #     "loss2: ": self.loss2_tracker.result(),
        #     "binary_crossentropy: ": self.recontruction_loss_tracker.result()
        # }
    
    #will run during model.predict()
    def predict(self, data):
        encoder_output = self.encoder(data)
        reconstruction, z_mean , z_logvar = self.decoder(encoder_output)
        return reconstruction
        