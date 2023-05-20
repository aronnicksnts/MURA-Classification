import tensorflow.keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, Activation
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import GlorotUniform
import tensorflow.keras.losses as losses
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

class encoder_decoder(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,1), multiplier: int = 4, latent_size: int = 16):
        super(encoder_decoder, self).__init__()
        self.upae = upae
        
        self.multiplier = multiplier
        self.latent_size = latent_size
        self.fm = input_shape[0] // 16
        self.encoder_layers = []
        self.latent_encoder_layers = []
        self.latent_decoder_layers = []
        self.decoder_layers = []
        self.out_channels = 2 if self.upae else 1

        # Encoder layers
        self.encoder_layers.append(Conv2D(int(16*multiplier), 4, strides=2, padding='same', use_bias=False))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(32*multiplier), 4, strides=2, padding='same', use_bias=False))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same', use_bias=False))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same', use_bias=False))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Reshape((int(64*multiplier) * self.fm * self.fm,)))
        # TODO: CHANGE TO LINEAR SPACE REP FOR ENC, ADD THE Z_MEAN AND Z_LOG_VAR FOR BOTH
        
        if upae:
            self.latent_encoder_layers.append(Dense(2048, activation='relu', 
                                                          kernel_initializer=GlorotUniform(),
                                                          use_bias=False))
            self.latent_encoder_layers.append(BatchNormalization())
            self.latent_encoder_layers.append(Activation('relu'))
            self.latent_encoder_layers.append(Dense(latent_size, 
                                                          kernel_initializer=GlorotUniform(),
                                                          use_bias=False))
        else:
            self.latent_encoder_layers.append(Dense(2048, activation='relu', 
                                                          kernel_initializer=GlorotUniform(),
                                                          use_bias=False))
            self.latent_encoder_layers.append(BatchNormalization())
            self.latent_encoder_layers.append(Activation('relu'))
            self.latent_encoder_layers.append(Dense(latent_size*2, 
                                                          kernel_initializer=GlorotUniform(),
                                                          use_bias=False))
            
        self.latent_decoder_layers.append(Dense(2048, kernel_initializer=GlorotUniform(),
        use_bias=False))
        self.latent_decoder_layers.append(BatchNormalization())
        self.latent_decoder_layers.append(Activation('relu'))
        self.latent_decoder_layers.append(Dense(int(64 * multiplier) * self.fm * self.fm, 
                                                      kernel_initializer=GlorotUniform(),
                                                      use_bias=False))
        
        self.latent_decoder_layers.append(Reshape((self.fm, self.fm, int(64*multiplier))))
        
        # Decoder layers
        self.decoder_layers.append(Conv2DTranspose(int(64*multiplier), 2, strides=2, padding='same',
                                                   kernel_initializer=GlorotUniform()))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(self.out_channels, 4, strides=2, padding='same',
                                                   kernel_initializer=GlorotUniform()))

        # Building of the Sequential Model
        self.encoder = keras.Sequential(self.encoder_layers, name="encoder")

        self.latent_encoder = keras.Sequential(self.latent_encoder_layers,
                                               name="latent_encoder")
        
        self.latent_decoder = keras.Sequential(self.latent_decoder_layers,
                                               name="latent_decoder")
        
        self.decoder = keras.Sequential(self.decoder_layers, name="decoder")

    def build(self, input_shape):
        super(encoder_decoder, self).build(input_shape)
        print("INPUT SHAPE ACCEPTED: ", input_shape)
        self.decoder.build(input_shape)
        self.latent_encoder.build(input_shape)  # Build the latent_decoder model

    def call(self, inputs):
        # Encode
        encoder_output = self.encoder(inputs)
        latent_vectors = self.latent_encoder(encoder_output)
        
        # Decode
        # latent_vectors = self._sample_latent(z_mean, z_log_var)
        latent_vectors = self.latent_decoder(latent_vectors)
        reconstructed = self.decoder(latent_vectors)

        z_mean = 0

        z_log_var = 0

        return reconstructed, z_mean, z_log_var
    
    def _sample_latent(self, z_mean, z_log_var):
        # batch_size = tf.shape(z_mean)[0]
        # latent_dim = tf.shape(z_mean)[3]
        # epsilon = tf.random.normal(shape=(batch_size, 16, 16, latent_dim))

        # sampled_latent = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # return sampled_latent
        return 0


class VAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,1), multiplier: int = 4, latent_size: int = 16):
        super(VAE, self).__init__()
        self.encoder_decoder = encoder_decoder(upae=upae, input_shape=input_shape, 
                                               multiplier=multiplier, latent_size=latent_size)
        
        # Convert input_shape to tuple
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)

        self.encoder_decoder.build(input_shape=(None,) + input_shape)  # Build the encoder_decoder model
        self.encoder_decoder.summary()
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        return self.encoder_decoder(inputs)

    @property
    def metrics(self):
        return [
            self.mse_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    #will run during fit()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            print("Vanilla Loss")
            reconstructed, z_mean, z_log_var = self.encoder_decoder(data)

            # Cast the tensors to float32
            data_float32 = tf.cast(data, tf.float32)
            reconstructed_float32 = tf.cast(reconstructed, tf.float32)
            reconstructed_float32 = tf.squeeze(reconstructed_float32, axis=-1)
            
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data_float32, reconstructed_float32)
            )

           
            print("Data Shape: ", data_float32.shape)
            print("Reconstructed Shape: ", reconstructed_float32.shape)
            # Compute MSE
            mse_loss_tracker = tf.reduce_mean(tf.square(data_float32 - reconstructed_float32))
            # kl_loss = self._calculate_kl_loss(z_mean, z_log_var)

        grads = tape.gradient(mse_loss_tracker, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.mse_loss_tracker.update_state(mse_loss_tracker)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)

        return {
            "mse_loss": self.mse_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def _calculate_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        return tf.reduce_mean(kl_loss)
    

    
    #will run during model.predict()
    def predict(self, data, batch_size=32):
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        predictions = []
        abnormality_scores = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch_data = data[start:end]

            reconstruction, z_mean, z_log_var = self.encoder_decoder(batch_data)

            #Abnormality Score
            abnormality_score = (reconstruction-batch_data) ** 2
            abnormality_score = tf.reduce_mean(abnormality_score, axis=(1,2,3))
            predictions.extend(reconstruction)
            abnormality_scores.extend(abnormality_score)

        return predictions, abnormality_scores

######################################################################

class UPAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,3), multiplier: int = 4, latent_size: int = 16, 
                 **kwargs):
        super(UPAE, self).__init__()
        self.encoder_decoder = encoder_decoder(upae=upae, input_shape=input_shape, 
                                               multiplier=multiplier, latent_size=latent_size)
        
        # Convert input_shape to tuple
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)

        self.encoder_decoder.build(input_shape=(None,) + input_shape)  # Build the encoder_decoder model
        self.encoder_decoder.summary()

        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.recontruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name="accuracy")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.loss1_tracker = keras.metrics.Mean(name="loss1")
        self.loss2_tracker = keras.metrics.Mean(name="loss2")

    def call(self, inputs):
        return self.encoder_decoder(inputs)

    @property
    def metrics(self):
        return [
            self.mse_loss_tracker,
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

            reconstructed, z_mean, z_log_var = self.encoder_decoder(data)
            #to be used for training vs validation loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstructed), axis=(1, 2)
                )
            )

            mse_error = (tf.cast(z_mean, tf.float32) - tf.cast(data, tf.float32)) ** 2
            loss1 = K.mean(K.exp(-z_log_var)*mse_error)
            loss2 = K.mean(z_log_var)
            loss = loss1 + loss2

        #calculate gradients update the weights of the model during backpropagation
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.mse_loss_tracker.update_state(mse_error)
        self.recontruction_loss_tracker.update_state(reconstruction_loss)
        self.accuracy_tracker.update_state(data, reconstructed)
        self.total_loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss1)
        self.loss2_tracker.update_state(loss2)

        #outputs every epoch
        return {
            "mse_loss: ": self.mse_loss_tracker.result(),
            "total_loss: ": self.total_loss_tracker.result(),
            "loss1: ": self.loss1_tracker.result(),
            "loss2: ": self.loss2_tracker.result(),
            "binary_crossentropy: ": self.recontruction_loss_tracker.result(),
            "accuracy: ": self.accuracy_tracker.result()
        }

    #will run during model.evaluate()
    def test_step(self, data):
        print("UPAE Validation")
     
        with tf.GradientTape() as tape:
            print("UPAE Loss")

            reconstructed, z_mean, z_log_var = self.encoder_decoder(data)
            #to be used for training vs validation loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstructed), axis=(1, 2)
                )
            )

            rec_err = (tf.cast(z_mean, tf.float32) - tf.cast(data, tf.float32)) ** 2
            loss1 = K.mean(K.exp(-z_log_var)*rec_err)
            loss2 = K.mean(z_log_var)
            loss = loss1 + loss2

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        #updating the metrics trackers 
        self.recontruction_loss_tracker.update_state(reconstruction_loss)
        self.accuracy_tracker.update_state(data, reconstructed)
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

    def predict(self, data, batch_size=32):
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        predictions = []
        abnormality_scores = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch_data = data[start:end]

            reconstruction, z_mean, z_log_var = self.encoder_decoder(batch_data)
            reconstruction_err = (reconstruction-batch_data) ** 2

            #Abnormality Score
            abnormality_score = tf.exp(-z_log_var) * reconstruction_err
            abnormality_score = tf.reduce_mean(abnormality_score, axis=(1,2,3))
            predictions.extend(reconstruction)
            abnormality_scores.extend(abnormality_score)

        return predictions, abnormality_scores
        

class SaveImageCallback(keras.callbacks.Callback):
    def __init__(self, image_data, save_directory, upae=False):
        super().__init__()
        
        self.image_data = image_data  # saving per epoch progress on one image only, you can change this
        self.save_directory = save_directory 

    def on_epoch_end(self, epoch, logs=None):
        print(len(self.image_data))
        Get the reconstructed images for the current epoch
        reconstructed_images = self.model.predict(self.image_data)
        reconstructed_images = reconstructed_images[0].numpy()

        # # Make sure reconstructed_images has the correct shape
        # if len(reconstructed_images.shape) == 3:
        reconstructed_images = np.expand_dims(reconstructed_images, axis=0)
        
        # Save each image separately
        # TODO: Create folder for each image

        # TODO: Have each image be saved in a separate folder

        
        for i, image in enumerate(reconstructed_images):
            generated_rescaled = (image- image.min()) / (image.max() - image.min())
            plt.imshow(generated_rescaled.reshape(64,64,3))
            filename = f"epoch_{epoch}_image_{i}.png"
            save_path = os.path.join('Images/images_epochs', filename)
            plt.savefig(save_path)
            
        #print(f"Saved images for epoch {epoch}.")