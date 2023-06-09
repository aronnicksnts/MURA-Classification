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
import cv2

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


        initializer = GlorotUniform(seed=42)

        # Encoder layers
        self.encoder_layers.append(Conv2D(int(16*multiplier), 4, strides=2, padding='same', use_bias=True))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(32*multiplier), 4, strides=2, padding='same', use_bias=True))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same', use_bias=True))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Conv2D(int(64*multiplier), 4, strides=2, padding='same', use_bias=True))
        self.encoder_layers.append(BatchNormalization())
        self.encoder_layers.append(Activation('relu'))

        self.encoder_layers.append(Reshape((int(64*multiplier) * self.fm * self.fm,)))
        
        # Latent space representation for the encoder
        if upae:
            self.latent_encoder_layers.append(Dense(2048, activation='relu', 
                                                          kernel_initializer=initializer,
                                                          use_bias=True))
            self.latent_encoder_layers.append(BatchNormalization())
            self.latent_encoder_layers.append(Activation('relu'))
            self.latent_encoder_layers.append(Dense(latent_size, 
                                                          kernel_initializer=initializer,
                                                          use_bias=True))
        else:
            self.latent_encoder_layers.append(Dense(2048, activation='relu', 
                                                          kernel_initializer=initializer,
                                                          use_bias=True))
            self.latent_encoder_layers.append(BatchNormalization())
            self.latent_encoder_layers.append(Activation('relu'))
            self.latent_encoder_layers.append(Dense(latent_size*2, 
                                                          kernel_initializer=initializer,
                                                          use_bias=True))
            
        self.latent_decoder_layers.append(Dense(2048, kernel_initializer=initializer,
        use_bias=True))
        self.latent_decoder_layers.append(BatchNormalization())
        self.latent_decoder_layers.append(Activation('relu'))
        self.latent_decoder_layers.append(Dense(int(64 * multiplier) * self.fm * self.fm, 
                                                      kernel_initializer=initializer,
                                                      use_bias=True)) 

        # Latent space representation for the decoder
        self.latent_decoder_layers.append(Reshape((self.fm, self.fm, int(64*multiplier))))
        
        # Decoder layers
        self.decoder_layers.append(Conv2DTranspose(int(64*multiplier), 2, strides=2, padding='same',
                                                   kernel_initializer=initializer))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same'))
        self.decoder_layers.append(BatchNormalization())
        self.decoder_layers.append(Activation('relu'))

        self.decoder_layers.append(Conv2DTranspose(self.out_channels, 4, strides=2, padding='same',
                                                   kernel_initializer=initializer))

        # Building of the Sequential Model
        self.encoder = keras.Sequential(self.encoder_layers, name="encoder")

        self.latent_encoder = keras.Sequential(self.latent_encoder_layers,
                                               name="latent_encoder")
        
        # Z_MEAN and Z_LOG_VAR
        if upae:
            self.z_mean_layer = Dense(latent_size, name="z_mean")
            self.z_log_var_layer = Dense(latent_size, name="z_log_var")
        else:
            self.z_mean_layer = Dense(2*latent_size, name="z_mean")
            self.z_log_var_layer = Dense(2*latent_size, name="z_log_var")
        
        self.latent_decoder = keras.Sequential(self.latent_decoder_layers,
                                               name="latent_decoder")
        
        self.decoder = keras.Sequential(self.decoder_layers, name="decoder")

    def build(self, input_shape):
        super(encoder_decoder, self).build(input_shape)
        # Encoder Build
        self.encoder.build(input_shape)
        encoder_output_shape = self.encoder.compute_output_shape(input_shape)
        # Latent Encoder Build
        self.latent_encoder.build(encoder_output_shape) 
        latent_encoder_output_shape = self.latent_encoder.compute_output_shape(encoder_output_shape)

        # z_mean_layer Build
        self.z_mean_layer.build(latent_encoder_output_shape)
        z_mean_output_shape = self.z_mean_layer.compute_output_shape(latent_encoder_output_shape)

        # z_log_var_layer build
        self.z_log_var_layer.build(latent_encoder_output_shape)
        z_log_var_output_shape = self.z_log_var_layer.compute_output_shape(z_mean_output_shape)

        # Latent Decoder Build
        self.decoder.build((input_shape[0], 16, 16, self.latent_size))


    def call(self, inputs):
        # Encode
        encoder_output = self.encoder(inputs)
        latent_vectors = self.latent_encoder(encoder_output)
        z_mean = self.z_mean_layer(latent_vectors)
        z_log_var = self.z_log_var_layer(latent_vectors)
        latent_vectors = self._sample_latent(z_mean, z_log_var, latent_vectors.shape)
        # Decode
        latent_vectors = self.latent_decoder(latent_vectors)
        reconstructed = self.decoder(latent_vectors)

        # Check if UPAE is true
        if self.upae:
            chunk1, chunk2 = tf.split(reconstructed, 2, axis=3)
            return chunk1, chunk2, z_mean, z_log_var
        else:
            return reconstructed, z_mean, z_log_var
    
    def _sample_latent(self, z_mean, z_log_var, target_shape):
        batch_size = tf.shape(z_mean)[0]
        latent_dim = self.latent_size
        epsilon = tf.random.normal(shape=(batch_size,) + tuple(target_shape[1:]), mean=0.0, stddev=1.0)
        epsilon_expanded = epsilon[:, tf.newaxis, tf.newaxis, :]
        z_mean_expanded = tf.expand_dims(tf.expand_dims(z_mean, axis=1), axis=2)
        z_log_var = z_log_var + 1e-6
        z_log_var_expanded = tf.expand_dims(tf.expand_dims(z_log_var, axis=1), axis=2)
        sampled_latent = z_mean_expanded + (z_log_var_expanded * epsilon_expanded)
        sampled_latent = tf.reshape(sampled_latent, shape=(batch_size,) + tuple(target_shape[1:]))
        return sampled_latent

class VAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,1), multiplier: int = 4, latent_size: int = 16):
        super(VAE, self).__init__()
        self.encoder_decoder = encoder_decoder(upae=upae, input_shape=input_shape, 
                                               multiplier=multiplier, latent_size=latent_size)
        
        # Convert input_shape to tuple
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)

        self.encoder_decoder.build(input_shape=(None,) + input_shape)  # Build the encoder_decoder model
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.encoder_decoder.summary()

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
    
        with tf.GradientTape() as train_tape:
            reconstructed, z_mean, z_log_var = self.encoder_decoder(data)
            data_float32 = tf.cast(data, tf.float32)
            reconstructed_float32 = tf.cast(reconstructed, tf.float32)
            reconstructed_float32 = tf.squeeze(reconstructed_float32, axis=-1)

            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data_float32, reconstructed_float32)
            )
            mse_loss = tf.reduce_mean(tf.square(reconstructed_float32 - data_float32))
            kl_loss = self._calculate_kl_loss(z_mean, z_log_var)
            total_loss = mse_loss + kl_loss

        train_gradient = train_tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(train_gradient, self.trainable_weights))
        
        

        self.mse_loss_tracker.update_state(mse_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "mse_loss": self.mse_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):

        with tf.GradientTape() as train_tape:
            reconstructed, z_mean, z_log_var = self.encoder_decoder(data[0])
            data_float32 = tf.cast(data, tf.float32)
            reconstructed_float32 = tf.cast(reconstructed, tf.float32)
            reconstructed_float32 = tf.squeeze(reconstructed_float32, axis=-1)

            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data_float32, reconstructed_float32)
            )
            mse_loss = tf.reduce_mean(tf.square(reconstructed_float32 - data_float32))
            kl_loss = self._calculate_kl_loss(z_mean, z_log_var)
            total_loss = mse_loss + kl_loss
            
        self.mse_loss_tracker.update_state(mse_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "mse_loss": self.mse_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def _calculate_kl_loss(self, z_mean, z_log_var):
        epsilon = 1e-8
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var + epsilon), axis=-1
        )
        return tf.reduce_mean(kl_loss)
    

    
    #will run during model.predict()
    def predict(self, data, batch_size=32, forCallback=False):
        self.forCallback = forCallback
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        predictions = []
        abnormality_scores = []
        if self.forCallback is False:
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples)
                batch_data = data[start:end]
                
                reconstructed, z_mean, z_log_var = self.encoder_decoder(batch_data)

                reconstructed_float32 = tf.cast(reconstructed, tf.float32)
                reconstructed_float32 = tf.squeeze(reconstructed_float32, axis=-1)

                #Abnormality Score
                abnormality_score = (reconstructed_float32-batch_data) ** 2
                abnormality_score = tf.reduce_mean(abnormality_score, axis=(1,2))
                predictions.extend(reconstructed_float32)
                abnormality_scores.extend(abnormality_score)

            return predictions, abnormality_scores
       
        elif self.forCallback is True:
            #only need reconstructed image thus no abnormality score computation
            reconstruction, z_mean, z_log_var = self.encoder_decoder(data)

            return reconstruction
        

########################################################################################################################


class UPAE(keras.Model):
    def __init__(self, upae=False, input_shape: tuple = (64,64,1), multiplier: int = 4, latent_size: int = 16, 
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
            chunk1, chunk2, z_mean, z_log_var = self.encoder_decoder(data)
            data_float32 = tf.cast(data, tf.float32)
            chunk1_float32 = tf.cast(chunk1, tf.float32)
            chunk1_float32 = tf.squeeze(chunk1_float32, axis=-1)

            chunk2 = tf.cast(chunk2, tf.float32)
            chunk2 = tf.squeeze(chunk2, axis=-1)

            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data_float32, chunk1_float32)
            )
            mse_loss = tf.reduce_mean(tf.square(chunk1_float32 - data_float32 ))
            kl_loss = self._calculate_kl_loss(z_mean, z_log_var)
            loss1 = K.mean(K.exp(-chunk2)*mse_loss)
            loss2 = K.mean(chunk2)
            loss = loss1 + loss2

        #calculate gradients update the weights of the model during backpropagation
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.mse_loss_tracker.update_state(mse_loss)
        self.recontruction_loss_tracker.update_state(reconstruction_loss)
        self.accuracy_tracker.update_state(data, chunk1)
        self.total_loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss1)
        self.loss2_tracker.update_state(loss2)

        #outputs every epoch
        return {
            "mse_loss": self.mse_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "loss1": self.loss1_tracker.result(),
            "loss2": self.loss2_tracker.result(),
            "reconstruction_loss": self.recontruction_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }
    
    def _calculate_kl_loss(self, z_mean, z_log_var):
        epsilon = 1e-8
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var + epsilon), axis=-1
        )
        return tf.reduce_mean(kl_loss)

    #will run during model.evaluate()
    def test_step(self, data):

     
        with tf.GradientTape() as tape:

            chunk1, chunk2, z_mean, z_log_var = self.encoder_decoder(data[0])
            data_float32 = tf.cast(data, tf.float32)
            chunk1_float32 = tf.cast(chunk1, tf.float32)
            chunk1_float32 = tf.squeeze(chunk1_float32, axis=-1)

            chunk2 = tf.cast(chunk2, tf.float32)
            chunk2 = tf.squeeze(chunk2, axis=-1)

            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data_float32, chunk1_float32)
            )
            mse_loss = tf.reduce_mean(tf.square(data_float32 - chunk1_float32))
            kl_loss = self._calculate_kl_loss(z_mean, z_log_var)
            loss1 = K.mean(K.exp(-chunk2)*mse_loss)
            loss2 = K.mean(chunk2)
            loss = loss1 + loss2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        #updating the metrics trackers 
        self.mse_loss_tracker.update_state(mse_loss)
        self.recontruction_loss_tracker.update_state(reconstruction_loss)
        self.accuracy_tracker.update_state(data[1], chunk1)
        self.total_loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss1)
        self.loss2_tracker.update_state(loss2)
        

        #outputs every epoch
        return {
            "mse_loss": self.mse_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "loss1": self.loss1_tracker.result(),
            "loss2": self.loss2_tracker.result(),
            "reconstruction_loss": self.recontruction_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def predict(self, data, batch_size=32 , forCallback=False):
        self.forCallback = forCallback
        num_samples = data.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        predictions = []
        abnormality_scores = []

        if self.forCallback is False:
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples)
                batch_data = data[start:end]

                chunk1, chunk2, z_mean, z_log_var = self.encoder_decoder(batch_data)

                chunk1_float32 = tf.cast(chunk1, tf.float32)
                chunk1_float32 = tf.squeeze(chunk1_float32, axis=-1)
                reconstruction_err = (chunk1_float32-batch_data) ** 2

                chunk2_float32 = tf.cast(chunk2, tf.float32)
                chunk2_float32 = tf.squeeze(chunk2_float32, axis=-1)

                #Abnormality Score
                abnormality_score = tf.exp(-chunk2_float32) * reconstruction_err
                abnormality_score = tf.reduce_mean(abnormality_score, axis=(1,2))
                predictions.extend(chunk1_float32)
                abnormality_scores.extend(abnormality_score)

            return predictions, abnormality_scores
        
        elif self.forCallback is True:
            #only need reconstructed image thus no abnormality score computation

            chunk1, chunk2, z_mean, z_log_var = self.encoder_decoder(data)
            return chunk1, chunk2


class SaveImageCallback(keras.callbacks.Callback):
    def __init__(self, image_data, save_directory, vae):
        super().__init__()
        self.image_data = image_data[:8] # saving per epoch progress on 4 images only, you can change this
        # self.save_directory = save_directory 
        self.save_directory = save_directory
        self.vae = vae
        os.makedirs(self.save_directory, exist_ok=True) #make the folder if non-existent

    def on_epoch_end(self, epoch, logs=None):
        # Check if original images already exists in folder
        if not os.path.exists(f"{self.save_directory}/image0_original.png"):
            for i, image in enumerate(self.image_data):
                cv2.imwrite(f"{self.save_directory}/image{i}_original.png", image)
        # Get the reconstructed images for the current epoch
        if not self.vae:
            reconstructed_images, z_log_vars = self.model.predict(self.image_data, forCallback=True)
            for i, image in enumerate(z_log_vars):
                filename = f"epoch_{epoch}.png"

                subfolder = f"z_log_vars{i}"
                save_directory_perImg = os.path.join(self.save_directory, subfolder)
                os.makedirs(save_directory_perImg, exist_ok=True) #make the folder if non-existent

                save_path = os.path.join(save_directory_perImg, filename)

                # Save the image
                image = image.numpy().astype(np.uint8)
                cv2.imwrite(save_path, image)
        else:
            reconstructed_images = self.model.predict(self.image_data, forCallback=True)
        reconstructed_images = reconstructed_images.numpy()

        
        # Save each image separately
        for i, image in enumerate(reconstructed_images):
            filename = f"epoch_{epoch}.png"

            subfolder = f"image{i}"
            save_directory_perImg = os.path.join(self.save_directory, subfolder)
            os.makedirs(save_directory_perImg, exist_ok=True) #make the folder if non-existent

            save_path = os.path.join(save_directory_perImg, filename)

            # Save the image
            image = image.astype(np.uint8)
            cv2.imwrite(save_path, image)