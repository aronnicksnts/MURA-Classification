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



class VAE(keras.Model):
    def __init__(self, encoder, decoder, upae=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )


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
            encoder_output  = self.encoder(data)
            reconstruction = self.decoder(encoder_output)

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
            "binary_crossentropy: ": self.recontruction_loss_tracker.result()
        }
    
    def test_step(self, data):
        print("Vanilla Validation")
        encoder_output  = self.encoder(data)
        reconstruction = self.decoder(encoder_output)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )

        #getting mean squared error after making data type equal
        mse_loss = tf.reduce_mean(tf.square(tf.cast(data, tf.float32) - tf.cast(reconstruction, tf.float32)))
        total_loss = mse_loss
       
        #updating the metrics trackers 
        self.total_loss_tracker.update_state(total_loss), 
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "mse_loss": self.total_loss_tracker.result(),
            "binary_crossentropy: ": self.recontruction_loss_tracker.result()
        }
    
        #will run during model.predict()
    def predict(self, data):
        encoder_output = self.encoder(data)
        reconstruction, z_mean , z_logvar = self.decoder(encoder_output)
        return reconstruction

######################################################################

class UPAE(keras.Model):
    def __init__(self, encoder, decoder, upae=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            # train_accuracy_results = []
            encoder_output  = self.encoder(data)
            reconstruction, z_mean, z_log_var = self.decoder(encoder_output)

            #to be used for training vs validation loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            #to be used for learning curve
            #accuracy
            # accuracy = accuracy.update_state(data, reconstruction)
            # train_accuracy_results.append(accuracy)

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
    # def test_step(self, data):
        # print("UPAE Validation")
        # recon_loss_valid = []
        # encoder_output  = self.encoder(data)
        # reconstruction, z_mean, z_log_var = self.decoder(encoder_output)

        # reconstruction_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
        #     )
        # )

        # rec_err = (tf.cast(z_mean, tf.float32) - tf.cast(data, tf.float32)) ** 2
        # loss1 = K.mean(K.exp(-z_log_var)*rec_err)
        # loss2 = K.mean(z_log_var)
        # loss = loss1 + loss2

        # recon_loss_valid.append(reconstruction_loss)

        # #updating the metrics trackers 
        # self.recontruction_loss_tracker.update_state(reconstruction_loss)
        # self.total_loss_tracker.update_state(loss)
        # self.loss1_tracker.update_state(loss1)
        # self.loss2_tracker.update_state(loss2)

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