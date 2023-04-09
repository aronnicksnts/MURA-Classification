from keras import layers, models
from keras.metrics import AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

class Autoencoder(models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        '''layers based on Mao's AE Implementation:
            - The encoder contains four layers (each with one 4 × 4 convolution with a stride 2)
            - The decoder is connected by two fully connected layers and four
              transposed convolutions
            - encoder = 16-32-64-64   decoder=64-64-32-16
        '''
        multiplier = 4
        
        self.encoder = models.Sequential([
            layers.Conv2D(int(16*multiplier), 4, strides=2, padding='same', input_shape=(64,64,3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(int(32*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(int(64*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(int(64*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        
        self.decoder = models.Sequential([
            layers.Conv2DTranspose(int(64*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(3, 4, strides=2, padding='same')
        ])
        
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def compile_AE(self):
        self.compile(optimizer = 'adam', loss='mse',
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
        return self.fit(x_train, y_train, epochs = epochs, batch_size=batch_size)