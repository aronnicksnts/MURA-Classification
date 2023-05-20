# Autoencoders using Uncertainty Prediction

Train VAE <br>
``python3 main.py``

Train UPAE <br>
``python3 main.py --u``


## current progress

- :heavy_check_mark:  UPAE training loop <br>
- :heavy_check_mark:  VAE training loop <br>
- :heavy_check_mark:  validation with performance metrics (learning curve and validation loss) <br> 
    to interpret curves use [this](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
    <br>
- :heavy_check_mark: testing - show reconstructed image <br>
- :o:	Testing with performance metrics (AUC, F1, EER) <br>	
- :heavy_check_mark: save each reconstruction in training loop to make gif on training progress <br>
- :o: in testing, get image highlighting : (1) variance -mse (2) pixel-wise uncertainty (3) area og abnormality  <br>
- :o: in testing, show difference between reconstruction image output of normal and abnormal <br>
- :heavy_check_mark: save model / best model. - done by aron


### Autoencoder Structure
- encoder: 16-32-64-64 channel sizes <br>
- Latent_enc: seperated na for UPAE and vanilla <br>
- Latent_dec: prepares data from latent_enc for decoder<br>
- decoder: 64-64-32-16<br><br>

1. Training :

    `history_train = model.fit(image_datasets[0], 
                    epochs=epochs, 
                    batch_size=batch_size)` 


2. Validation:

    `history_train = model.fit(image_datasets[1], 
                    epochs=epochs, 
                    batch_size=batch_size)`
<br>
- note: training and validation both using model.fit() since we're checking lang naman the performance on a different dataset. <br>
- model perfromance checked using validation vs training loss (binary cross entropy) and learning curve (accuracy)
<br>
<br>

 3. Testing:

    `generated = model.predict(input_images)`

- this will get the variance of the output (z_logvar). need to get image of this too for results presentation<br>
- compute for abnormality score (recons error normalized with variance)
<br>

<br>


### Vanilla AE loss function
- computed for MSE between input and reconstruction. <br>
- gets diff bw input and output -> get square of score -> get mean <br>

    `mse_loss = tf.reduce_mean(tf.square(tf.cast(data, tf.float32) - tf.cast(reconstruction, tf.float32)))`

<br>

### UPAE loss function
- got mean, and logvariance after decoder. not in encoder. <br>
    - cannot get difference between z and output if z is taken from encoder
    - bottle neck z score has a different size. it has lower dimensionality
    - thus got mean and logvar after decoder. 
<br>
- mean = mean of the distribution of values in the latent space <br>
- logvar = variance of the distribution of values
<br>


    `
    rec_err = (tf.cast(z_mean, tf.float32) - tf.cast(data, tf.float32)) ** 2
            loss1 = K.mean(K.exp(-z_log_var)*rec_err)
            loss2 = K.mean(z_log_var)
            loss = loss1 + loss2
    `
<br>
 
<br>
Why the two losses:

- loss 1 
    "..discourages the auto-encoder from predicting very small uncertainty values for those pixels with higher reconstruction errors" - mao et al. <br>

- loss2  
    "...will prevent the auto-encoder from predicting larger uncertainty for all reconstructed pixels." mao et al. <br>


<br>

### References

[reference for AE](https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/)

[Keras AE reference](https://blog.keras.io/building-autoencoders-in-keras.html)

[Learning Curve interpretation](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)


[basis for getting mean and logvar](https://keras.io/examples/generative/vae/)
