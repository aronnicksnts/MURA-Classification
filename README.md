# Autoencoders using Uncertainty Prediction

## cara-main.ipynb (current progress)
- keras implementation of the model. 
- structure as copied from Mao
-  metrics in testing
    - AUC, Precision, Recal, TP, TP, FP, FN
- parameters:
    - epochs 300
    - learning rate 0.01
- reconstruction error based on MSE will be calculated when can already train on set of images

<img src="https://github.com/aronnicksnts/MURA-Classification.git/blob/cara/Images/inputVreconstructed.png?raw=true" alt="Input vs Reconstruted">


## Autoencoder
- In models.py<br>
    - keras implementation of the VAE model<br>
    - layers based on Mao's AE Implementation:<br>
            - The encoder contains four layers (each with one 4 × 4 convolution with a stride 2)<br>
            - The decoder is connected by two fully connected layers and four<br>
              transposed convolutions<br>
            - encoder = 16-32-64-64   decoder=64-64-32-16<br><br>
- added latent representation (notthe same with UPAE code found online)





    


