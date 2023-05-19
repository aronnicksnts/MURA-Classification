from matplotlib import pyplot

class validation_plots():
    def __init__(self, history_train, file_path):
        self.file_path = file_path
        self.train_binary_cross = history_train.history['binary_crossentropy: ']
        self.valid_binary_cross = history_train.history['val_binary_crossentropy: ']
        self.train_acc = history_train.history['accuracy: ']
        self.valid_acc = history_train.history['val_accuracy: ']

    def valid_loss(self):
        pyplot.plot(self.train_binary_cross , label='train')
        pyplot.plot(self.valid_binary_cross , label='test')
        pyplot.title('Train vs Validation Loss')
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Reconstruction Loss')
        pyplot.legend()
        pyplot.savefig(self.file_path + '/valid_loss.png')
        print("Image saved as valid_loss.png")
        pyplot.show()

    def learning_curve(self):
        pyplot.plot(self.train_acc, label='train')
        pyplot.plot(self.valid_acc, label='test')
        pyplot.title('Learning Curve')
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Accuracy')
        pyplot.legend()
        pyplot.savefig(self.file_path + '/learning_curve.png')
        print("Image saved as learning_curve.png")
        pyplot.show()

# class testing_plots():
# i need labels from each image for this
#     def __init__(self, history_train):

#     def f1_plot(self):

#     def EER_plot(self):

#     def AUC_ROC_plot(self):
