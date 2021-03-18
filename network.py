from keras.applications import VGG16
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.applications import VGG16
import numpy as np

OPTION=["vgg","dense_triplet","cnn_triplet"]

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


class Network():

    def __init__(self, nb_classes, input_shape):
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.base_network = self.use_VGG16()
        self.network_train = self.build_model()
        self.test_function()
        self.base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    def test_function(self):
        input=self.base_network.inputs[0]
        # input=self.base_network.get_layer("vgg16").get_input_at(0)
        output=self.base_network.get_layer("vgg16").get_output_at(-1)
        self.function=K.function([input],[output])

    def predict_imgs(self,imgs,option=None):
        ##using dense layer of tripletloss model
        if option==OPTION[1]:
            return self.base_network.predict(imgs)

        ##using cnn layer of triletloss model
        if option==OPTION[2]:
            pred=self.function([imgs])[0]

        ##using cnn laer of vgg16 model
        if option==OPTION[0]:
            pred=self.base_model.predict(imgs)

        pred = pred.reshape((-1, np.prod(pred.shape[1:])))
        return pred

    def use_VGG16(self):
        model = Sequential()
        vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=self.input_shape)
        model.add(vgg16)
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                        kernel_regularizer=l2(1e-3),
                        kernel_initializer='he_uniform'))

        model.add(
            Dense(self.nb_classes, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform'))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
        print(model.summary())
        return model

    def predict_image(self,img):
        return self.base_network.predict(img)

        # pred= self.base_network.get_layer('vgg16').get_output_at(-1)
        # pred=K.reshape(pred,(-1,np.prod(pred.shape[1:])))
        # return pred

    def build_network(self):
        '''
        Define the neural network to learn image similarity
        Input :
                input_shape : shape of input images
                embeddingsize : vectorsize used to encode our picture
        '''
        # Convolutional Neural Network
        network = Sequential()
        input_shape = self.input_shape

        network.add(Conv2D(128, (7, 7), activation='relu',
                           input_shape=input_shape,
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(2e-4)))
        network.add(MaxPooling2D())
        network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                           kernel_regularizer=l2(2e-4)))
        network.add(MaxPooling2D())
        network.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform',
                           kernel_regularizer=l2(2e-4)))
        network.add(Flatten())
        network.add(Dense(256, activation='relu',
                          kernel_regularizer=l2(1e-3),
                          kernel_initializer='he_uniform'))

        network.add(Dense(self.nb_classes, activation=None,
                          kernel_regularizer=l2(1e-3),
                          kernel_initializer='he_uniform'))

        # Force the encoding to live on the d-dimentional hypershpere
        network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

        return network

    def build_model(self, margin=0.2):

        # Define the tensors for the three input images
        network = self.base_network
        input_shape=self.input_shape
        anchor_input = Input(input_shape, name="anchor_input")
        positive_input = Input(input_shape, name="positive_input")
        negative_input = Input(input_shape, name="negative_input")

        # Generate the encodings (feature vectors) for the three images
        encoded_a = network(anchor_input)
        encoded_p = network(positive_input)
        encoded_n = network(negative_input)

        # TripletLoss Layer
        loss_layer = TripletLossLayer(alpha=margin, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

        # Connect the inputs with the outputs
        network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
        print(network_train.summary())

        # return the model
        return network_train


# network = Network(10,input_shape=(1000,700,3))
# print(network.use_VGG16())
