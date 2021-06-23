from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
from keras.optimizers import SGD

class VGG16Convolutions:

    def __global_average_pooling(self, x):
        return K.mean(x, axis=(2, 3))


    def __global_average_pooling_shape(self, input_shape):
        return input_shape[0:2]


    def __VGG16_convolutions(self):
        model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #chuju
        return model


    def get_model(self):
        return self.__VGG16_convolutions()


    def __load_model_weights(self, model, weights_path):
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            model.layers[k].trainable = False
        f.close()
        return model


    def get_output_layer(self, model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer