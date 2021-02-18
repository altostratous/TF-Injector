import json
import os
import pickle
from collections import defaultdict
from urllib.request import urlopen

from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from matplotlib import pyplot as plt
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.utils import data_utils

from src import tfi
import re
from tensorflow.keras.models import Model


class_index = None


def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            insert_layer_name = '{}_{}'.format(layer.name,
                                               insert_layer_name)
            new_layer = insert_layer_factory(insert_layer_name)
            x = new_layer(x)
            # print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
            #                                                 layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)
fpath = data_utils.get_file(
    'imagenet_class_index.json',
    CLASS_INDEX_PATH,
    cache_subdir='models',
    file_hash='c2c37ea517e94d9795004a39431a14cb')
with open(fpath) as f:
    class_index = json.load(f)

class_names = []
class_titles = []
for i in range(len(class_index)):
    entry = class_index[str(i)]
    class_names.append(entry[0])
    class_titles.append(entry[1])

# intel_class_names = [
#     'n02058221',
#     'n01983481',
#     'n02777292',
#     'n02280649',
#     'n02992529',
#     'n03355925',
#     'n02483362',
#     'n02105056',
#     'n02488291',
#     'n02105162',
#     'n03796401',
#     'n02486261',
#     'n03956157',
#     # 'n09403211',
#     'n01797886',
#     'n04120489',
#     'n04162706',
#     'n04347754'
# ]
#
# inverted_class_index = {e[0]: int(x) for x, e in class_index.items()}
# intel_filter = np.zeros((1000,))
# for intel_class_name in intel_class_names:
#     intel_filter[inverted_class_index[intel_class_name]] = 1
# intel_filter = tf.constant(intel_filter, dtype='float32')

for class_name in class_names:
    class_path = '../../ImageNet-Datasets-Downloader/imagenet/intel/{}'.format(class_name)
    if not os.path.exists(class_path):
        os.mkdir(class_path)

training_dataset = image_dataset_from_directory(
    '../../ImageNet-Datasets-Downloader/imagenet/intel',
    label_mode='categorical',
    class_names=class_names,
    image_size=(224, 224),
    validation_split=0.5,
    subset='training',
    seed=0,
    batch_size=2000
)
# ).filter(lambda x, label: tf.equal(1., tf.reduce_max(tf.multiply(intel_filter, label))))

validation_dataset = image_dataset_from_directory(
    '../../ImageNet-Datasets-Downloader/imagenet/intel',
    label_mode='categorical',
    class_names=class_names,
    image_size=(224, 224),
    validation_split=0.5,
    subset='validation',
    seed=0,
    batch_size=2000
)
# ).filter(lambda x, label: tf.equal(1., tf.reduce_max(tf.multiply(intel_filter, label))))

img_path = '../../ranger-code/Ranger-benchmarks/ResNet18/tiny-imagenet-200/val/images/val_5.JPEG'


class ProfileLayer(Layer):
    profile = defaultdict(list)

    def call(self, inputs, **kwargs):
        if hasattr(inputs, 'numpy'):
            self.profile[self.name].append(inputs.numpy())
        return super().call(inputs, **kwargs)


class RangeRestrictionLayer(Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.bounds = kwargs.pop('bounds')
        super().__init__(trainable, name, dtype, dynamic, **kwargs)


class RangerLayer(RangeRestrictionLayer):

    def call(self, inputs, **kwargs):
        upper = self.bounds[self.name]['upper']
        lower = self.bounds[self.name]['lower']
        return tf.minimum(tf.maximum(super().call(inputs, **kwargs), lower), upper)


class ClipperLayer(RangeRestrictionLayer):

    def call(self, inputs, **kwargs):
        upper = self.bounds[self.name]['upper']
        lower = self.bounds[self.name]['lower']
        result = super().call(inputs, **kwargs)
        mask = tf.logical_or(
            tf.greater(result, upper),
            tf.less(result, lower),
        )
        return tf.where(mask, 0., result)


def profile_layer_factory(insert_layer_name):
    return ProfileLayer(name=insert_layer_name)


if not os.path.exists('ranger.pkl'):
    model = ResNet50(weights='imagenet')
    model = insert_layer_nonseq(model, '.*relu.*', profile_layer_factory, 'dummy')
    loss = CategoricalCrossentropy()
    model.compile(loss=loss, metrics=[TopKCategoricalAccuracy(k=3)])
    model.run_eagerly = True
    model.evaluate(*next(iter(training_dataset)), batch_size=64)
    bounds = {n: {'upper': max(map(np.max, p)), 'lower': min(map(np.min, p))} for n, p in ProfileLayer.profile.items()}
    with open('ranger.pkl', mode='wb') as f:
        pickle.dump(bounds, f)
    exit()

with open('ranger.pkl', 'rb') as f:
    bounds = pickle.load(f)
def ranger_layer_factory(insert_layer_name):
    return RangerLayer(name=insert_layer_name, bounds=bounds)
def clipper_layer_factory(insert_layer_name):
    return ClipperLayer(name=insert_layer_name, bounds=bounds)

# model.evaluate(*next(iter(validation_dataset)), batch_size=64)
# 32/32 [==============================] - 241s 7s/step - loss: 2.3540 - top_k_categorical_accuracy: 0.7613

# tfi.inject(model=model, fiConf={
#     'Artifact': 0,
#     'Type': 'mutate',
#     'Amount': 1,
#     'Bit': '23-30',
# })
# model.evaluate(*next(iter(validation_dataset)), batch_size=64)
# 32/32 [==============================] - 212s 7s/step - loss: 2.3538 - top_k_categorical_accuracy: 0.7613


# tfi.inject(model=model, fiConf={
#     'Artifact': 0,
#     'Type': 'mutate',
#     'Amount': 10,
#     'Bit': '23-30',
# })
# model.evaluate(*next(iter(validation_dataset)), batch_size=64)
# 32/32 [==============================] - 213s 7s/step - loss: 2.3582 - top_k_categorical_accuracy: 0.7670

# tfi.inject(model=model, fiConf={
#     'Artifact': 0,
#     'Type': 'mutate',
#     'Amount': 100,
#     'Bit': '23-30',
# })
# model.evaluate(*next(iter(validation_dataset)), batch_size=64)
# 32/32 [==============================] - 207s 6s/step - loss: 5.7059 - top_k_categorical_accuracy: 0.3726

data = {
    'ranger': {
        1: [],
        10: [],
        100: [],
    },
    'clipper': {
        1: [],
        10: [],
        100: [],
    },
    'none': {
        1: [],
        10: [],
        100: [],
    },
    'nofault': {
        1: [],
        10: [],
        100: [],
    },
}

for epoch in range(60):
    print('epoch', epoch)
    for a in (10, 1, 100):
        fiConf = {
            'Artifact': 'convs',
            'Type': 'mutate',
            'Amount': a,
            'Bit': '23-30',
        }
        faulty_model = ResNet50(weights='imagenet')
        print(fiConf)
        tfi.inject(model=faulty_model, fiConf=fiConf)
        for m in ('clipper', 'nofault', 'ranger', 'none'):
            print(m)
            if m != 'nofault':
                model = ResNet50()
                model.set_weights(faulty_model.get_weights())
            else:
                model = ResNet50(weights='imagenet')
            print('set weights')
            layer_factory = globals().get(m + '_layer_factory')
            if layer_factory:
                model = insert_layer_nonseq(model, '.*relu.*', layer_factory, 'dummy')
            else:
                assert m in ('none', 'nofault')
                model = model
            print('augmented')
            loss = CategoricalCrossentropy()
            model.compile(
                loss=loss,
                metrics=[TopKCategoricalAccuracy(k=3)],
            )

            loss, acc = model.evaluate(*next(iter(training_dataset)), batch_size=64)
            print(acc)
            data[m][a].append(acc)
            with open('conv_v2_{}.pkl'.format(epoch), 'wb') as f:
                pickle.dump(data, f)

print(data)

# def path2input(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x
#
# x = path2input(img_path)
#
#
# preds = model.predict(x)
#
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=1))
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
