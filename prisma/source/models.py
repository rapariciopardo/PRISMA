import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

# This work was partially based on openai/baselines available on https://github.com/openai/baselines/tree/tf2/baselines/deepq


class SplitLayer(layers.Layer):
    """Split input layer
    """
    def __init__(self,
                 num_or_size_splits,
                 **kwargs):
        """Split input layer init.

        Args:
            num_or_size_splits (List or int): if list, shape of the split. if int, number of tensors to get after split.
        """
        super(SplitLayer, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits

    def build(self, input_shape):
        # Call the build function of the parent class to build this layer
        super(SplitLayer, self).build(input_shape)
        # Save the shape, use other functions
        self.shape = input_shape

    def call(self, x):
        # Split x into N tensors, where N is the number of non zero elements in self.num_or_size_splits
        seq = tf.split(x, num_or_size_splits=self.num_or_size_splits, axis=1)

        return seq

def DQN_buffer_model(observation_shape, num_actions, num_nodes, input_size_splits):
    """The DQN buffer model : 
        - The input : tensor with shape (batch_size, num_actions + 1) containing the : destination of the packet and the length of each output buffer in bytes.
        - The output : tensor with shape (batch_size, num_actions) containing the estimated delay for routing the packet to an output buffer.
        - The architecture : 
            1- Split the input to separate the destination from the output buffers.
            2- Encode the destination id using one hot encoding.
            3- Apply a layer Normalisation to the output buffer input. 
            4- Push each block (destination and output buffers) to a dense layer with size 32.
            5- Concat the output of each of the two blocks.
            6- Apply a Dense layer of size 64.
            7- Apply a Dense layer of size 64.
            8- Apply a Dense layer of size num_actions.   

    Args:
        observation_shape (List): shape of the inputs.
        num_actions (list): shape of the outputs.
        num_nodes (int): number of nodes in the network.
        input_size_splits (list): the shape of the split.

    Returns:
        model : keras NN model.
    """
    inp = layers.Input(shape=observation_shape)
    one_hot_layer = layers.Lambda(lambda x: K.one_hot(K.cast(x,'int64'), num_nodes))
    split = SplitLayer(num_or_size_splits=input_size_splits)(inp)
    
    tensors_2_concat = []
    for s in range(len(input_size_splits)):
        if input_size_splits[s] == 0: continue

        if s ==0:
            flattened_split = layers.Flatten()(one_hot_layer(split[s]))
        else:
            flattened_split = layers.LayerNormalization(center=False, scale=False, trainable=False, axis=1)(split[s])

        out_split = layers.Dense(units=32, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(flattened_split)
        tensors_2_concat.append(out_split)
    
    if len(tensors_2_concat) > 1:    
        concatted = layers.Concatenate(axis=1)(tensors_2_concat)
    else:
        concatted = tensors_2_concat[0]
                 
    out = layers.Dense(units=64, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(concatted)
    out = layers.Dense(units=64, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(out)
    out = layers.Dense(num_actions, activation='elu', kernel_initializer='he_uniform', bias_initializer='he_uniform')(out)

    return tf.keras.Model(inputs=inp, outputs=out)


def DQN_routing_model(observation_shape, num_actions, num_nodes, input_size_splits):
    """The DQ routing : 
        - The input : tensor with shape (batch_size, 1) containing the : destination of the packet.
        - The output : tensor with shape (batch_size, num_actions) containing the estimated delay for routing the packet to an output buffer.
        - The architecture : 
            1- Split the input to separate the destination from the output buffers.
            2- Encode the destination id using one hot encoding.
            3- Push destination to a dense layer with size 32.
            4- Apply a Dense layer of size 64.
            5- Apply a Dense layer of size 64.
            6- Apply a Dense layer of size num_actions.   

    Args:
        observation_shape (List): shape of the inputs.
        num_actions (list): shape of the outputs.
        num_nodes (int): number of nodes in the network.
        input_size_splits (list): the shape of the split.

    Returns:
        model : keras NN model.
    """
    inp = layers.Input(shape=observation_shape)
    one_hot_layer = layers.Lambda(lambda x: K.one_hot(K.cast(x,'int64'), num_nodes))
    split = SplitLayer(num_or_size_splits=input_size_splits)(inp)
    
    flattened_split = layers.Flatten()(one_hot_layer(split[0]))
    concatted = layers.Dense(units=32, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(flattened_split)

    out = layers.Dense(units=64, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(concatted)
    out = layers.Dense(units=64, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(out)
    out = layers.Dense(num_actions, activation='elu', kernel_initializer='he_uniform', bias_initializer='he_uniform')(out)

    return tf.keras.Model(inputs=inp, outputs=out)

