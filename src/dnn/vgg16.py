from .dnn import NeuralNetImagenet, checkpoints_dir
import tensorflow as tf
import os
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import vgg


class Vgg16(NeuralNetImagenet):
    def __init__(self):
        super().__init__()
        self.name = 'Vgg16'

    def get_logits_prob(self, batch_input):
        """
        Prediction from the model on a single batch.
        :param batch_input: the input batch. Must be from size [?, 224, 224, 3]
        :return: the logits and probabilities for the batch
        """

        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_16(batch_input, num_classes=1000, is_training=False)
            probs = tf.squeeze(tf.nn.softmax(logits))[1:]
        return logits, probs

    def init_session(self, sess):
        """
        Initialize TF session based on the checkpoint file.
        :param sess: a tf session
        :return:
        """
        saver = \
            tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
        saver.restore(sess, os.path.join(
            checkpoints_dir, 'vgg_16.ckpt'))

    def get_input_shape(self):
        """
        Returns the input width and height
        :return: tuple
        """
        return 224, 224
