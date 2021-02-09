import tensorflow as tf
import os
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
from .dnn import NeuralNetImagenet, checkpoints_dir


class InceptionV3(NeuralNetImagenet):
    def __init__(self):
        super().__init__()
        self.name = 'InceptionV3'

    def get_logits_prob(self, batch_input):
        """
        Prediction from the model on a single batch.
        :param batch_input: the input batch. Must be from size [?, 299, 299, 3]
        :return: the logits and probabilities for the batch
        """
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ = inception.inception_v3(batch_input, num_classes=1001, is_training=False)
            logits = logits[:, 1:]
            probs = tf.squeeze(tf.nn.softmax(logits))
        return logits, probs

    def init_session(self, sess):
        """
        Initialize TF session based on the checkpoint file.
        :param sess: a tf session
        :return:
        """
        saver = \
            tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(sess, os.path.join(checkpoints_dir, 'inception_v3.ckpt'))

    def get_input_shape(self):
        """
        Returns the input width and height
        :return: tuple
        """
        return 299, 299
