from .dnn import NeuralNetImagenet, checkpoints_dir
import tensorflow as tf
import os
from tensorflow.contrib import slim
from models import nets


class InceptionV4(NeuralNetImagenet):
    def __init__(self):
        super().__init__()
        self.name = 'InceptionV4'

    def get_logits_prob(self, batch_input):
        """
        Prediction from the model on a single batch.
        :param batch_input: the input batch. Must be from size [?, 299, 299, 3]
        :return: the logits and probabilities for the batch
        """
        with slim.arg_scope(nets.inception_v4_arg_scope()):
            logits, _ = nets.inception_v4(batch_input, num_classes=1001, is_training=False)
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
            tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        saver.restore(sess, os.path.join(checkpoints_dir, 'inception_v4.ckpt'))

    def get_input_shape(self):
        """
        Returns the input width and height
        :return: 299, 299
        """
        return 299, 299
