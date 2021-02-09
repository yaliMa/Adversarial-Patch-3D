import logging
import os
import json
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
checkpoints_dir = os.path.join(models_dir, 'checkpoints')


class NeuralNetImagenet(ABC):
    def __init__(self):
        self.name = 'NN'
        # Reads labels
        self.imagenet_labels = json.load(open(os.path.join(models_dir, 'imagenet_labels.json')))

    def get_k_top(self, probs, k=3):
        """
        Returns the names of the k top classes
        :param probs: a list of probabilities.
        :param k: the number of the top desired classes
        :return: a list with the names of the k top classes (according to
        the probabilities).
        """
        topk = list(probs.argsort()[-k:][::-1])
        return [self.imagenet_labels[i] for i in topk]

    def get_k_top_with_probs(self, probs, k=3):
        """
        Returns the names of the k top classes
        :param probs: a list of probabilities.
        :param k: the number of the top desired classes
        :return: a list of tuples: (label, probabilty).
        """
        topk = list(probs.argsort()[-k:][::-1])
        return [(self.imagenet_labels[i], probs[i]) for i in topk]

    def get_label_from_index(self, index):
        return self.imagenet_labels[index]

    @abstractmethod
    def get_logits_prob(self, batch_input):
        """
        Prediction from the model on a single batch.
        :param batch_input: the input batch.
        :return: the logits and probabilities for the batch
        """
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def init_session(self, sess):
        """
        Initialize TF session based on the checkpoint file.
        :param sess: a tf session
        :return:
        """
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_input_shape(self):
        """
        Returns the input width and height
        :return: width, height (two integers)
        """
        raise NotImplementedError('Called parent method')
