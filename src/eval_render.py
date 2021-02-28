import logging
import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod
from pyrr import Matrix44
from .utils3d.scene3d import DisplayScene, Fbo

import matplotlib
matplotlib.use("agg")

logger = logging.getLogger(__name__)


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.info('{:s} function took {:.3f} ms'.format(
            f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class EvalRender(ABC):
    def __init__(self, ctx, ann, width, height):
        """
        Constructor
        :param batch_size: Size of the iteration batch.
        :param width: the width of the image input to the ANN
        :param height: the height of the image input to the ANN
        """

        # ModernGL Context
        self.ctx = ctx
        self.ctx.viewport = (0, 0, width, height)

        # Scene for 3D rendering
        self.scene = self.create_scene()

        self.fbo = Fbo(self.ctx, self.ctx.viewport, components=4)

        self.ann = ann

        self.tf_rgb = tf.placeholder(tf.float32, [None, None, 3], name='rgb')

        dnn_input = tf.expand_dims(self.tf_rgb, 0)
        _, self.probs = self.ann.get_logits_prob(dnn_input)

    def render_single_frame(self, r, theta, phi, to_lookat_center=True):
        self.scene.render_rgb_to_fbo_polar_degrees(
            r, theta, phi, self.fbo, model=Matrix44.from_y_rotation(np.pi), lookat_center=to_lookat_center)
        buffer = self.fbo.fbo.read(components=3, dtype='f4')
        return np.frombuffer(buffer, dtype=np.float32).reshape((self.fbo.viewport[1], self.fbo.viewport[0], 3))[::-1]

    def render_batch_polar_degrees(self, thetas, phis=[0], R=[10]):
        """
        renders frames in the angles specified
        :param thetas: iterable of theta angles(in degrees)
        :param phis: iterable of phi angles(in degrees)
        :param R: iterable of r values
        :returns: list of numpy arrays representing the frames and a tuple of angles (r,θ,φ)
        """
        res = []
        angles = []

        for r in R:
            for theta in thetas:
                for phi in phis:
                    res.append(self.render_single_frame(r, theta, phi))
                    angles.append((r, theta, phi))
        return res, angles

    def render_and_classify(self, thetas, phis=[0], R=[10]):
        """
        more memory efficient version that doesn't aggregate the frames
        """
        probs = []
        angles = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.ann.init_session(sess)

            for r in R:
                for theta in thetas:
                    for phi in phis:
                        frame = self.render_single_frame(r, theta, phi)
                        probs.append(sess.run([self.probs], {self.tf_rgb: frame})[0])
                        angles.append((r, theta, phi))
        return probs, angles

    def classify_batch(self, frames):
        res = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.ann.init_session(sess)

            for frame in frames:
                probabilities = sess.run([self.probs], {self.tf_rgb: frame})
                res.append(probabilities[0])

        return res

    def classify_frames_by_batches(self, frames, batch_size):
        frames_len = len(frames)
        num_of_batches = int(np.floor(frames_len / batch_size))
        res = []

        for i_batch in range(0, num_of_batches):
            logger.info(f"Batch {i_batch}")
            res.extend(self.classify_batch(frames[i_batch * batch_size : i_batch * batch_size + batch_size]))

        if frames_len % batch_size != 0:
            res.extend(self.classify_batch(frames[num_of_batches * batch_size: frames_len]))

        return res

    def print_classifications(self, probs, angles, target, true=504, batch_size=None, csv_file=None):
        probs, angles = np.asarray(probs), np.asarray(angles)
        if probs.shape[0] != angles.shape[0]:
            assert "probs and angles should be from the same length"

        count_true, count_target, count_other = 0, 0, 0
        prob_sum_true, prob_sum_target = 0, 0
        classify_prob_sum_true, classify_prob_sum_target, classify_prob_sum_other = 0, 0, 0

        for i in range(probs.shape[0]):
            if np.argmax(probs[i]) == true:
                count_true += 1
                classify_prob_sum_true += probs[i][true]
            elif np.argmax(probs[i]) == target:
                count_target += 1
                classify_prob_sum_target += probs[i][target]
            else:
                count_other += 1
                classify_prob_sum_other += probs[i][np.argmax(probs[i])]

            prob_sum_true += probs[i][true]
            prob_sum_target += probs[i][target]

            lable, probability = self.ann.get_k_top_with_probs(probs[i], k=1)[0]
            logger.info(f"Position {angles[i]} classified as {lable}, [{probability}]")

        avg_classify_true = 0 if count_true == 0 else classify_prob_sum_true / count_true
        avg_classify_target = 0 if count_target == 0 else classify_prob_sum_target / count_target
        avg_classify_other = 0 if count_other == 0 else classify_prob_sum_other / count_other

        logger.info(f"\nTrue: {count_true}\nTarget: {count_target}\nOther: {count_other}")
        logger.info(
            f"Avg probability for classified label:"
            f"\nAvg Classified True: {avg_classify_true}"
            f"\nAvg Classified Target: {avg_classify_target}"
            f"\nAvg Classified Other: {avg_classify_other}")

        total_true = prob_sum_true / probs.shape[0]
        total_target = prob_sum_target / probs.shape[0]
        logger.info(
            f"Avg probability for total true and target label:"
            f"\nTotal Avg True: {total_true}"
            f"\nTotal Avg Target: {total_target}")

        if csv_file and batch_size:
            import csv
            with open(csv_file, 'a') as file:
                writer = csv.writer(file, delimiter=',')

                if file.tell() == 0:
                    writer.writerow([
                        "Batch Size",
                        "True Count", "Target Count", "Other Count",
                        "Avg Classified True", "Avg Classified Target", "Avg Classified Other",
                        "Total True", "Total Target",
                    ])

                writer.writerow([
                    batch_size,
                    count_true, count_target, count_other,
                    avg_classify_true, avg_classify_target, avg_classify_other,
                    total_true, total_target,
                ])

    @timing
    def eval_circle(self, sticker_img, thetas, phis=[0], R=[10], batch_size=-1):
        self.scene.set_patch_tex(sticker_img)

        frames, angles = self.render_batch_polar_degrees(thetas, phis, R)

        if batch_size > 0:
            probs = self.classify_frames_by_batches(frames, batch_size)
        else:
            probs = self.classify_batch(frames)

        return probs, angles

    @abstractmethod
    def create_scene(self) -> DisplayScene:
        """
        Generate DisplayScene object for initialization.
        :return: DisplayScene object
        """
        raise NotImplementedError('Called parent method')
