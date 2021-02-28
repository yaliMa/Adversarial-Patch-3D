
import logging
import moderngl
import numpy as np
import tensorflow as tf
import moderngl_window as mglw
from pyrr import Matrix44

from src.logger import set_logger
from src.dnn import InceptionV3
from src.utils3d.fbo import Fbo
from src.mug.mug import TestMugScene3D

set_logger()
logger = logging.getLogger(__name__)


class PatchTester(mglw.WindowConfig):
    """
    A visualization for the digital evaluation process. It is a bit slower,
    but you can use it to identify bugs in the replica.
    Please note that this is not the evaluation process we used in the paper!
    We are publishing it to give you an additional tool.

    We are using the class TestMugScene3D to evaluate an adversarial patch.
    Make sure that the patch's texture is saved as a PNG file in date/mug/
    (i.e., data/mug/patch_adv.png). You can change this by editing the code
    of Patch3D in src/mug/mug. If you add a new scene, use the code in
    src/mug/ as an example.
    """
    gl_version = (3, 3)
    title = "Test Classification"
    window_size = (299, 299)
    aspect_ratio = window_size[0] / window_size[1]
    resizable = False
    samples = 4

    cursor = False
    cam_speed = 1

    # CHANGE HERE!
    # Add the indexes of the true and classes (e.g., 504 for coffee mug, 363
    # for an armadillo, etc.) You can use more than one label as shown here.
    def __init__(self,
                 target=[363],
                 true=[504, 968],
                 batch_size=1,
                 **kwargs):

        super().__init__(**kwargs)

        self.target = target
        self.true = true
        self.batch_size = batch_size

        self.ann = InceptionV3()
        self.input_width, self.input_height = self.ann.get_input_shape()
        self.buf_shape = (batch_size, self.input_height, self.input_width)
        self.tf_rgb = tf.placeholder(tf.float32,
                                     [None, None, 3],
                                     name='rgbs')
        dnn_input = tf.expand_dims(self.tf_rgb, 0)
        self.t_logits, self.t_preds = self.ann.get_logits_prob(dnn_input)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.ann.init_session(self.sess)

        self.true_label = \
            [(self.ann.get_label_from_index(i)) for i in self.true]
        self.target_label = \
            [(self.ann.get_label_from_index(i)) for i in self.target]

        # ModernGL Context
        self.ctx = moderngl.create_context(require=330)
        self.ctx.viewport = (0, 0, self.input_width, self.input_height)
        self.scene = TestMugScene3D(self.ctx)
        self.fbo = Fbo(self.ctx, self.ctx.viewport)

        self.cam_curr_position = self.scene.camera.get_view_position()

        self.model = Matrix44.from_y_rotation(np.pi)

        # CHANGE HERE!!!
        # Here you can define the boundaries for the digital evaluation
        # and step. This is not the evaluation process that was used in the
        # paper. Additionally, the values are just an example.
        self.cam_z_start_position = 10
        self.cam_y_start_position = 0
        self.cam_x_start_position = -4

        self.cam_z_end_position = 30
        self.cam_y_end_position = 5
        self.cam_x_end_position = 7

        self.cam_z_step_size = 1
        self.cam_y_step_size = 1
        self.cam_x_step_size = 1
        # End

        self.cam_z_position = self.cam_z_start_position
        self.cam_y_position = self.cam_y_start_position
        self.cam_x_position = self.cam_x_start_position

        self.total_images = 0
        self.classified_true = 0
        self.classified_target = 0
        self.classified_other = 0

    def render(self, time: float, frame_time: float):
        self.update_camera()

        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.clear(1.0, 1.0, 1.0)

        self.scene.render_rgb(self.model)
        self.classify_frame()

    def update_camera(self):
        if self.cam_z_position > self.cam_z_end_position:
            self.finish_test()

        self.scene.camera.update_position_cartesian(self.cam_x_position,
                                                    self.cam_y_position,
                                                    self.cam_z_position)
        self.cam_curr_position = self.scene.camera.get_view_position()

        if self.cam_x_position >= self.cam_x_end_position:
            self.cam_x_position = self.cam_x_start_position

            if self.cam_y_position >= self.cam_y_end_position:
                self.cam_y_position = self.cam_y_start_position
                self.cam_z_position += self.cam_z_step_size
            else:
                self.cam_y_position += self.cam_y_step_size
        else:
            self.cam_x_position += self.cam_x_step_size

    def classify_frame(self):
        data = self.ctx.fbo.read(components=3, dtype='f4')
        frame = np.frombuffer(data,
                              dtype='f4').reshape((*self.ctx.viewport[2:], 3))

        probabilities = self.sess.run([self.t_preds], {self.tf_rgb: frame})
        probabilities = probabilities[0]

        predictions = self.ann.get_k_top_with_probs(probabilities, k=1)
        self.check_classification(predictions[0][0], predictions[0][1])

    def check_classification(self, label, prob):
        self.total_images += 1

        if label in self.true_label:
            self.classified_true += 1
        elif label in self.target_label:
            self.classified_target += 1
        else:
            self.classified_other += 1

        logger.info("Position {} classified as {}, [{}]".format(
            self.cam_curr_position,
            label,
            prob
        ))

    def finish_test(self):
        logger.info("{} images were tested".format(self.total_images))
        logger.info("{}/{} classified as the -true- label ({})".format(
            self.classified_true,
            self.total_images,
            self.true_label
        ))
        logger.info("{}/{} classified as the -target- ({})".format(
            self.classified_target,
            self.total_images,
            self.target_label
        ))
        logger.info("{}/{} classified as something else".format(
            self.classified_other,
            self.total_images
        ))
        exit(0)  # Kill them all!!!!

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


if __name__ == "__main__":
    PatchTester.run()

