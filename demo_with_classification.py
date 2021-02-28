import numpy as np
import tensorflow as tf
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44, Quaternion

from src.mug.mug import MugScene3D
from src.dnn import InceptionV3
from src.utils3d.fbo import Fbo


class LiveScene(mglw.WindowConfig):
    """
    A live demo for the scene. Can be used for demos, as well as for
    identifying problems in the replica.
    Make sure that you are pressing the keys after checking the demo's window.
    If you do it in the IDE window then nothing will happen and you will get
    mad because my code if a piece of sheep :)

    The list of key bindings (in addition to your mouse movement):
       - W:     camera dolly in
       - S:     camera dolly out
       - A:     camera truck right
       - D:     camera truck left
       - SPACE: camera pedestal up
       - E:     camera pedestal down
       - C:     classify frame
    """
    gl_version = (3, 3)
    title = "Live Scene Example"
    window_size = (500, 500)
    aspect_ratio = window_size[0] / window_size[1]
    resizable = False
    samples = 4

    cursor = False

    cam_speed = 0.05

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)

    def init_dnn(self):
        """
        initializes the tf session and the computation graph
        """
        self.dnn = InceptionV3()
        self.input_shape = self.dnn.get_input_shape()
        self.sess = tf.Session()

        self.inp = tf.placeholder(
            dtype=np.float32, shape=(*self.input_shape, 3))

        processed_images = tf.expand_dims(self.inp, 0)
        _, self.probs = self.dnn.get_logits_prob(processed_images)

        self.sess.run(tf.global_variables_initializer())
        self.dnn.init_session(self.sess)

    def classify_frame(self):
        def crop_center(img, cropx, cropy):
            y, x, _ = img.shape
            startx = x//2 - cropx//2
            starty = y//2 - cropy//2
            return img[starty:starty+cropy, startx:startx+cropx, :]

        print("classifying the current frame:")

        data = self.ctx.fbo.read(components=3, dtype='f4')
        frame = np.frombuffer(data, dtype='f4').reshape(
            (*self.ctx.viewport[2:], 3))[::-1]
        cropped_frame = crop_center(frame, *self.input_shape)

        assert(cropped_frame.shape == (*self.input_shape, 3))  # for inception

        probabilities = self.sess.run([self.probs], {self.inp: cropped_frame})
        probabilities = probabilities[0]

        predictions = self.dnn.get_k_top_with_probs(probabilities, k=1)
        print(*predictions)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scene = MugScene3D(self.ctx)

        self.init_dnn()
        self.fbo = Fbo(self.ctx, self.ctx.viewport)

        self.model = Matrix44.from_y_rotation(np.pi)

        self.actions = [
            # list of keys and the function to call while they are pressed
            (self.wnd.keys.W, self.scene.camera.dolly_in),
            (self.wnd.keys.S, self.scene.camera.dolly_out),
            (self.wnd.keys.A, self.scene.camera.truck_right),
            (self.wnd.keys.D, self.scene.camera.truck_left),
            (self.wnd.keys.SPACE, self.scene.camera.pedestal_up),
            (self.wnd.keys.E, self.scene.camera.pedestal_down),
            (self.wnd.keys.C, self.classify_frame)
        ]

    def check_for_movement(self):
        for key, action in self.actions:
            if self.wnd.is_key_pressed(key):
                action()

    def mouse_position_event(self, x, y):
        self.scene.camera.mouse_update(x, y)

    def mouse_scroll_event(self, x_offset, y_offset):
        if y_offset > 0:
            self.scene.camera.zoom_in()
        else:
            self.scene.camera.zoom_out()

    def render(self, time: float, frame_time: float):
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.clear(1.0, 1.0, 1.0)

        self.scene.render_rgb(self.model)

        self.check_for_movement()


if __name__ == "__main__":
    LiveScene.run()
