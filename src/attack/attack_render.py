import logging
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from src.utils3d.attack_scene import AttackScene

logger = logging.getLogger(__name__)


class AttackRender(ABC):
    def __init__(self, ctx, width, height):
        """
        Constructor
        :param ctx: ModernGL context that is used in this run
        :param width: the width of the image input to the ANN
        :param height: the height of the image input to the ANN
        """
        self.width, self.height = width, height
        # ModernGL Context
        self.ctx = ctx
        self.ctx.viewport = (0, 0, width, height)

        # Scene for 3D rendering
        self.scene = self.create_scene()

        # TF object for the attack
        self.t_patch_origin_tex = tf.constant(self._load_patch_tex_image(), name='origin_patch_tex_img')
        self.t_patch_uv_map = tf.placeholder(tf.float32, [None, height, width, 2], name='patch_uv_map')
        self.t_patch_mask = tf.placeholder(tf.float32, [None, height, width, 1], name='patch_mask')
        self.t_patch_light = tf.placeholder(tf.float32, [None, height, width, 3], name='patch_light')
        self.t_background_image = tf.placeholder(tf.float32, [None, height, width, 3], name='background_rgb')

    def set_up_batch(self, batch_size):
        """
        sets up the batch size, after the graph is created
        YOU MUST CALL THIS!
        When the batch size is known in the object creation run this at the constructor,
        when the batch size will change during the loop, run this each iteration.
        """
        self.batch_size = batch_size
        self.buffers_shape = (batch_size, self.height, self.width)

        # np arrays for tf dictionary
        self.patch_uv_maps = np.empty((*self.buffers_shape, 2), dtype=np.float32)
        self.patch_masks = np.empty((*self.buffers_shape, 1), dtype=np.float32)
        self.patch_lights = np.empty((*self.buffers_shape, 3), dtype=np.float32)
        self.background_rgbs = np.empty((*self.buffers_shape, 3), dtype=np.float32)

    def generate_attack_tensors(self):
        """
        Generates the tensors for the attack
        :return: Four tensors. The first is a variable for the patch's texture;
        the second is a tensor with the scene's view images; the third is the adversarial perturbation;
        and the fourth is the batch size (dynamic)
        """
        t_patch_adv_tex = tf.get_variable('adv_texture', initializer=self.t_patch_origin_tex)
        t_perturbation = tf.subtract(t_patch_adv_tex, self.t_patch_origin_tex)
        t_view_images, t_batch_size = self._generate_attack_view_images(t_patch_adv_tex)

        return t_patch_adv_tex, t_view_images, t_perturbation, t_batch_size

    def _generate_attack_view_images(self, t_patch_adv_tex):
        """
        Generates the tensor of the batch of the scene's view images (with the transformations)
        :param t_patch_adv_tex: a variable tensor for the adversarial patch
        :return: a tensor with the scene's view images and a tensor of batch size (dynamic)
        """
        t_batch_size = tf.shape(self.t_patch_uv_map)[0]

        t_view = tf.tile(tf.expand_dims(t_patch_adv_tex, 0), [t_batch_size, 1, 1, 1])
        t_view = tf.contrib.resampler.resampler(t_view, self.t_patch_uv_map)
        t_view = t_view * self.t_patch_light
        t_view = self._combine_patch_and_background(t_view)

        return t_view, t_batch_size

    def _combine_patch_and_background(self, t_view):
        """
        Combine the background RGB behind the patch according to the depth mask.
        :param t_view: a batch of view images with the patch image but without the background of the scene.
        :return: a batch of view images that contains all the scene image.
        """
        # TODO: Handel ugly casting
        t_patch_mask_rgb = tf.tile(self.t_patch_mask, [1, 1, 1, 3])
        t_inverted_mask = tf.logical_not(tf.cast(t_patch_mask_rgb, tf.bool))

        return (t_patch_mask_rgb * t_view) + (tf.cast(t_inverted_mask, tf.float32) * self.t_background_image)

    def _load_patch_tex_image(self):
        """

        :return: a numpy array of the patch texture with RGB values between 0 to 1.
        """
        texture = self.scene.get_patch_texture_image()
        return np.asarray(texture).astype(np.float32)[..., : 3] / 255.0

    def _build_dictionary(self):
        return {self.t_patch_uv_map: self.patch_uv_maps,
                self.t_patch_mask: self.patch_masks.astype(np.float32),
                self.t_patch_light: self.patch_lights.astype(np.float32),
                self.t_background_image: self.background_rgbs.astype(np.float32)}

    def get_iteration_dictionary_s_random_t_scene(self):
        """
        Random sampling, scene transformations
        :return:
        """
        for i in range(self.batch_size):
            self.patch_uv_maps[i], self.patch_masks[i], \
                self.patch_lights[i], self.background_rgbs[i]\
                = self.scene.get_render_info_s_random_t_scene()

        return self._build_dictionary()

    def get_iteration_dictionary_s_systematic_r_camera_polar(self, angles, to_lookat_center=True):
        """
        Systematic sampling, camera transformations with given polar degrees
        :return:
        """
        if len(angles) != self.batch_size:
            assert "Angles list must sent doesn't match the batch size sent in initilaztion"
        # TODO: Stop writing C in Python
        i = 0
        for r, theta, phi in angles:
            self.patch_uv_maps[i], self.patch_masks[i], \
                self.patch_lights[i], self.background_rgbs[i] = \
                self.scene.get_render_info_s_systematic_r_camera_polar(r, theta, phi, to_lookat_center)
            i += 1

        return self._build_dictionary()

    def get_iteration_dictionary_s_systematic(self, translations, rotations):
        # TODO: Stop writing C in Python
        i = 0
        for translation, rotation in zip(translations, rotations):
            self.patch_uv_maps[i], self.patch_masks[i], \
                self.patch_lights[i], self.background_rgbs[i] = \
                self.scene.get_render_info_s_systematic(translation, rotation)
            i += 1

        return self._build_dictionary()

    def build_dataset_s_random_t_scene(self, dataset_size):
        size = (dataset_size, self.buffers_shape[1], self.buffers_shape[2])

        patch_uv_maps = np.empty((*size, 2), dtype=np.float32)
        patch_masks = np.empty((*size, 1), dtype=np.float32)
        patch_lights = np.empty((*size, 3), dtype=np.float32)
        background_rgbs = np.empty((*size, 3), dtype=np.float32)

        for i in range(dataset_size):
            patch_uv_maps[i], patch_masks[i], patch_lights[i], background_rgbs[i] = \
                self.scene.get_render_info_s_random_t_scene()

        return patch_uv_maps, patch_masks, patch_lights, background_rgbs

    def build_dataset_s_random_t_camera(self, dataset_size):
        size = (dataset_size, self.buffers_shape[1], self.buffers_shape[2])

        patch_uv_maps = np.empty((*size, 2), dtype=np.float32)
        patch_masks = np.empty((*size, 1), dtype=np.float32)
        patch_lights = np.empty((*size, 3), dtype=np.float32)
        background_rgbs = np.empty((*size, 3), dtype=np.float32)

        for i in range(dataset_size):
            patch_uv_maps[i], patch_masks[i], patch_lights[i], background_rgbs[i] = \
                self.scene.get_render_info_s_random_t_camera()

        return patch_uv_maps, patch_masks, patch_lights, background_rgbs

    def build_dataset_s_systematic_t_camera_polar(self, angles, to_lookat_center=True):
        dataset_size = len(angles)
        size = (dataset_size, self.height, self.width)

        patch_uv_maps = np.empty((*size, 2), dtype=np.float32)
        patch_masks = np.empty((*size, 1), dtype=np.float32)
        patch_lights = np.empty((*size, 3), dtype=np.float32)
        background_rgbs = np.empty((*size, 3), dtype=np.float32)

        for i in range(dataset_size):
            r,  theta, phi = angles[i]

            patch_uv_maps[i], patch_masks[i], patch_lights[i], background_rgbs[i] = \
                self.scene.get_render_info_s_systematic_r_camera_polar(
                r, theta, phi, to_lookat_center
            )

        return patch_uv_maps, patch_masks, patch_lights, background_rgbs

    def build_dataset_s_systematic_t_scene(self, transforms):
        dataset_size = len(transforms)
        size = (dataset_size, self.height, self.width)

        patch_uv_maps = np.empty((*size, 2), dtype=np.float32)
        patch_masks = np.empty((*size, 1), dtype=np.float32)
        patch_lights = np.empty((*size, 3), dtype=np.float32)
        background_rgbs = np.empty((*size, 3), dtype=np.float32)

        for i in range(dataset_size):
            rotation, translation = transforms[i]

            patch_uv_maps[i], patch_masks[i], patch_lights[i], background_rgbs[i] = \
                self.scene.get_render_info_s_systematic_r_scene_random_light(rotation, translation)

        return patch_uv_maps, patch_masks, patch_lights, background_rgbs

    def build_dictionary(self, uv_maps, masks, lights, rgbs):
        if (len(uv_maps) != self.batch_size) \
                or (len(masks) != self.batch_size) \
                or (len(lights) != self.batch_size) \
                or (len(rgbs) != self.batch_size):
            assert "Input size doesn't match batch size"

        return {self.t_patch_uv_map: uv_maps.astype(np.float32),
                self.t_patch_mask: masks.astype(np.float32),
                self.t_patch_light: lights.astype(np.float32),
                self.t_background_image: rgbs.astype(np.float32)}

    @abstractmethod
    def create_scene(self) -> AttackScene:
        """
        Generate AttackScene object for initialization.
        :return: AttackScene object
        """
        raise NotImplementedError('Called parent method')
