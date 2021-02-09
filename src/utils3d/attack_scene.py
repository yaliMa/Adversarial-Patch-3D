import logging
import moderngl
import copy
import numpy as np
from abc import abstractmethod

from .scene3d import Scene3D
from .fbo import AttackFbo
from .program import ProgramTexPhongShadowMultiOut
from .transformation import get_translation, get_rotation

logger = logging.getLogger(__name__)


class AttackScene(Scene3D):
    def __init__(self, ctx):
        self.background_objects = None
        self.patch_object = None

        self.model = self.get_default_model_matrix()

        super().__init__(ctx)

        self.fbo_background = AttackFbo(ctx, self.ctx.viewport[2:])
        self.fbo_patch = AttackFbo(ctx, self.ctx.viewport[2:])

        # TODO: option to select program, also separate light and shadow from scene
        self.program = ProgramTexPhongShadowMultiOut(self.ctx)
        self.generate_scene_vaos(self.program, is_main_vao=True)

    def _get_render_info(self, model, light, camera):
        self._render(model, light, camera)
        return self._get_attack_info_after_render()

    def get_render_info_s_systematic(self, translation, rotation):
        """
        Systematic sampling
        :param translation: (x, y, z)
        :param rotation: (x, y, z)
        :return:
        """
        model = self.create_model_matrix(rotation, translation)
        return self._get_render_info(model, self.light, self.camera)

    def create_model_matrix(self, rotation, translation):
        """

        :param rotation: (x, y, z)
        :param translation: (x, y, z)
        :return:
        """
        # TODO: Check input
        model = get_rotation(*rotation) * get_translation(*translation)
        return model

    def get_render_info_s_random_t_scene(self):
        """
        Random sampling, scene transformations
        :return:
        """
        model = self.get_random_model_matrix()
        light = self.get_random_light_obj()
        return self._get_render_info(model, light, self.camera)

    def get_render_info_s_random_t_camera(self):
        """
        Random sampling, camera transformations
        :return:
        """
        light = self.get_random_light_obj()
        camera = self.get_random_camera_obj()
        return self._get_render_info(self.model, light, camera)

    def get_render_info_s_systematic_r_camera_polar(self, r, theta, phi, to_lookat_center=True):
        """
        Systematic sampling, camera transformations for given polar angles
        :param r:
        :param theta:
        :param phi:
        :param to_lookat_center:
        :return:
        """
        copy_camera = copy.deepcopy(self.camera)
        copy_camera.update_position_polar_degs(r, theta, phi, to_lookat_center)
        return self._get_render_info(self.model, self.light, copy_camera)

    def get_render_info_s_systematic_r_scene_random_light(self, rotation, translation):
        """
        Systematic sampling, scene transformations and random light transformations.
        :param rotation: (x, y, z)
        :param translation: (x, y, z)
        :return:
        """
        model = self.create_model_matrix(rotation, translation)
        light = self.get_random_light_obj()

        return self._get_render_info(model, light, self.camera)

    def get_default_render_info(self):
        return self._get_render_info(self.model, self.light, self.camera)

    def get_patch_texture_image(self):
        return self.patch_object.get_texture_image()

    def _render(self, model, light, camera):
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        light_map = self.generate_scene_shadow_map(model)

        self.program.update_program_scene(camera, model, light, light_map)

        self.fbo_background.fbo.clear()
        with self.fbo_background.scope:
            for obj in self.background_objects:
                self.program.update_program_object(obj.material, obj.get_texture())
                obj.get_vao(is_main_vao=True).render()

        self.fbo_patch.fbo.clear()
        with self.fbo_patch.scope:
            self.program.update_program_object(self.patch_object.material, self.patch_object.get_texture())
            self.patch_object.get_vao(is_main_vao=True).render()

    def load_scene_objects(self):
        self.background_objects = self.load_background_objects()
        self.patch_object = self.load_patch_objects()
        return [self.patch_object] + self.background_objects

    def _get_attack_info_after_render(self):
        target_mask = self._create_patch_mask()
        target_uv = self.fbo_patch.read_uv()
        target_light = self.fbo_patch.read_light()

        static_rgb = self.fbo_background.read_color()[:, :, :3]

        return target_uv, target_mask, target_light, static_rgb

    def _create_patch_mask(self):
        """
        Must be called only after self._render()!!!
        :return:
        """
        depth_background = self.fbo_background.read_depth()
        depth_patch = self.fbo_patch.read_depth()

        mask = np.less(depth_patch, depth_background).astype(np.float32)
        mask = np.flip(mask, 0)

        return np.reshape(mask, (*mask.shape, 1))

    # abstract inherited
    @abstractmethod
    def generate_scene_light(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def generate_scene_camera(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def load_background_objects(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def load_patch_objects(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_default_model_matrix(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_random_model_matrix(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_random_light_obj(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_random_camera_obj(self):
        raise NotImplementedError('Called parent method')
