import logging
from abc import ABC, abstractmethod
from .program import ProgramShadow, ProgramTexPhongShadow
from .fbo import *


logger = logging.getLogger(__name__)


class Scene3D(ABC):
    def __init__(self, ctx):
        self.ctx = ctx
        self.light = self.generate_scene_light()
        self.objects = self.load_scene_objects()
        self.prog_shadow = ProgramShadow(self.ctx)

        # Writing to the same texture every time no avoid memory leaks.
        self.shadow_texture = self.ctx.texture(self.ctx.viewport[2:], 1, dtype='f4')

        self.fbo_depth = Fbo(ctx, self.ctx.viewport[2:], components=1)

        self.generate_scene_vaos(self.prog_shadow, is_main_vao=False)

        self.camera = self.generate_scene_camera()

    @abstractmethod
    def generate_scene_light(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def generate_scene_camera(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def load_scene_objects(self):
        raise NotImplementedError('Called parent method')

    def generate_scene_vaos(self, program, is_main_vao=True):
        for obj in self.objects:
            obj.update_and_get_vao(program, is_main_vao)

    def generate_scene_shadow_map(self, model):
        """renders and updates the scenes shadow texture

        Arguments:
            model {Matrix44} -- model matrix

        Returns:
            Context.Texture -- the updated shadow texture
        """
        light_mvp = self.light.get_view_proj_matrix() * model

        self.prog_shadow.update_program(light_mvp)

        self.fbo_depth.fbo.clear()
        with self.fbo_depth.scope:
            for obj in self.objects:
                obj.get_vao(is_main_vao=False).render()

        depf = self.fbo_depth.read_depth()

        self.shadow_texture.write(depf)
        return self.shadow_texture


class DisplayScene(Scene3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        # TODO: option to select program, also separate light and shadow from scene
        self.program_rgb = ProgramTexPhongShadow(self.ctx)
        self.generate_scene_vaos(self.program_rgb, is_main_vao=True)

    def render_rgb(self, model):
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        light_map = self.generate_scene_shadow_map(model)

        self.program_rgb.update_program_scene(self.camera, model, self.light, light_map)

        for obj in self.objects:
            self.program_rgb.update_program_obj(obj.material, obj.get_texture())
            obj.get_vao(is_main_vao=True).render()

    def render_rgb_to_fbo(self, model, fbo):
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        light_map = self.generate_scene_shadow_map(model)

        self.program_rgb.update_program_scene(self.camera, model, self.light, light_map)
        fbo.fbo.clear()
        with fbo.scope:
            for obj in self.objects:
                self.program_rgb.update_program_obj(obj.material, obj.get_texture())
                obj.get_vao(is_main_vao=True).render()

    def render_rgb_to_fbo_polar_degrees(self, r, theta, phi, fbo, model, lookat_center=True):
        self.camera.update_position_polar_degs(r, theta, phi, lookat_center)
        self.render_rgb_to_fbo(model, fbo)

    def render_rgb_polar_degrees(self, r, theta, phi, model, lookat_center=True):
        self.camera.update_position_polar_degs(r, theta, phi, lookat_center)
        self.render_rgb(model)

    @abstractmethod
    def generate_scene_light(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def generate_scene_camera(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def load_scene_objects(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def set_patch_tex(self, img):
        raise NotImplementedError('Called parent method')
