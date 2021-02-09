import copy
import numpy as np
from pyrr import Vector3, Matrix44

from src.utils3d import IndirectionalLight
from src.utils3d.attack_scene import AttackScene
from src.utils3d.transformation import get_random_translation
from src.utils3d.transformation import get_random_rotation
from src.utils3d.camera import PerspectiveCamera
from .mug import Mug3D, Wall3D, Ceiling3D, Floor3D, Table3D, Light3D, Patch3D


# TODO: move values to configuration file
class MugAttackScene3D(AttackScene):
    def __init__(self, ctx):
        super().__init__(ctx)

    def generate_scene_light(self):
        return IndirectionalLight((-15, 50, 50),
                                  (139 / 255, 140 / 255, 124 / 255),
                                  120)

    def generate_scene_camera(self):
        return PerspectiveCamera(self.ctx.viewport[2] / self.ctx.viewport[3],
                                 fov_degrees=30.0,
                                 near=1.0,
                                 far=1000.0,
                                 position=Vector3([(0.0, 0.0, 15.0)]),
                                 )

    def load_background_objects(self):
        return [Mug3D(self.ctx),
                Wall3D(self.ctx),
                Ceiling3D(self.ctx),
                Floor3D(self.ctx),
                Table3D(self.ctx),
                Light3D(self.ctx)
                ]

    def load_patch_objects(self):
        return Patch3D(self.ctx)

    def get_default_model_matrix(self):
        return Matrix44.from_y_rotation(np.pi)

    def get_random_model_matrix(self):
        rotation = get_random_rotation(0.0, np.pi / 8,
                                       (np.pi * 3 / 5), (np.pi * 7 / 5),
                                       -0.05, 0.05)

        translation = get_random_translation(-1.5, 1.5,
                                             -1.5, 1.5,
                                             -1., 1.)
        # TODO: scale
        return rotation * translation

    def get_random_light_obj(self):
        copy_light = copy.deepcopy(self.light)
        copy_light.update_color_to_random_in_offset(2, 2, 2, 2, 2, 2)

        return copy_light

    def get_random_camera_obj(self):
        copy_cam = copy.deepcopy(self.camera)
        # copy_cam.update_position_to_random_in_bound_polar_deg(
        #     10, 20,
        #     30, 120,
        #     0, 30,
        #     to_lookat_center=True
        # )

        copy_cam.update_target_to_random_in_bound_by_position_cartesian(
            -1.1, 1.1,
            -0.5, 0.5,
            -.5, .5,
            5
        )

        return copy_cam
