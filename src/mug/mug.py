from pyrr import Vector3
from src.utils3d import Object3D, DisplayScene, Material, IndirectionalLight
from src.utils3d.camera import PerspectiveCamera


class MugScene3D(DisplayScene):
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
                                 position=Vector3([(0.0, 0.0, 15.0)])
                                 )

    def load_scene_objects(self):
        return [Mug3D(self.ctx),
                Patch3D(self.ctx),
                Wall3D(self.ctx),
                Ceiling3D(self.ctx),
                Floor3D(self.ctx),
                Table3D(self.ctx),
                Light3D(self.ctx)
                ]

    def set_patch_tex(self, img):
        self.objects[1].set_texture(img)


class CleanMugScene3D(DisplayScene):
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
                                 position=Vector3([(0.0, 0.0, 15.0)])
                                 )

    def load_scene_objects(self):
        return [Mug3D(self.ctx),
                Wall3D(self.ctx),
                Ceiling3D(self.ctx),
                Floor3D(self.ctx),
                Table3D(self.ctx),
                Light3D(self.ctx)
                ]

    def set_patch_tex(self, img):
        assert "Clean scene!"


class TestMugScene3D(DisplayScene):
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

    def load_scene_objects(self):
        return [Mug3D(self.ctx),
                Patch3D(self.ctx, True),
                Wall3D(self.ctx),
                Ceiling3D(self.ctx),
                Floor3D(self.ctx),
                Table3D(self.ctx),
                Light3D(self.ctx)
                ]

    def set_patch_tex(self, img):
        self.objects[1].set_texture(img)

class Mug3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Mug'

        self.name_dir = 'mug'
        self.name_obj = 'mug'
        self.name_tex_file = 'mug.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((0.8,) * 3,
                        (0.970328, 0.94118, 1.0),
                        (1.0,) * 3,
                        828)


class Patch3D(Object3D):
    def __init__(self, ctx, use_adv_sticker=False):
        super().__init__(ctx)

        self.name = 'Patch'

        self.name_dir = 'mug'
        self.name_obj = 'patch'
        if use_adv_sticker:
            self.name_tex_file = 'patch_adv.png'
        else:
            self.name_tex_file = 'patch.png'

        self.transpose_texture = True

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((0.7,) * 3,
                        (0.588,) * 3,
                        (0.9,) * 3,
                        80)


class Table3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Table'

        self.name_dir = 'mug'
        self.name_obj = 'table'
        self.name_tex_file = 'table.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((0.4,) * 3,
                        (0.8667,) * 3,
                        (0.0,) * 3,
                        10)


class Wall3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Wall'

        self.name_dir = 'mug'
        self.name_obj = 'wall'
        self.name_tex_file = 'wall.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((0.8,) * 3,
                        (1.0,) * 3,
                        (0.0,) * 3,
                        10)


class Ceiling3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Ceiling'

        self.name_dir = 'mug'
        self.name_obj = 'ceiling'
        self.name_tex_file = 'ceiling.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((1.0,) * 3,
                        (1.0,) * 3,
                        (0.0,) * 3,
                        10)


class Floor3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Floor'

        self.name_dir = 'mug'
        self.name_obj = 'floor'
        self.name_tex_file = 'floor.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((1.0,) * 3,
                        (1.0,) * 3,
                        (0.0,) * 3,
                        10)


class Light3D(Object3D):
    def __init__(self, ctx):
        super().__init__(ctx)

        self.name = 'Light'

        self.name_dir = 'mug'
        self.name_obj = 'light'
        self.name_tex_file = 'light.png'

    def generate_material(self):
        """
        Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        return Material((1.0,) * 3,
                        (1.0,) * 3,
                        (0.0,) * 3,
                        10)
