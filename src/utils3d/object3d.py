import logging
import numpy as np
from objloader import Obj
from PIL import Image
from abc import ABC, abstractmethod
from data import get_model_path

logger = logging.getLogger(__name__)


class Object3D(ABC):
    """
    The abstract class for a single 3D object (not a scene!)
    Includes shared methods.
    """

    def __init__(self, ctx):
        """
        (Abstract) Creates the programs and build the object's VAOs from an .obj file.
        :param ctx: The scene context. Could be standalone or window
        """
        self.ctx = ctx
        self.name_dir = ''
        self.name_obj = ''
        self.name_tex_file = ''
        self.material = self.generate_material()
        self.obj = None
        self.texture = None
        self.vao_main = None
        self.vao_shadow = None

        self.transpose_texture = False

    @abstractmethod
    def generate_material(self):
        """
        (Abstract) Generate the Material object of the 3D model
        :return: Material object of the 3D model
        """
        raise NotImplementedError('Called parent method')

    def get_obj_file(self):  # TODO: throw exp is file not exist
        """
        Getter for the obj file's data. Reads the file if needed.
        :return: obj file's data
        """
        if self.obj is None:
            path = get_model_path(
                self.name_dir, '{}.obj'.format(self.name_obj))
            logger.debug('Openning: {}'.format(path))
            self.obj = Obj.open(path)

        return self.obj

    def get_texture_image(self):
        if self.texture is None:
            path = get_model_path(self.name_dir, self.name_tex_file)
            logger.debug('Openning: {}'.format(path))

            image = Image.open(path)
            if self.transpose_texture:
                logger.debug('Transposing {} texture'.format(self.name_obj))
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            return image.convert('RGB')
        else:
            return Image.frombytes(self.texture.read())

    def set_texture(self, image: Image, to_flip=True):
        if to_flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        converted = np.asarray(image).astype(np.float32) / 255
        self.texture = self.ctx.texture(converted.shape[:2][::-1], 3, converted.tobytes(), dtype='f4')

    def get_texture(self):
        if self.texture is None:
            image = self.get_texture_image()
            converted = np.asarray(image).astype(np.float32) / 255

            self.texture = self.ctx.texture(converted.shape[:2][::-1], 3, converted.tobytes(), dtype='f4')
        return self.texture

    # TODO: change is_main_vao=True to something more readable
    def get_vao(self, is_main_vao=True):
        if is_main_vao:
            vao = self.vao_main
        else:
            vao = self.vao_shadow
        if vao is None:
            raise TypeError("vao not initialized")

        return vao

    def update_and_get_vao(self, program, is_main_vao=True):
        if is_main_vao:
            self.vao_main = program.load_vao(self.get_obj_file())
            return self.vao_main
        else:
            self.vao_shadow = program.load_vao(self.get_obj_file())
            return self.vao_shadow
