import logging
import numpy as np
from pyrr import Matrix44
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        """

        :param ambient: Ka
        :param diffuse: Kd
        :param specular: Ks
        :param shininess: Ns
        """
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess


class Light(ABC):
    def __init__(self, color):
        self.color = color

    def get_depth_bias_mvp(self, model):
        bias_matrix = Matrix44([[0.5, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.0, 0.0],
                                [0.0, 0.0, 0.5, 0.0],
                                [0.5, 0.5, 0.5, 1.0]])
        light_mvp = self.get_view_proj_matrix() * model
        return bias_matrix * light_mvp

    def get_random_rgb(self, min_r, max_r, min_g, max_g, min_b, max_b):
        return (self.get_legal_color_channel(min_r, max_r, self.color[0]),
                self.get_legal_color_channel(min_g, max_g, self.color[1]),
                self.get_legal_color_channel(min_b, max_b, self.color[2]))

    def get_legal_color_channel(self, in_low, in_up, channel_color):
        change = 1 / 255

        low = channel_color - (change * in_low)
        up = channel_color + (change * in_up)

        low = max(0, low)
        up = min(1, max(low, up))
        low = min(low, up)

        return np.random.uniform(low, up)

    def update_color_to_random_in_offset(self, min_r, max_r, min_g, max_g, min_b, max_b):
        """
        Update the color to a random color. For each color, the random color
        value is unified sampled according to the offset in the input params.

        :param min_r: The lower bound for the red channel. First, it will
        calculate R - (1/255)*min_r. Than the result will be clipped to
        be between 0 to max_r.
        :param max_r: The upper bound for the red channel. First, it will
        calculate R + (1/255)*min_r. Than the result will be clipped to
        be between min_r to 1
        :param min_g: The lower bound for the green channel. First, it will
        calculate G - (1/255)*min_g. Than the result will be clipped to
        be between 0 to max_g.
        :param max_g: The upper bound for the green channel. First, it will
        calculate G + (1/255)*min_g. Than the result will be clipped to
        be between min_g to 1
        :param min_b: The lower bound for the blue channel. First, it will
        calculate B - (1/255)*min_b. Than the result will be clipped to
        be between 0 to max_b.
        :param max_b: The upper bound for the blue channel. First, it will
        calculate B + (1/255)*min_b. Than the result will be clipped to
        be between min_b to 1
        """
        self.color = self.get_random_rgb(min_r, max_r, min_g, max_g, min_b, max_b)
        logger.debug('Update RGB color to: {}'.format(self.color))

    @abstractmethod
    def get_view_proj_matrix(self):
        raise NotImplementedError('Called parent method')


# TODO: Add get shadow proj+view matrix
class IndirectionalLight(Light):
    """
    Point light and spotlight
    """
    def __init__(self, position, color, fov=60):
        super().__init__(color)
        self.position = position
        self.fov = fov

    def get_view_proj_matrix(self): # TODO: params
        proj = Matrix44.perspective_projection(self.fov, 1, 1, 1000)
        view = Matrix44.look_at(self.position, (0, 0, 0), (0, 1, 0))
        return proj * view


# TODO: Add get shadow proj+view matrix
class DirectionalLight(Light):
    def __init__(self, color, inv_direction):
        super().__init__(color)
        self.inv_direction = inv_direction

    def get_view_proj_matrix(self):
        proj = Matrix44.orthogonal_projection(-100, 100, -100, 100, 1, 1000)
        view = Matrix44.look_at(self.inv_direction, (0, 0, 0), (0, 1, 0))
        return proj * view

