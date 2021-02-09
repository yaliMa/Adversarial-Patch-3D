import logging
import numpy as np
from pyrr import Matrix44

logger = logging.getLogger(__name__)


def get_random_z_view_position(far=3.0, close=2.0):
    """
    Return a view position vector with 0.0 for x and y.
    The value of z will be a random (uniform)
    between :param close: to :param far
    :param far: max
    :param close:
    :return:
    """
    return 0.0, 0.0, np.random.uniform(close, far)


def get_random_light(x_min, x_max, y_min, y_max, z_min, z_max):
    """

    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param z_min:
    :param z_max:
    :return:
    """
    return (np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max))


def get_random_rotation(x_min, x_max, y_min, y_max, z_min, z_max):
    return get_rotation(np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                        np.random.uniform(z_min, z_max))


def get_random_translation(x_min, x_max, y_min, y_max, z_min, z_max):
    return get_translation(np.random.uniform(x_min, x_max),
                           np.random.uniform(y_min, y_max),
                           np.random.uniform(z_min, z_max))


def get_rotation(x, y, z):
    logger.debug(
        'Create rotation matrix with (x:{}, y:{}, z:{})'.format(x, y, z))
    x_rotation = Matrix44.from_x_rotation(x)
    y_rotation = Matrix44.from_y_rotation(y)
    z_rotation = Matrix44.from_z_rotation(z)
    return z_rotation * y_rotation * x_rotation


def get_translation(x, y, z):
    logger.debug(
        'Create translation matrix with (x:{}, y:{}, z:{})'.format(x, y, z))
    return Matrix44.from_translation((x, y, z))

