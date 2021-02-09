import logging
import numpy as np
from math import pi, sin, cos, radians
from abc import ABC, abstractmethod
from pyrr import Matrix44, Vector3, vector

logger = logging.getLogger(__name__)


# TODO: Need to make it less complex and less ugly....
# TODO: Move configuration to conf file


def polar_to_cartesian_radians(r, theta, phi):
    """

    :param r: radius
    :param theta: from x-axis on plane xz (around y). 0 to 2pi
    :param phi: from x-axis up. 2 to pi
    :return:
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(phi)
    z = r * np.sin(theta) * np.cos(phi)

    return x, y, z


def polar_to_cartesian_degrees(r, theta, phi):
    return polar_to_cartesian_radians(r, np.radians(theta), np.radians(phi))


class ObjTransform:
    def __init__(self, x_rotate_step=0.05, y_rotate_step=0.05, z_rotate_step=0.05):
        self._x_rotate_theta = 0.0
        self._y_rotate_theta = 0.0
        self._z_rotate_theta = 0.0

        self._x_rotate_step = x_rotate_step
        self._y_rotate_step = y_rotate_step
        self._z_rotate_step = z_rotate_step

    def x_rotate_backward(self):
        self._x_rotate_theta += self._x_rotate_step

    def x_rotate_forward(self):
        self._x_rotate_theta -= self._x_rotate_step

    def y_rotate_left(self):
        self._y_rotate_theta += self._y_rotate_step

    def y_rotate_right(self):
        self._y_rotate_theta -= self._y_rotate_step

    def z_rotate_left(self):
        self._z_rotate_theta -= self._z_rotate_step

    def z_rotate_right(self):
        self._z_rotate_theta += self._z_rotate_step

    def get_rotation_matrix(self):
        self._module_thetas()
        return Matrix44.from_x_rotation(self._x_rotate_theta) \
            * Matrix44.from_y_rotation(self._y_rotate_theta) \
            * Matrix44.from_z_rotation(self._z_rotate_theta)

    def _module_thetas(self):
        self._x_rotate_theta %= 2 * pi
        self._y_rotate_theta %= 2 * pi
        self._z_rotate_theta %= 2 * pi


class Camera(ABC):

    def __init__(self,
                 position=Vector3([(0.0, 0.0, 4.0)]),
                 front=Vector3([(0.0, 0.0, -4.0)]),
                 up=Vector3([(0.0, 1.0, 0.0)]),
                 dolly_step_size=0.5,
                 pedestal_step_size=0.8,
                 truck_step_size=0.8,
                 mouse_sensitivity=0.3,
                 ):

        self._position = position
        self._front = front
        self._up = up

        # keyboard control ### TODO: move into a keyboard camera class
        self._mouse_sensitivity = mouse_sensitivity

        self._first_mouse = True

        self._pitch = 0.0
        self._yaw = 0.0

        self._dolly_step_size = dolly_step_size
        self._pedestal_step_size = pedestal_step_size
        self._truck_step_size = truck_step_size

    def mouse_update(self, x, y):
        if self._first_mouse:
            self._last_x = x
            self._last_y = y
            self._first_mouse = False

        dx, dy = x - self._last_x, self._last_y - y
        self._last_x, self._last_y = x, y

        dx *= self._mouse_sensitivity
        dy *= self._mouse_sensitivity

        self._yaw += dx
        self._pitch += dy

        if self._pitch > 89.0:
            self._pitch = 89.0
        if self._pitch < -89.0:
            self._pitch = -89.0

        self._front.x = cos(radians(self._yaw)) * cos(radians(self._pitch))
        self._front.y = sin(radians(self._pitch))
        self._front.z = sin(radians(self._yaw)) * cos(radians(self._pitch))

        self._front.normalise()

    def dolly_in(self):
        self._position += self._front * self._dolly_step_size

    def dolly_out(self):
        self._position -= self._front * self._dolly_step_size

    def pedestal_up(self):
        self._position += self._up * self._pedestal_step_size

    def pedestal_down(self):
        self._position -= self._up * self._pedestal_step_size

    def truck_left(self):
        self._position += self._right() * self._truck_step_size

    def truck_right(self):
        self._position -= self._right() * self._truck_step_size

    def _right(self):
        return vector.normalize(self._front ^ self._up)

    def lookat_matrix(self):
        return Matrix44.look_at(self._position, self._position + self._front, self._up)

    def get_view_position(self):
        return tuple(self._position)

    def update_target_to_random_in_bound_by_position_cartesian(self, min_x, max_x,
                                                               min_y, max_y,
                                                               min_z, max_z,
                                                               z_scale):
        scale = self._position.z / z_scale

        target_x = np.random.uniform(min_x, max_x) * scale
        target_y = np.random.uniform(min_y, max_y) * scale
        target_z = np.random.uniform(min_z, max_z) * scale

        self._front = Vector3([target_x, target_y, target_z]) - self._position
        logger.debug('Update front to: {}'.format(self._front))

    # TODO: Test boundaries!
    def update_position_to_random_in_bound_cartesian(self,
                                                     min_x, max_x,
                                                     min_y, max_y,
                                                     min_z, max_z,
                                                     to_lookat_center=True
                                                     ):
        self.update_position_cartesian(np.random.uniform(min_x, max_x),
                                       np.random.uniform(min_y, max_y),
                                       np.random.uniform(min_z, max_z),
                                       to_lookat_center)

    def update_position_to_random_in_bound_polar_deg(self,
                                                     min_r, max_r,
                                                     min_theta, max_theta,
                                                     min_phi, max_phi,
                                                     to_lookat_center=True
                                                     ):
        self.update_position_polar_degs(np.random.uniform(min_r, max_r),
                                        np.random.uniform(min_theta, max_theta),
                                        np.random.uniform(min_phi, max_phi),
                                        to_lookat_center)

    # TODO: encapsulation & input validation
    def update_position_cartesian(self, x, y, z, to_lookat_center=True):
        self._position = Vector3([x, y, z])

        if to_lookat_center:
            self._front = self._position.inverse

        logger.debug('Update position to: {}'.format(self._position))

    def update_position_polar_rads(self, r, theta, phi, to_lookat_center=True):
        x, y, z = polar_to_cartesian_radians(r, theta, phi)
        self.update_position_cartesian(x, y, z, to_lookat_center)

    def update_position_polar_degs(self, r, theta, phi, to_lookat_center=True):
        x, y, z = polar_to_cartesian_degrees(r, theta, phi)
        self.update_position_cartesian(x, y, z, to_lookat_center)


    @abstractmethod
    def get_projection_matrix(self):
        pass


class PerspectiveCamera(Camera):
    def __init__(self,
                 aspect_ratio,
                 fov_degrees=45,
                 near=1.0,
                 far=1000.0,
                 position=Vector3([(0.0, 0.0, 70.0)]),
                 front=Vector3([(0.0, 0.0, -4.0)]),
                 up=Vector3([(0.0, 1.0, 0.0)]),
                 dolly_step_size=0.5,
                 pedestal_step_size=0.8,
                 truck_step_size=0.8,
                 zoom_step_size=0.8
                 ):
        super().__init__(position,
                         front,
                         up,
                         dolly_step_size,
                         pedestal_step_size,
                         truck_step_size,
                         )

        # TODO check input
        self._fov_degrees = fov_degrees
        self._aspect_ratio = aspect_ratio
        self._near = near
        self._far = far

        self._zoom_step_size = zoom_step_size

    def zoom_in(self):
        self._fov_degrees -= self._zoom_step_size

    def zoom_out(self):
        self._fov_degrees += self._zoom_step_size

    def get_projection_matrix(self):
        return Matrix44.perspective_projection(self._fov_degrees, self._aspect_ratio, self._near, self._far)
