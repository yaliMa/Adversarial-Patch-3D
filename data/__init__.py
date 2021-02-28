import os
import errno
import logging


logger = logging.getLogger(__name__)


def get_model_path(model_name, file_name):
    dir_path = _get_3d_model_folder_path(model_name)
    file_path = os.path.join(dir_path, file_name)

    return _return_file_is_exist(file_path)


def _get_3d_model_folder_path(model_name):
    path = os.path.normpath(os.path.abspath(os.path.join(__file__,
                                                         '..',
                                                         'model3d',
                                                         model_name)))
    if not os.path.isdir(path):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                'Directory not found: {}'.format(path))
    return path


def get_3d_stickers_path(sticker_file_name):
    dir_path = os.path.normpath(os.path.abspath(os.path.join(__file__,
                                                             '..',
                                                             'stickers'))
                                )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, sticker_file_name)

    return _return_file_is_exist(file_path)


def _return_file_is_exist(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                'File not found: {}'.format(path))
    return path


