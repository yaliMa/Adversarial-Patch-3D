import logging
import moderngl
import itertools
import numpy as np

from src.logger import set_logger
from src.mug.mug_attack import MugAttackBatchesSystematicScene

# Change if needed
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_logger()  # Call once at main!
logger = logging.getLogger(__name__)  # Call in every file


def main():
    ctx = moderngl.create_standalone_context(require=330)

    # Rotation - Change
    rx = np.linspace(0.0, np.pi / 8, 5)
    ry = np.linspace((np.pi * 3 / 5), (np.pi * 7 / 5), 8)
    rz = np.linspace(-0.05, 0.05, 3)

    # Translation - Change
    tx = np.linspace(-1.5, 1.5, 5)
    ty = np.linspace(-1.5, 1.5, 5)
    tz = np.linspace(-1., 1., 3)

    transformations = np.array([((transform[0], transform[1], transform[2]),
                                 (transform[3], transform[4], transform[5]))
                                for transform in itertools.product(rx, ry, rz,
                                                                   tx, ty, tz)])
    # CHANGE HERE!
    attack = MugAttackBatchesSystematicScene(ctx,
                                             transformations,
                                             target_class=363,
                                             true_class=504,
                                             num_of_batchs=125,
                                             learning_rate=0.75,
                                             iteration_num=400,
                                             iter_to_log=20,
                                             iter_to_save_img=20)
    img = attack.attack()
    # The patches will also be available in /out/patch_textures/


if __name__ == '__main__':
    main()
