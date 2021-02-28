import logging
import moderngl

from src.logger import set_logger
from src.mug.mug_attack import MugAttackBatchesRandomScene

# Change if needed
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_logger()  # Call once at main!
logger = logging.getLogger(__name__)  # Call in every file


def main():
    ctx = moderngl.create_standalone_context(require=330)

    # CHANGE HERE!
    attack = MugAttackBatchesRandomScene(ctx,
                                         target_class=363,
                                         true_class=504,
                                         batch_size=64,
                                         num_of_batchs=140,
                                         learning_rate=0.75,
                                         iteration_num=400,
                                         iter_to_log=20,
                                         iter_to_save_img=20)

    img = attack.attack()
    # The patches will also be available in /out/patch_textures/


if __name__ == '__main__':
    main()
