import logging
import tensorflow as tf
import numpy as np
import moderngl
from PIL import Image

from src.logger import set_logger
from src.mug.mug_render import MugEvalRender
from src.dnn import InceptionV3
from data import get_3d_patch_path

# Change if needed
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_logger()  # Call once at main!
logger = logging.getLogger(__name__)  # Call in every file


def main():
    """
    This is the digital evaluation process that we used in our paper.
    """

    # CHANGE HERE!
    # The name of the patch you want to evaluate. The patch should be placed
    # in /data/patches/ (i.e., in this example, the patch you are evaluating
    # is /data/patches/patch_name.png).
    # If you are evaluating our scene, use a patch of size 471x181 px
    path = get_3d_patch_path("patch_name.png")

    img = Image.open(path)
    ctx = moderngl.create_standalone_context()

    dnn = InceptionV3()
    eval_graph = tf.Graph()

    with eval_graph.as_default():
        renderer = MugEvalRender(ctx, dnn, *dnn.get_input_shape())

        # CHANGE HERE!
        # The batch size and the ranges for the camera's position
        # (in polar degrees). You can also change the batch's size.
        probs, angles = renderer.eval_circle(img,
                                             thetas=range(10, 130),
                                             phis=range(4, 11),
                                             R=range(12, 16),
                                             batch_size=128
                                             )

        # CHANGE HERE!
        # True and target labels. The results can be saved to a CSV.
        # We also added more information that we didn't check in the paper
        # to help you with your work. Please check print_classifications
        # for more information.
        renderer.print_classifications(np.asarray(probs),
                                       angles,
                                       target=363,
                                       true=504)

if __name__ == '__main__':
    main()
