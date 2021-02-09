
from src.attack import AttackSystematicCamera, AttackBatchesRandomScene, \
    AttackBatchesRandomCamera, AttackBatchesSystematicCamera, \
    AttackSystematicTransform, AttackBatchesSystematicScene
from src.dnn import InceptionV3
from src.mug.mug_render import MugAttackRender


class MugAttackSystematicCamera(AttackSystematicCamera):
    def __init__(self,
                 ctx,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(
            target_class,
            true_class,
            learning_rate,
            iteration_num,
            iter_to_log,
            iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)


class MugAttackSystematicTransform(AttackSystematicTransform):
    def __init__(self,
                 ctx,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(
            target_class,
            true_class,
            learning_rate,
            iteration_num,
            iter_to_log,
            iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)


class MugAttackBatchesRandomScene(AttackBatchesRandomScene):
    def __init__(self,
                 ctx,
                 target_class,
                 true_class=504,
                 batch_size=64,
                 num_of_batchs=128,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(target_class,
                         true_class,
                         batch_size,
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)


class MugAttackBatchesRandomCamera(AttackBatchesRandomCamera):
    def __init__(self,
                 ctx,
                 target_class,
                 true_class=504,
                 batch_size=64,
                 num_of_batchs=128,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(target_class, true_class,
                         batch_size, num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)


class MugAttackBatchesSystematicCamera(AttackBatchesSystematicCamera):
    def __init__(self,
                 ctx,
                 angles,
                 target_class,
                 true_class=504,
                 num_of_batchs=64,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(angles,
                         target_class,
                         true_class,
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)


class MugAttackBatchesSystematicScene(AttackBatchesSystematicScene):
    def __init__(self,
                 ctx,
                 transformations,
                 target_class,
                 true_class=504,
                 num_of_batchs=64,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.ctx = ctx

        super().__init__(transformations,
                         target_class,
                         true_class,
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def _get_ann(self):
        """
        Generate NeuralNetImagenet object for initialization.
        :return: InceptionV3 object
        """
        return InceptionV3()

    def _get_attack_renderer(self):
        """
        Generate AttackRender object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackRender(self.ctx, self.input_width, self.input_height)
