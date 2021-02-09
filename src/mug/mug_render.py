from src.attack.attack_render import AttackRender
from src.eval_render import EvalRender
from .mug_attack_scene import MugAttackScene3D
from .mug import MugScene3D, CleanMugScene3D


class MugEvalRender(EvalRender):
    def __init__(self, ctx, ann, width, height):
        super().__init__(ctx, ann, width, height)

    def create_scene(self):
        """
        Generate DisplayScene object for initialization.
        :return: MugAttackRender object
        """
        return MugScene3D(self.ctx)


class MugCleanEvalRender(EvalRender):
    def __init__(self, ctx, ann, width, height):
        super().__init__(ctx, ann, width, height)

    def create_scene(self):
        """
        Generate DisplayScene object for initialization.
        :return: MugAttackRender object
        """
        return CleanMugScene3D(self.ctx)


class MugAttackRender(AttackRender):
    def __init__(self, ctx, width, height):
        super().__init__(ctx, width, height)

    def create_scene(self):
        """
        Generate AttackScene object for initialization.
        :return: MugAttackRender object
        """
        return MugAttackScene3D(self.ctx)
