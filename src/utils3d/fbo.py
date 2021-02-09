import numpy as np
import moderngl


class Fbo:
    def __init__(self, ctx, viewport, components=4, scope_flags=moderngl.DEPTH_TEST):
        self.ctx = ctx
        self.viewport = self._set_viewport_dims(viewport)
        self.color_components = components
        self.color_tex = ctx.texture(self.viewport, components=components, dtype='f4')
        self.depth_buffer = ctx.depth_renderbuffer(self.viewport)
        self.fbo = self._init_fbo()

        self.scope = self.ctx.scope(self.fbo, scope_flags)

    def _set_viewport_dims(self, viewport):
        if len(viewport) == 4:
            return viewport[2], viewport[3]
        elif len(viewport) == 2:
            return viewport[0], viewport[1]
        else:
            raise TypeError("viewport must be of length 4 or 2")

    def _init_fbo(self):
        return self.ctx.framebuffer([self.color_tex], self.depth_buffer)

    def _read_color_tex(self, tex, components):
        return np.frombuffer(tex.read(), dtype=np.float32).reshape((self.viewport[1],
                                                                    self.viewport[0],
                                                                    components))[::-1]

    def read_color(self):
        return self._read_color_tex(self.color_tex, self.color_components)

    def read_depth(self):
        return np.frombuffer(self.fbo.read(attachment=-1, dtype='f4'), dtype=np.float32).reshape(self.viewport)


class AttackFbo(Fbo):
    def __init__(self, ctx, viewport):
        self.uv_tex = ctx.texture(viewport, components=2, dtype='f4')
        self.light_tex = ctx.texture(viewport, components=4, dtype='f4')

        super().__init__(ctx, viewport)

    def _init_fbo(self):
        # The color texture must be in the same order as the program!!!!
        return self.ctx.framebuffer([self.color_tex, self.uv_tex, self.light_tex], self.depth_buffer)

    def read_uv(self):
        return self._read_color_tex(self.uv_tex, 2)

    def read_light(self):
        # getting rid of that pesky 4th element
        return self._read_color_tex(self.light_tex, 4)[..., :3]
