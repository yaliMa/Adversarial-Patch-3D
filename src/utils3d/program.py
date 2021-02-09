import logging
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Program(ABC):
    """
    An abstract class for programs.
    """

    def __init__(self, ctx):
        """
        Builds the program object
        :param ctx: moderngl context
        """
        self.ctx = ctx
        self.prog = self.get_program()

    @abstractmethod
    def load_vao(self, obj):
        """
        Creates a simple vertex array based on an object.
        An abstract method!
        :param obj: obj file. The output of objectloader.open
        :return: a simple vertex array based on the
        input object and this program.
        """
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def get_program(self):
        """
        Create and return this program using self.ctx
        An abstract method!
        :return: moderngl program object
        """
        raise NotImplementedError('Called parent method')


class ProgramTexPhongShadow(Program):
    """
    Program for rendering a 3D model. Including rendering the model's
    texture and a directional light and shadow. Uses the Blinn-Phong light
    model with one light source. The light source is indirection light.
    Notice!!! This program must be used with depth texture! Use ProgShadow
    to generate this texture. In addition, this program will use the depth
    texture as texture(1) and the object color texture as texture(0)
    """

    def __init__(self, ctx):
        """
        Builds the program object
        :param ctx: moderngl context
        """
        Program.__init__(self, ctx)

    def load_vao(self, obj):
        """
        Creates a simple vertex array based on an object.
        :param obj: obj file. The output of objectloader.open. The model must
        have position (vec3), normals (vec3) and uv mapping (vec2)
        :return: a simple vertex array based on the
        input object and this program.
        """
        vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        return self.ctx.simple_vertex_array(self.prog, vbo, 'in_pos', 'in_norm', 'in_uv')

    def get_program(self):
        """
        Create and return this program using self.ctx
        :return: moderngl program object
        """
        return self.ctx.program(
            vertex_shader='''
                        #version 330

                        uniform mat4 ViewModel;
                        uniform mat4 NormalViewModel;
                        uniform mat4 Projection;
                        uniform mat4 DepthBiasMVP;

                        in vec3 in_pos;
                        in vec3 in_norm;
                        in vec2 in_uv;

                        out vec3 v_pos;
                        out vec3 v_norm;
                        out vec2 v_uv;
                        out vec4 v_shadow_uv;

                        void main() {
                            vec4 position = ViewModel * vec4(in_pos, 1.0);
                            gl_Position = Projection * position;

                            v_pos = position.xyz;
                            v_norm = normalize(NormalViewModel * vec4(in_norm, 1.0)).xyz;
                            v_uv = in_uv;
                            v_shadow_uv = DepthBiasMVP * vec4(in_pos, 1.0);
                        }
                    ''',
            fragment_shader='''
                            #version 330

                            uniform vec3 LightPos;
                            uniform vec3 LightColor;

                            uniform vec3 Ka;
                            uniform vec3 Kd;
                            uniform vec3 Ks;
                            uniform float Ns;

                            uniform vec3 ViewPos;

                            uniform sampler2D Texture;
                            uniform sampler2D ShadowMap;

                            in vec3 v_pos;
                            in vec3 v_norm;
                            in vec2 v_uv;
                            in vec4 v_shadow_uv;

                            out vec4 out_color;

                            void main() {    
                                vec3 norm = normalize(v_norm);

                                float bias = 0.005;
                                float visibility = 1.0;
                                if (textureProj(ShadowMap, (v_shadow_uv.xyw)).r < (v_shadow_uv.z-bias)/v_shadow_uv.w) {
                                    visibility = 0.7;
                                }

                                // ambient
                                vec3 ambient = LightColor * Ka;

                                // diffuse 
                                vec3 lightDir = normalize(LightPos - v_pos);
                                float diff = max(dot(norm, lightDir), 0.0);
                                vec3 diffuse = visibility * LightColor * (diff * Kd);

                                // specular
                                vec3 viewDir = normalize(ViewPos - v_pos);
                                vec3 reflectDir = reflect(-lightDir, norm);  
                                float spec = pow(max(dot(viewDir, reflectDir), 0.0), Ns);
                                vec3 specular = visibility * LightColor * (spec * Ks);  

                                vec3 obj_color = texture(Texture, v_uv).rgb;

                                vec3 lum = diffuse + specular + ambient;
                                out_color = vec4(obj_color * lum , 1.0);
                            }

                        ''',
        )

    def update_program(self, camera, model, light, depth_texture, material, texture):
        self.update_program_scene(camera, model, light, depth_texture)
        self.update_program_obj(material, texture)

    def update_program_scene(self, camera, model, light, depth_texture):
        view_model = camera.lookat_matrix() * model
        n_view_model = np.transpose(np.linalg.inv(view_model))

        self.prog["Projection"].write(camera.get_projection_matrix().astype('f4').tobytes())
        self.prog["ViewModel"].write(view_model.astype('f4').tobytes())
        self.prog["NormalViewModel"].write(n_view_model.astype('f4').tobytes())
        self.prog["ViewPos"].value = camera.get_view_position()

        depth_bias_mvp = light.get_depth_bias_mvp(model)
        self.prog["DepthBiasMVP"].write(depth_bias_mvp.astype('f4').tobytes())

        self.prog["LightPos"].value = light.position
        self.prog["LightColor"].value = light.color

        self.prog['Texture'].value = 0
        self.prog['ShadowMap'].value = 1

        depth_texture.use(1)

    def update_program_obj(self, material, texture):
        """

        :param material: the material object of this vao
        :return:
        """
        self.prog["Ka"].value = material.ambient
        self.prog["Kd"].value = material.diffuse
        self.prog["Ks"].value = material.specular
        self.prog["Ns"].value = material.shininess

        texture.use(0)


class ProgramTexPhongShadowMultiOut(Program):
    """
    Program for rendering a 3D model. Including rendering the model's
    texture and a directional light. Uses the Blinn-Phong light model
    with one light source.
    """

    def __init__(self, ctx):
        """
        Builds the program object
        :param ctx: moderngl context
        """
        Program.__init__(self, ctx)

    def load_vao(self, obj):
        """
        Creates a simple vertex array based on an object.
        :param obj: obj file. The output of objectloader.open. The model must
        have position (vec3), normals (vec3) and uv mapping (vec2)
        :return: a simple vertex array based on the
        input object and this program.
        """
        vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        return self.ctx.simple_vertex_array(self.prog, vbo,
                                            'in_pos', 'in_norm', 'in_uv')

    def get_program(self):
        """
        Create and return this program using self.ctx
        :return: moderngl program object
        """
        return self.ctx.program(
            vertex_shader='''
                        #version 330

                        uniform mat4 ViewModel;
                        uniform mat4 NormalViewModel;
                        uniform mat4 Projection;
                        uniform mat4 DepthBiasMVP;

                        in vec3 in_pos;
                        in vec3 in_norm;
                        in vec2 in_uv;

                        out vec3 v_pos;
                        out vec3 v_norm;
                        out vec2 v_uv;
                        out vec4 v_shadow_uv;


                        void main() {
                            vec4 position = ViewModel * vec4(in_pos, 1.0);
                            gl_Position = Projection * position;

                            v_pos = position.xyz;
                            v_norm = normalize(NormalViewModel * vec4(in_norm, 1.0)).xyz;
                            v_uv = in_uv;
                            v_shadow_uv = DepthBiasMVP * vec4(in_pos, 1.0);

                        }
                    ''',
            fragment_shader='''
                            #version 330

                            uniform vec3 LightPos;
                            uniform vec3 LightColor;

                            uniform vec3 Ka;
                            uniform vec3 Kd;
                            uniform vec3 Ks;
                            uniform float Ns;

                            uniform vec3 ViewPos;

                            uniform sampler2D Texture;
                            uniform sampler2D ShadowMap;

                            in vec3 v_pos;
                            in vec3 v_norm;
                            in vec2 v_uv;
                            in vec4 v_shadow_uv;

                            layout(location = 0) out vec4 f_color;
                            layout(location = 1) out vec2 f_uv;
                            layout(location = 2) out vec4 f_light; // some opengl implementations don't support 3 component buffers so one value is not needed

                            void main() {
                                vec3 norm = normalize(v_norm);

                                float bias = 0.005;
                                float visibility = 1.0;
                                if (textureProj(ShadowMap, (v_shadow_uv.xyw)).r < (v_shadow_uv.z-bias)/v_shadow_uv.w){
                                    visibility = 0.5;
                                }

                                // ambient
                                vec3 ambient = LightColor * Ka;

                                // diffuse
                                vec3 lightDir = normalize(LightPos - v_pos);
                                float diff = max(dot(norm, lightDir), 0.0);
                                vec3 diffuse = visibility * LightColor * (diff * Kd);

                                // specular
                                vec3 viewDir = normalize(ViewPos - v_pos);
                                vec3 reflectDir = reflect(-lightDir, norm);
                                float spec = pow(max(dot(viewDir, reflectDir), 0.0), Ns);
                                vec3 specular = visibility * LightColor * (spec * Ks);

                                vec3 obj_color = texture(Texture, v_uv).rgb;
                                vec3 lum = diffuse + specular + ambient;

                                f_color = clamp(vec4(obj_color * lum , 1.0), 0.0, 1.0);
                                f_uv = textureSize(Texture, 0) * v_uv;
                                f_light = vec4(lum, 6.9);
                            }

                        ''',
        )

    def update_program(self, camera, model, light, depth_texture, material, texture):
        self.update_program_scene(camera, model, light, depth_texture)
        self.update_program_object(material, texture)

    def update_program_scene(self, camera, model, light, depth_texture):
        view_model = camera.lookat_matrix() * model
        n_view_model = np.transpose(np.linalg.inv(view_model))

        self.prog["Projection"].write(camera.get_projection_matrix().astype('f4').tobytes())
        self.prog["ViewModel"].write(view_model.astype('f4').tobytes())
        self.prog["NormalViewModel"].write(n_view_model.astype('f4').tobytes())
        self.prog["ViewPos"].value = camera.get_view_position()

        depth_bias_mvp = light.get_depth_bias_mvp(model)
        self.prog["DepthBiasMVP"].write(depth_bias_mvp.astype('f4').tobytes())

        self.prog["LightPos"].value = light.position
        self.prog["LightColor"].value = light.color

        self.prog['Texture'].value = 0
        self.prog['ShadowMap'].value = 1

        depth_texture.use(1)

    def update_program_object(self, material, texture):
        """

        :param material: the material object of this vao
        :return:
        """
        self.prog["Ka"].value = material.ambient
        self.prog["Kd"].value = material.diffuse
        self.prog["Ks"].value = material.specular
        self.prog["Ns"].value = material.shininess

        texture.use(0)


# TODO: consider change to transform (no frag)
class ProgramShadow(Program):
    """
    Program for rendering a depth map from the light perspective. A preprocess
    for rendering shadow.
    """

    def __init__(self, ctx):
        """
        Builds the program object
        :param ctx: moderngl context
        """
        Program.__init__(self, ctx)

    def load_vao(self, obj):
        """
        Creates a simple vertex array based on an object.
        :param obj: obj file. The output of objectloader.open.
        The model must have position (vec3)
        :return: a simple vertex array based on the
        input object and this program.
        """
        vbo = self.ctx.buffer(obj.pack('vx vy vz'))
        return self.ctx.simple_vertex_array(self.prog, vbo, 'in_pos')

    def get_program(self):
        """
        Create and return this program using self.ctx
        :return: moderngl program object
        """
        return self.ctx.program(
            vertex_shader='''
                        #version 330

                        uniform mat4 DepthMVP;

                        in vec3 in_pos;

                        void main() {
                            gl_Position = DepthMVP * vec4(in_pos, 1.0);
                        }
                    ''',
            fragment_shader='''
                        #version 330

                        out float out_depth;

                        void main() {    
                            out_depth = gl_FragCoord.z;
                        }

                    ''',
        )

    def update_program(self, depth_mvp):
        """
        Update this program using the input params.
        :param depth_mvp: the MVP matrix from the light perspective. Use
        orthogonal projection matrix for directional light or perspective
        projection matrix for indirect light (spotlight or point light)
        :return:
        """
        self.prog["DepthMVP"].write(depth_mvp.astype('f4').tobytes())
