import logging
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from abc import ABC, abstractmethod

from .attack_render import AttackRender

logger = logging.getLogger(__name__)

# UTILS
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out')
scene_img_dir = os.path.join(out_dir, 'scene_images')
patch_tex_dir = os.path.join(out_dir, 'patch_textures')
perturbation_dir = os.path.join(out_dir, 'perturbations')


class Attack(ABC):
    def __init__(self,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        self.learning_rate = learning_rate
        self.target_class = target_class
        self.true_class = true_class
        self.iteration_num = iteration_num
        # when to log the attack info?
        self.iter_to_log = iter_to_log
        self.iter_to_save_img = iter_to_save_img

        self.ann = self._get_ann()
        self.input_width, self.input_height = self.ann.get_input_shape()

        self.renderer = self._get_attack_renderer()

        self.sess = None  # TODO: Potential bug...
        # Tensors for calculating attack + dynamic batch size
        self.t_patch_adv_tex, self.t_view_images, self.t_perturbation, self.t_batch_size \
            = self.renderer.generate_attack_tensors()

        self.t_logits, self.t_preds = self.ann.get_logits_prob(self.t_view_images)

        self.t_loss = self.define_loss_tv()
        self.t_patch_update = self.define_optimizer()

        self.last_patch_texture = None

    def attack(self):
        """
        Runs an attack to create adversarial patch
        :return:
        """
        self.last_patch_texture = None

        logger.info('Start attack to target #{}: '.format(self.target_class)
                    + str(self.ann.get_label_from_index(self.target_class)))
        logger.info('The true label is #{}: '.format(self.true_class)
                    + str(self.ann.get_label_from_index(self.true_class)))
        logger.info(
            'Running {} iterations. Print every {}.'.format(self.iteration_num,
                                                            self.iter_to_log))

        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.ann.init_session(sess)

            for i in range(self.iteration_num):
                self.attack_iteration(i)

        if self.last_patch_texture is None:
            assert "Last perturbation texture wasn't sent correctly!"

        return self.last_patch_texture  # np array!

    def define_loss_tv(self):
        """
        Define the attack loss.
        :return:
        """
        # tf.constant cannot use a dynamic shape
        t_target_labels = tf.fill([self.t_batch_size], self.target_class)
        t_true_labels = tf.fill([self.t_batch_size], self.true_class)

        self.t_ce_target_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t_target_labels,logits=self.t_logits))
        self.t_ce_true_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t_true_labels,logits=self.t_logits)) * 0.5
        self.t_tv_loss = tf.reduce_mean(tf.image.total_variation(self.t_patch_adv_tex)) * 0.0005

        return self.t_ce_target_loss + self.t_tv_loss - self.t_ce_true_loss

    def define_optimizer(self):
        t_optm_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss, var_list=[self.t_patch_adv_tex])

        with tf.control_dependencies([t_optm_step]):
            t_update = tf.assign(self.t_patch_adv_tex, tf.clip_by_value(self.t_patch_adv_tex, 0, 1))
        return t_update

    def handel_iter_output(self, iter_num, perturbation, loss, preds, dictionary):
        if (iter_num % self.iter_to_log) == 0:
            self._log_attack_iter_info(perturbation, iter_num, loss, preds, is_info=True)
        else:
            self._log_attack_iter_info(perturbation, iter_num, loss, preds, is_info=False)

        if (iter_num % self.iter_to_save_img) == 0 or (iter_num == self.iteration_num - 1):
            self._attack_iter_save_images_tex(iter_num, dictionary)

    def _attack_iter_save_view_images(self, iter_num, dictionary, batch_num=-1):
        """
        Save the iteration's scene's view images in the out folder
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :param dictionary: the iteration dictionary.
        :param batch_num: used for logging.
        """
        init_log_str = 'Saving scene view images in:\n' + scene_img_dir + '\nIteration ' + str(iter_num)
        file_name = f'view_I{iter_num:04}'

        if batch_num >= 0:
            init_log_str += ' Batch ' + str(batch_num)
            file_name += f'B{batch_num:03}'

        logger.debug(init_log_str)

        images = self.sess.run([self.t_view_images], feed_dict=dictionary)

        # create directories if they don't exist
        if not os.path.exists(scene_img_dir):
            os.makedirs(scene_img_dir)

        for indx, image in enumerate(images[0]):
            self.save_image_from_nparray(image,  # PATCH! need to find out why somethimes you need the 0 and sometimes you don't
                                         os.path.join(scene_img_dir, file_name + f'_S{indx:03}.png'))

        logger.debug('Successfully saved views images.')

    def _attack_iter_save_patch_tex(self, iter_num, dictionary, batch_num=-1):
        """
        Save the iteration's patch texture in the out folder
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :param dictionary: the iteration dictionary.
        :param batch_num: used for logging.
        """
        init_log_str = 'Saving patch texture in:\n' + patch_tex_dir + '\nIteration ' + str(iter_num)
        file_name = f'patch_I{iter_num:04}'

        if batch_num >= 0:
            init_log_str += ' Batch ' + str(batch_num)
            file_name += f'B{batch_num:03}'

        texture = self.sess.run([self.t_patch_adv_tex], feed_dict=dictionary)

        # create directories if they don't exist
        # TODO: Do it once on the beginning of the attack?
        if not os.path.exists(patch_tex_dir):
            os.makedirs(patch_tex_dir)

        # Save adv texture
        self.save_image_from_nparray(texture[0],  # PATCH! need to find out why somethimes you need the 0 and sometimes you don't
                                     os.path.join(patch_tex_dir, (file_name + f'.png')),
                                     to_flip=True,
                                     to_save_last=True)

        logger.debug('Successfully saved the patch\'s texture.')

    def save_image_from_nparray(self, array, file_name, to_flip=False, to_save_last=False):
        np_arr = np.rint(array * 255)
        img = Image.fromarray(np_arr.astype(np.uint8))
        if to_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(file_name)
        if to_save_last:
            self.last_patch_texture = img

    def _attack_iter_save_images_tex(self, iter_num, dictionary, batch_num=-1):
        self._attack_iter_save_view_images(iter_num, dictionary, batch_num)
        self._attack_iter_save_patch_tex(iter_num, dictionary, batch_num)

    def _log_attack_iter_info(self, perturbation, iter_num, loss, preds, is_info=False):
        log_str = '[TRAIN {}]: Loss: {}, Perturbation: {}'.format(iter_num, loss, perturbation.sum())  # TODO: abs

        if is_info:
            logger.info(log_str)
        else:
            logger.debug(log_str)

        for i, pred in enumerate(preds):
            preds_labels = self.ann.get_k_top(pred)
            if is_info:
                logger.info('Prediction sample {}: {}'.format(i, preds_labels))
            else:
                logger.debug('Prediction sample {}: {}'.format(i, preds_labels))

    @abstractmethod
    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :return:
        """
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self) -> AttackRender:
        raise NotImplementedError('Called parent method')


class AttackRandomCamera(Attack):
    def __init__(self,
                 batch_size,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):
        super().__init__(
            target_class,
            true_class,
            learning_rate,
            iteration_num,
            iter_to_log,
            iter_to_save_img
        )
        self.renderer.set_up_batch(batch_size)        

    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :return:
        """
        dictionary = self.renderer.get_attack_iteration_dictionary()

        loss, tv_loss, ce_target_loss, ce_true_loss, perturbation, _, preds = \
            self.sess.run([self.t_loss, self.t_tv_loss, self.t_ce_target_loss,
                           self.t_ce_true_loss, self.t_perturbation,
                           self.t_patch_update, self.t_preds],
                          feed_dict=dictionary
                          )

        self.handel_iter_output(iter_num, perturbation, loss, preds, dictionary)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackRandomScene(Attack):
    def __init__(self,
                 batch_size,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):
        super().__init__(target_class,
                         true_class,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img
                         )
        self.renderer.set_up_batch(batch_size)        

    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        Used for logging.
        :return:
        """
        dictionary = self.renderer.get_iteration_dictionary_s_random_t_scene()

        loss, tv_loss, ce_target_loss, ce_true_loss, perturbation, _, preds = \
            self.sess.run([self.t_loss, self.t_tv_loss, self.t_ce_target_loss,
                           self.t_ce_true_loss, self.t_perturbation,
                           self.t_patch_update, self.t_preds],
                          feed_dict=dictionary
                          )

        self.handel_iter_output(iter_num, perturbation, loss, preds, dictionary)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackSystematicTransform(Attack):
    def __init__(self,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        super().__init__(target_class,
                         true_class,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img
                         )

    def attack(self, translation: np.ndarray, rotation: np.ndarray):
        """
        new attack with params
        translation: np array of shape (batch_size,3) -> translation on x,y,z
        rotation: np array of shape (batch_size,3) -> rotation on x,y,z
        """
        translation = np.asarray(translation)
        rotation = np.asarray(rotation)

        assert translation.shape[0] == rotation.shape[0], "translation and rotation should have the same batch size" 
        assert translation.shape[-1] == 3 and rotation.shape[-1] == 3, "the arrays should have 3 components for x, y, z"

        self.renderer.set_up_batch(len(translation))
        self.update_transformations(translation, rotation)

        return super().attack()

    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :return:
        """
        loss, tv_loss, ce_target_loss, ce_true_loss, perturbation, _, preds = \
            self.sess.run([self.t_loss, self.t_tv_loss, self.t_ce_target_loss,
                           self.t_ce_true_loss, self.t_perturbation,
                           self.t_patch_update, self.t_preds],
                          feed_dict=self.dictionary
                          )
        self.handel_iter_output(iter_num, perturbation, loss, preds, self.dictionary)

    def update_transformations(self, translations, rotations):
        self.dictionary = self.renderer.get_iteration_dictionary_s_systematic(translations, rotations)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackSystematicCamera(Attack):
    def __init__(self,
                 target_class,
                 true_class=504,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        super().__init__(target_class,
                         true_class,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img
                         )

    def attack(self, angles):
        """
        new attack
        """
        self.update_angles_degrees(angles)

        return super().attack()

    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        Used for logging.
        :return:
        """
        loss, tv_loss, ce_target_loss, ce_true_loss, perturbation, _, preds = \
            self.sess.run([self.t_loss, self.t_tv_loss, self.t_ce_target_loss,
                           self.t_ce_true_loss, self.t_perturbation,
                           self.t_patch_update, self.t_preds],
                          feed_dict=self.dictionary)
        self.handel_iter_output(iter_num, perturbation, loss, preds, self.dictionary)

    def update_angles_degrees(self, angles):
        """

        :param angles: an array from the same length as the initialization
        angles array. The values inside must be between 0 to 360 (not tested!)
        :return:
        """
        self.renderer.set_up_batch(len(angles))

        self.dictionary = \
            self.renderer.get_iteration_dictionary_s_systematic_r_camera_polar(angles, to_lookat_center=True)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


# TODO new file
class AttackBatches(Attack):
    def __init__(self,
                 target_class,
                 true_class=504,
                 batch_size=64,
                 num_of_batchs=128,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        super().__init__(target_class,
                         true_class,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img
                         )

        self.num_of_batchs = num_of_batchs
        self.dataset_size = num_of_batchs * batch_size

        self.batch_size = batch_size
        self.renderer.set_up_batch(batch_size) # static batch size

        self.dataset_uv, self.dataset_mask, self.dataset_light, self.dataset_rgb = self.build_dataset()

        self.t_uvs = self.renderer.t_patch_uv_map
        self.t_masks = self.renderer.t_patch_mask
        self.t_lights = self.renderer.t_patch_light
        self.t_rgbs = self.renderer.t_background_image

    def attack_iteration(self, iter_num):
        """
        One attack iteration. Render the scene batch_size times and update the patch according to the target class.
        :param iter_num: the number of the current iteration (starting from 0). Used for logging.
        :return:
        """
        first_element = True
        np.random.seed(iter_num)
        permutation = list(np.random.permutation(self.dataset_size))

        for i_batch in range(0, self.num_of_batchs):
            batch_index = permutation[i_batch * self.batch_size: i_batch * self.batch_size + self.batch_size]

            dictionary = {self.t_uvs: self.dataset_uv[batch_index],
                          self.t_masks: self.dataset_mask[batch_index],
                          self.t_lights: self.dataset_light[batch_index],
                          self.t_rgbs: self.dataset_rgb[batch_index]}

            loss, perturbation, _, preds = \
                self.sess.run([self.t_loss, self.t_perturbation, self.t_patch_update, self.t_preds],
                              feed_dict=dictionary)
            if first_element:
                self.handel_iter_output(iter_num, perturbation, loss, preds, dictionary)
                first_element = False
            if iter_num == 0:
                self._attack_iter_save_view_images(iter_num, dictionary, batch_num=i_batch)
                                              
    def handel_iter_output(self, iter_num, perturbation, loss, preds, dictionary):
        if (iter_num % self.iter_to_log) == 0:
            self._log_attack_iter_info(perturbation, iter_num, loss, preds, is_info=True)
        else:
            self._log_attack_iter_info(perturbation, iter_num, loss, preds, is_info=False)

        if (iter_num % self.iter_to_save_img) == 0 or (iter_num == self.iteration_num - 1):
            self._attack_iter_save_patch_tex(iter_num, dictionary)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def build_dataset(self):
        raise NotImplementedError('Called parent method')


class AttackBatchesRandomScene(AttackBatches):
    def __init__(self,
                 target_class,
                 true_class=504,
                 batch_size=64,
                 num_of_batchs=128,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        super().__init__(target_class,
                         true_class,
                         batch_size,
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def build_dataset(self):
        return self.renderer.build_dataset_s_random_t_scene(self.dataset_size)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackBatchesRandomCamera(AttackBatches):
    def __init__(self,
                 target_class,
                 true_class=504,
                 batch_size=64,
                 num_of_batchs=128,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        super().__init__(target_class,
                         true_class,
                         batch_size,
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def build_dataset(self):
        return self.renderer.build_dataset_s_random_t_camera(self.dataset_size)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackBatchesSystematicCamera(AttackBatches):
    def __init__(self,
                 angles,
                 target_class,
                 true_class=504,
                 num_of_batchs=64,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        if len(angles) % num_of_batchs != 0:
            assert "Num of batches must divide the angles list!"

        self.angles = angles

        super().__init__(target_class,
                         true_class,
                         int(angles.shape[0] / num_of_batchs),
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def build_dataset(self):
        return self.renderer.build_dataset_s_systematic_t_camera_polar(self.angles, to_lookat_center=True)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')


class AttackBatchesSystematicScene(AttackBatches):
    def __init__(self,
                 transformations,
                 target_class,
                 true_class=504,
                 num_of_batchs=64,
                 learning_rate=0.25,
                 iteration_num=100,
                 iter_to_log=10,
                 iter_to_save_img=100):

        if len(transformations) % num_of_batchs != 0:
            assert "Num of batches must divide the angles list!"

        self.transformations = transformations

        super().__init__(target_class,
                         true_class,
                         int(transformations.shape[0] / num_of_batchs),
                         num_of_batchs,
                         learning_rate,
                         iteration_num,
                         iter_to_log,
                         iter_to_save_img)

    def build_dataset(self):
        return self.renderer.build_dataset_s_systematic_t_scene(self.transformations)

    @abstractmethod
    def _get_ann(self):
        raise NotImplementedError('Called parent method')

    @abstractmethod
    def _get_attack_renderer(self):
        raise NotImplementedError('Called parent method')