# Enhancing Real-World Adversarial Patches with 3D Modeling Techniques

## About the Project
**Please note that this repository is a work in progress! I'm working on uploading all the resources of this paper (including the data and examples). Everything will be available soon.**

This is the code and resources for the paper [Enhancing Real-World Adversarial Patches with 3D Modeling Techniques](https://arxiv.org/abs/2102.05334). The instructions for modeling the scene's replica and building the real-world evaluation setup are [available here](https://www.instructables.com/Evaluation-Setup-for-Real-World-Adversarial-Patche/). 

Our study presents a framework that uses a 3D digital replica of a real-world scene to create a realistic adversarial patch. We demonstrate the framework by creating several adversarial patches for a simple everyday scene: a coffee mug on a desk. The code in this repository can be used for crafting the adversarial patches (both for random and systematic sampling), perform a digital evaluation for the patch, and the code needed for the real-world evaluation. Please refer to the paper for more information.

## Project Overview
The configuration for the logger and the scene’s files are available in `configuration` and `data` respectively. All the information about the neural networks and ImageNet is available in `models`. The size of the checkpoints didn’t allow us to upload them, but they are [available here](TODO). The attack’s output, including images used for the attack and the adversarial patches throwout the attack, will be available in `out`. 

If you want to add a new scene then check `mug`, which contains all the files we used for the scene. Please use those files as an example for creating new scenes and attacks. In addition, `utils3d` contains all the code needed for working with a 3D object. Therefore, if you want to create a new scene you should inherit `Object3D` for each of the scene’s objects and `Scene3D` for the scene itself. Use the classes in `mug.py` as an example. You can also add new transformations (`transformation.py`), or use a different rendering program (`program.py`).

Additionally, check `attack`, which contains the two files needed to implement an attack. To create a new attack you can inherit `Attack` or `AttackBatches` (`attack.py`). We recommend using batches because the attack uses a lot of memory due to the data’s size. If you are attacking a new scene, inherit `AttackRender`, as shown in `mug_renderer.py`. You can also use one of the existing attack methods that were used during this study. If you wish to use systematic sampling in your attack you will have to supply the ranges for each transformation. Moreover, if you need to evaluate your attack then inherit `EvalRender` in `eval_renderer.py`. This evaluation process supports different evaluations, including the use of batches.

**TODO: Examples**

## Evaluation Process
TBD

## Our Configuration
TBD

## A Personal Note
I planned on publishing my code and resources from the start, so most of the code was written to be as clear as possible. However, there are several patches (not only from the adversarial kind) and probably many hidden bugs. In addition, I started working on this research at the beginning of 2019. Upgrading the packages I used wasn’t always an easy task (especially when it comes to TensorFlow) and as a result, the project uses old packages. I can’t promise an updated version, nor to be available for technical support. Still, I hope you will find this repository (as well as [the instructions for creating the replica and evaluation setup](https://www.instructables.com/Evaluation-Setup-for-Real-World-Adversarial-Patche/)) useful. 

### Acknowledgments
A special thanks to Matan Yesharim who helped me to write this code, overcome deadly bugs, and slay the monsters hidden in ModernGL and TensorFlow’s code. 


## Citation
```
@article{mathov2021enhancing,
  title={Enhancing Real-World Adversarial Patches with 3D Modeling Techniques},
  author={Mathov, Yael and Rokach, Lior and Elovici, Yuval},
  journal={arXiv preprint arXiv:2102.05334},
  year={2021}
}
```
