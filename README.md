<h1 align="center">
  Enhancing Real-World Adversarial Patches<br>with 3D Modeling Techniques
</h1>

<h3 align="center">
  <a href="https://arxiv.org/abs/2102.05334">
    Paper
  </a>
  <span> | </span>
  <a href="https://github.com/yaliMa/Adversarial-Patch-3D">
    Code
  </a>
  <span> | </span>
  <a href="https://www.instructables.com/Evaluation-Setup-for-Real-World-Adversarial-Patche/">
    Setup
  </a>
</h3>

<p align="center">
  A framework for crafting real-world adversarial patches using 3D modeling.
</p>

<p align="center">
  <img width=80% src="https://github.com/yaliMa/Adversarial-Patch-3D/blob/master/banner.png">
</p>


## About the Project
**Please note that this repository is a work in progress! I'm working on uploading all the resources of this paper (including the data and examples). Everything will be available soon.**

This is the code and resources for the paper [Enhancing Real-World Adversarial Patches with 3D Modeling Techniques](https://arxiv.org/abs/2102.05334). The instructions for modeling the scene's replica and building the real-world evaluation setup are [available here](https://www.instructables.com/Evaluation-Setup-for-Real-World-Adversarial-Patche/). 

Our study presents a framework that uses a 3D digital replica of a real-world scene to create a realistic adversarial patch. We demonstrate the framework by creating several adversarial patches for a simple everyday scene: a coffee mug on a desk. The code in this repository can be used for crafting the adversarial patches (both for random and systematic sampling), perform a digital evaluation for the patch, and the code needed for the real-world evaluation. Please refer to the paper for more information.

## Project Overview
The configuration for the logger and the scene’s files are available in `configuration` and `data` respectively. All the information about the neural networks and ImageNet is available in `models`. The size of the checkpoints didn’t allow us to upload them, but they are [available here](TODO). The attack’s output, including images used for the attack and the adversarial patches throwout the attack, will be available in `out`. 

If you want to add a new scene then check `mug`, which contains all the files we used for the scene. Please use those files as an example for creating new scenes and attacks. In addition, `utils3d` contains all the code needed for working with a 3D object. Therefore, if you want to create a new scene you should inherit `Object3D` for each of the scene’s objects and `Scene3D` for the scene itself. Use the classes in `mug.py` as an example. You can also add new transformations (`transformation.py`), or use a different rendering program (`program.py`).

Additionally, check `attack`, which contains the two files needed to implement an attack. To create a new attack you can inherit `Attack` or `AttackBatches` (`attack.py`). We recommend using batches because the attack uses a lot of memory due to the data’s size. If you are attacking a new scene, inherit `AttackRender`, as shown in `mug_renderer.py`. You can also use one of the existing attack methods that were used during this study. If you wish to use systematic sampling in your attack you will have to supply the ranges for each transformation. Moreover, if you need to evaluate your attack then inherit `EvalRender` in `eval_renderer.py`. This evaluation process supports different evaluations, including the use of batches.

## Examples
We added to the root directory several examples that can help you reproduce our work, as well as using our code for future work. 
### Tips
- In the example files, we marked places that you can change with a comment like `# CHANGE HERE`. Please use search to make sure that you don’t miss anything. If you find a `# TODO` comment with the word “change”, ignore it. This is not for you. We will fix it later. Maybe. 
- For your convenience, we tried to supply all the examples with the coffee mug true class and armadillo target class (indexes 504 and 363 respectively in the ImageNet dataset). If you see those numbers without an explanation, this is the meaning of this value. In general, if you see an unexplained integer between 100-999, it is probably an index of a class label.
- If you create a new patch (i.e., a new texture), we highly recommend that you will use an image of size 471x181 px.
- Make sure that you place your files in the right location. In the examples of the evaluation process, we added comments to explain how to add a patch for evaluation. Please read those comments and follow the instructions. Avoid using the full path if it isn’t needed (i.e., if the example doesn’t use the full path, neither should you). 
- In some examples, you can find commented code that you can uncomment to get an additional feature (like plotting a graph of the results). However, some examples might contain unused code\parameters. Not everything is a feature. Sorry.
- When I uploaded the project I reorganized it so it will be easier for you to understand it and use it. It includes changes in files’ location, name changes, etc. I hope I didn’t break everything in the process. The code runs in my old environment, but I still didn’t check it in the new project. I’ll edit this once I’ll make sure that everything runs as it should.

### Crafting the Patches
`random_attack.py` and `systematic_attack.py` are the attacks we used in the paper to create the random and systematic patches. The parameters in those files are the same as we used in our work. The output of the attack includes the log files and three types of images. The info+ logs will be displayed in your console and the debug+ logs will be available in `/logs/logs.out`. Each run overwrites the old logs. Images of the views that were used in the attack, the patches, and the perturbation are available in the relevant folders in `/out/`. If you try to run the attacks, we strongly suggest that you will start with `random_attack.py`, and change the attack’s parameter to create a lighter attack. By doing so, you can make sure that everything runs without crashing without waiting for the attack to end. A possible configuration can be:

```python
attack = MugAttackBatchesRandomScene(ctx,
                                     target_class=363,
                                     true_class=504,
                                     batch_size=16,
                                     num_of_batchs=2,
                                     learning_rate=0.75,
                                     iteration_num=10,
                                     iter_to_log=2,
                                     iter_to_save_img=2)
```

`digital_evaluation.py` and `real_world_evaluation.py` are the base code for our evaluation process in the digital space and the real world respectively. You can use it to reproduce our results. 

### Evaluation Process
`digital_evaluation.py` allows you to evaluate a digital patch by rendering the replica from predefined positions. First, place your patch in `/data/patches/`, and add its name to the first line of the main. Don’t write the full path! `digital_evaluation.py` will complete it for you. Write only the name of the patch. To determine the ranges for the camera’s position (polar degrees), change the parameters for `eval_circle`. Then, change the parameters for `print_classifications` according to your needs. Please note that you can change the camera’s position, but the camera will always look at the center of the scene (i.e., the mug). This is a different process from our attack, which rotates and translates the scene in all three axes. Thus, the attack uses all the six degrees of freedom, while the digital evaluation uses less.

`real_world_evaluation.py` can be used to process the video taken by the webcam in our [real-world evaluation setup](https://www.instructables.com/Evaluation-Setup-for-Real-World-Adversarial-Patche/). For each image in the video stream, the code crops the photo to the size of Inception V3 input, feeds it to the classifier, and processes the results. If `REPEAT_CLASSIFY = True`, the program will immediately start classifying the scene until you close it or press `Q`. We strongly suggest using `REPEAT_CLASSIFY = False`, since it allows you the following:
- When you run the code, a window showing you the scene but will not classify it. 
- Press `C` to classify a specific frame.
- Press `S` to start a classification session. The program will keep classifying the frames until you press `S` again. After finishing a classification session, the program will print to the console the summary of the batch’s result.
- Pressing `Q` will stop the classification (if running), print the summary of the overall results, and exit.

### Extra Fun
We also supply additional code that *we didn’t use in our paper* but can help you in your work. `visual_digital_evaluation.py` allows you to perform a digital evaluation process while looking at the rendered images of the scene in real-time. This process wasn’t used in our work, but it can be a helpful debugging tool. `demo_with_classification.py` is a demo of the replica that can be used for both debugging and explaining to your supervisor why you stole all the white mugs from the lab's kitchen. The demo allows you to inspect the digital replica, and even classify specific frames. Please note that both files are old and unused code, so read the comments in the code to avoid additional bugs. 



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
