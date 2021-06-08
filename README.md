# modelTranslation

## 1、TensorRT模型转换软件简介

本软件用于将深度学习训练模型转换为TensorRT engine，转换得到的engine与InferenceAPI框架配合使用，用作深度学习加速推断。目前支持转换的模型有：1）Tensorflow-RetinaNet；2）Detectron2-RetinaNet；3）MMdetection-Yolov3。

## 2、软件运行环境

- VS2017
- TensorRT7.2.1.6
- Cuda11.1 + cudnn7.6.5
- Qt5.14.0

## 3、软件功能说明及演示

软件初始界面如下：

![aaa](https://github.com/qinxianglinya/modelTranslation/blob/main/images/%E4%B8%BB%E7%95%8C%E9%9D%A2.png)


功能一：将Tensorflow框架下的RetinaNet算法转换为TensorRT模型

![image-20210608103948203](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608103948203.png)

功能二：将Detectron2框架下的RetinaNet算法转换为TensorRT模型

![image-20210608104056015](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608104056015.png)

功能三：将MMdetection框架下的Yolov3算法转换为TensorRT模型

![image-20210608104003666](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608104003666.png)

**以转换Detectron2-RetinaNet为例**，填写以下参数：原始模型路径、TensorRT模型存放路径、缩放后的图片宽高、batchsize大小、推理精度、检测目标类别、FPN层数、后处理置信度阈值，**点击开始转换按钮，进行转换**。

模型转换中界面演示：

![image-20210608104801814](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608104801814.png)

模型生成完成界面演示：

![image-20210608105001928](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608105001928.png)

![image-20210608105151543](C:\Users\50126\AppData\Roaming\Typora\typora-user-images\image-20210608105151543.png)

