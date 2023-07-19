# TextDet-Rec
这是一个用于文本检测和识别的程序，可以检测和识别场景中的文字（包括3000+汉字，数字，字母和部分符号）
该程序可以分为文本检测和字符识别两个部分，每个部分都通过深度神经网络实现。其中，文本检测部分采用模型CRAFT，字符识别部分采用模型EfficientNet
## 程序模块介绍
### 文本检测模型 CRAFT
[CRAFT](https://arxiv.org/abs/1904.01941)是一个文本检测模型，通过VGG16+UNet架构下的神经网络输出一张表示像素点属于字符区域或字符间区域的可能性的热力图，并根据该图来分割出文本区域。
![](./assertsForREADME/CRAFT_ex.png)
由于CRAFT模型可以输出一张表达字符区域的热力图，该输出也可以用来分割字符区域。将分割出的字符区域作为字符识别模型的输入，就可以获得字符识别的结果。
CRAFT预训练模型已经开源至[github](https://github.com/clovaai/CRAFT-pytorch)，对代码的后处理部分做部分修改和封装后，该模型可以达到我们的要求。
### 字符识别模型 EfficientNet
[EfficientNet]()是谷歌发布的一个CNN神经网络模型，通过对神经网络宽度，深度和输入图片分辨率组合的不断调整得到，可以快速准确的运行图片分类工作。
为了实现字符识别的功能，我使用了[中科大的手写汉字数据集](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html)HWDB1.1，包含3755个字符。
![](./assertsForREADME/HWDB_ex.png)
最终的字符识别模型在HWDB训练集上迭代了45次，在HWDB测试集上的准确率达到了92.3%
### 实现细节
为了能够识别自然场景下的文本，EfficientNet模型的输入是RGB三通道图片。但HWDB数据集是单通道图片且背景颜色统一，在训练时，我通过PIL库直接将单通道图片转成三通道图片，这种转化有一定的效果，但也导致模型难以准确识别背景颜色复杂的文字。因此，我将CRAFT输出的图片先转化成单通道图片，进行一些二值化处理后再传给EfficientNet，这比较好地解决的问题。 
程序接受一张图片后，会输出一张框出了字符区域并写出相应预测字符的图片。
## 程序运行方法
程序中含有4个参数: 
* image_path：输入图片的路径，需要输入main.py的相对路径，默认为``./asserts/poem.jpg``
* font_path：输出图片上预测字符的字体，需要输入main.py的相对路径，默认为``./fonts/DENG.ttf``
* result_root：输出图片所在的目录，需要输入main.py的相对路径，默认为``./results/``
* debug：打印调试信息，会在控制台输出调试信息，并在输出图片时额外输出每个单独文字区域的原图像，处理后图像，预测字符排序，默认为``True``

要运行程序，请在TextDet-Rec目录下运行指令``python main.py --image_path==./asserts/<your img>``。输出结果会保存在``./results/<your img>/``目录下

## 实验效果
目前，该项目可以分割出图片中的字符区域并尝试预测，对白色背景下的汉字字符识别准确率较高。但由于是脱机识别，且类别数过多，存在一些识别错误的现象。
![](./assertsForREADME/example1.png)
![](./assertsForREADME/example2.png)