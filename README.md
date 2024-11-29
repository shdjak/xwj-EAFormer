EAFformer: Edge-aware guided Adaptive Frequency-navigator Network for Image Restoration
===

Wenjie Xie, Dong Zhou, Wenshuai Zhang, Wenrui Wang, Dan Tian

>**Abstract**: Although many deep learning-based image restoration networks have emerged in various image restoration tasks, most can only
 perform well in a specific type of restoration task and still face challenges in the general performance of image restoration. The
 fundamental reason for this problem is that different types of degradation require different frequency features, and the image needs
 to be adaptively reconstructed according to the characteristics of input degradation. At the same time, we noticed that the previous
 image restoration network ignored the reconstruction of the edge contour details of the image, resulting in unclear contours of the
 restored image. Therefore, we proposed an edge-aware guided adaptive frequency navigation network, EAFormer, which extracts
 edge information in the image by applying different edge detection operators and reconstructs the edge contour details of the image
 more accurately during the restoration process. The adaptive frequency navigation perceives different frequency components in the
 image and interactively participates in the subsequent restoration process with high and low-frequency feature information, better
 retaining the global structural information of the image and making the restored image more visually coherent and realistic. We
 verified the versatility of EAFormer in five classic image restoration tasks, and many experimental results also show that our model
 has advanced performance.


Network Architecture
----
![](https://github.com/shdjak/xwj-EAFormer/blob/main/Network%20Architecture.jpg)

Installation
---
Prepare the environment of cuda11.8, torch2.0.1, torchaudio2.0.2, torchvision0.15.2.

After that, run:<br>
`python setup.py develop --no_cuda_ext`

Data preparation
---
The format of the dataset is [here](https://github.com/swz30/Restormer/tree/main/Deraining/Datasets)

Traning 
--
Take desnowing as an example:<br>
`python basicsr/train.py -opt Desnowing/Options/Desnowing.yml`

Testing 
--
You can refer to [here](https://github.com/swz30/Restormer/blob/main/Deraining/README.md#evaluation)

Contact
--
If you have any questions, please contact xiewenjie_uestc@163.com.

**Acknowledgment:** Our code is based on the [Restormer](https://github.com/swz30/Restormer) repository.<br>




  
