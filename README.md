# Sequential/Hierarchical ERF-PSPNet

This code is the Pytorch implementation of Sequential/Hierarchical ERF-PSPNet architectures, which are designed for real-time semantic segmentation.

Hierarchical designs include 4x2 and 3x3 hierarchical architectures.

The input resolution is set to 640x480.

The code is tested with Python 3.6, Pytorch 0.4.1, CUDA 8.0.
Additional Python packages: numpy, matplotlib, Pillow, torchvision.

For training/deployment, you can also use the environment of [[PASS](https://github.com/elnino9ykl/PASS)],

the environment of [[ERFNet](https://github.com/Eromera/erfnet_pytorch)],
or the environment of [[PIWISE](https://github.com/bodokaiser/piwise)].

# Features

Network architectures: ERFNet, ERF-PSPNet (ERFNet with PSPNet),

Loss functions: Cross entropy, Focal loss,

Datasets: Cityscapes, Mapillary Vistas,

Data augmentations: Textual and geometric augmentations.

# News

Tensorflow implementation of ERF-PSPNet is [[here](https://github.com/Katexiang/ERF-PSPNET)].


![Example segmentation](figure_comparison.jpg?raw=true "Example segmentation")


# Publications
If you use this code in your research, please consider citing any of these related publications:

**Unifying Terrain Awareness for the Visually Impaired through Real-Time Semantic Segmentation.**
K. Yang, K. Wang, L.M. Bergasa, E. Romera, W. Hu, D. Sun, J. Sun, R. Cheng, T. Chen, E. López.
Sensors, 18(5), p. 1506. [[PDF](http://www.mdpi.com/1424-8220/18/5/1506/pdf)]

**Predicting polarization beyond semantics for wearable robotics.**
K. Yang, L.M. Bergasa, E. Romera, X. Huang, K. Wang.
In IEEE-RAS International Conference on Humanoid Robots (Humanoids), Beijing, China, November 2018. [[PDF](http://wangkaiwei.org/file/publications/humanoids2018_kailun.pdf)]

**Unifying terrain awareness through real-time semantic segmentation.**
K. Yang, L.M. Bergasa, E. Romera, R. Cheng, T. Chen, K. Wang.
In IEEE Intelligent Vehicles Symposium (IV), Suzhou, China, June 2018. [[PDF](http://wangkaiwei.org/file/publications/iv2018_kailun.pdf)]
