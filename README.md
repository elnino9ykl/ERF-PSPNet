# Sequential/Hierarchical ERF-PSPNet

This code is the Pytorch implementation of Sequential/Hierarchical ERF-PSPNet architectures, which are designed for real-time semantic segmentation.

Hierarchical designs include 4x2 and 3x3 hierarchical archictures.
The input resolution is set to 640x480.

The code is tested with Python 3.6, Pytorch 0.4.1, CUDA 8.0.
Additional Python packages: numpy, matplotlib, Pillow, torchvision.

For training/deployment, you can also use the environment of [ERFNet] (https://github.com/Eromera/erfnet_pytorch)
                                 or the environment of [PIWISE] (https://github.com/bodokaiser/piwise)

Network architectures: ERFNet, ERF-PSPNet (ERFNet with PSPNet)
Loss functions: Cross entropy, Focal loss
Datasets: Cityscapes, Mapillary Vistas
Data augmentations: Textual and geometric augmentations

# Publications
If you use this code in your research, please consider citing any of these related publications:

**Unifying Terrain Awareness for the Visually Impaired through Real-Time Semantic Segmentation.**
Yang, K., Wang, K., Bergasa, L.M., Romera, E., Hu, W., Sun, D., Sun, J., Cheng, R., Chen, T. and López, E., 2018. 
Sensors, 18(5), p.1506. [PDF](http://www.mdpi.com/1424-8220/18/5/1506/pdf)

**Predicting polarization beyond semantics for wearable robotics.**
Yang, K., Bergasa, L.M., Romera, E., Huang, X. and Wang, K. In IEEE-RAS International Conference on Humanoid Robots (Humanoids2018), Beijing, China, November 2018. [PDF](http://wangkaiwei.org/file/publications/humanoids2018_kailun.pdf)
