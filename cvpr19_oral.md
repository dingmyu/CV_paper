### SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration Without Correspondences
> 是对RANSAC的一种改进方法，每次多取一些samples进行选取inliners。

### BAD SLAM: Bundle Adjusted Direct RGB-D SLAM
> 是用BA来做SLAM的一种方案。

### Revealing Scenes by Inverting Structure From Motion Reconstructions
> 稀疏点云 + RGB 重建真实场景，可以渲染出真实的frame

### Strand-Accurate Multi-View Hair Capture
> micro-scale, 3D hair重建, 使用了一种新型的line-based的multi-view stereo







### d-SNE: Domain Adaptation Using Stochastic Neighborhood Embedding
>

### Taking a Closer Look at Domain Shift: Category-Level Adversaries for Semantics Consistent Domain Adaptation
>

# ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation
> 对分割特征图进行minimize entropy，还加了GAN的loss进行训练。

# ContextDesc: Local Descriptor Augmentation With Cross-Modality Context
> 全局提取网络训练的特征描述符缺少local信息，因此选择加入keypoint信息，rgb content信息和keypoint geometry信息结合。

# Large-Scale Long-Tailed Recognition in an Open World
> 提出结合类别样本数目不均衡问题，结合分类，few-shot和noval class，训练了一个网络，bottom-up attention再top-down attention，记住原来的同时学习新的。attention in attention的思路。

# AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations Rather Than Data
> 是用auto-encoder对旋转平移等信息进行重建，给定一个图像和随机旋转评平移，通过重建无监督的学习更好的feature。

# SDC – Stacked Dilated Convolution: A Unified Descriptor Network for Dense Matching Tasks
> 使用dilated conv来学dense matching。kernel: [5,5,5,5], dilated rate: [1,2,3,4], 感受野为17，多个这样的module堆叠在一起。

# Learning Correspondence From the Cycle-Consistency of Time
> xiaolong wang的通过patch的cycle consistency学习dense matching的文章。

# AE2-Nets: Autoencoder in Autoencoder Networks
> 这篇做的是多任务？每个任务用autoencoder重建之后，为了证明隐变量是学到了相应的东西，再使用一个autoencoder对隐变量进行重建。

# Mitigating Information Leakage in Image Representations: A Maximum Entropy Approach
> 这篇是说把task无关的东西干掉，比如说superclass的分类不希望小类对其有影响。因此提出了Maximum Entropy，比直接最小化likelyhood要好。

# Learning Spatial Common Sense With Geometry-Aware Recurrent Networks
> 使用GRNN进行3D重建，这里说的主要就是传统的RNN连接的多个frame位置对不上，因此先基于geometry把2D都投影到3D空间，然后再连RNN就可以了

# Structured Knowledge Distillation for Semantic Segmentation
> 蒸馏网络，搞了pair-wise和pixel-wise的loss，认为需要蒸馏每个像素的概率。

# Scan2CAD: Learning CAD Model Alignment in RGB-D Scans
> release数据集，RGB和CAD model对于标注3D关键点.整体识别流程是寻找关键点，然后把某3dmodel关键点那一块丢进网络，会学到一个heat map出来，根据这个东西做align，最后基于ransac的方式得到9 dof的pose。

# Towards Scene Understanding: Unsupervised Monocular Depth Estimation With Semantic-Aware Representation
> 使用语义分割帮助优化单目depth，搞了一个左右眼的consistency，另外也可以帮助平滑。语义分割和depth网络共享一个主干网络，在一个位置concat不同的task identity会有不同的结果，这样就可以使用非pair的数据。

# Tell Me Where I Am: Object-Level Scene Context Prediction
> 使用部分的semantic segmentation map来生成完整的layout，比如给一个人，那么学习一个shape和一个bbox的proposal，可以是一个天空，把这个贴到bbox的部位就可以了，监督信息就是完整的segmentation map。搞了一堆gan loss

# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
> 搞了一个NOCSmap，使用颜色来表示？没太看懂

# Supervised Fitting of Geometric Primitives to 3D Point Clouds
> 3D物体重建，每一个物体分解成了好多个类型的面，把它们融合起来学习，没太看懂。

# Do Better ImageNet Models Transfer Better?
> 做了好多好多实验，探究不同网络不同task，用不用imagenet预训练模型的效果差异。
