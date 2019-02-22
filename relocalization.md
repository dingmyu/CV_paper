### DSAC - Differentiable RANSAC for Camera Localization
@inproceedings{brachmann2017dsac,
  title={DSAC-differentiable RANSAC for camera localization},
  author={Brachmann, Eric and Krull, Alexander and Nowozin, Sebastian and Shotton, Jamie and Michel, Frank and Gumhold, Stefan and Rother, Carsten},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  volume={3},
  year={2017}
}
> 将RANSAC用到camera localization中，是一个迭代优化的过程。网络预测2d到3d的投影，四个点可以确定一个pose,采样很多组，并对每一个估计的pose打分，取打分最高的一组，继续加入新的点并重新打分，直到找到最多点匹配上的一组对应的pose。经典方法RANSAC，每次随机初始化几个点，用这一部分点拟合，然后查看有多少点符合拟合结果。最后选取最多点符合的一次。是一种通过局部匹配得到全局模型的办法，或者说局部预测，全局拟合。不可微的原因是argmax，修改成可微有两种方案，第一种是softmax，第二种是基于概率的选择。作者后来发现基于概率的选择效果比较好。

### 2018cvpr  Learning Less is More – 6D Camera Localization via 3D Surface Regression
> 直接回归学习一个2d-3d的correspondence，把DSAC的网络改成了fcn，把DSAC里学习correspondence和打分的过程放到一个网络里联合学习。精度很高，室外20cm.

### scene coordinate regression with angle-based reprojection loss for camera relocalization
> 解决了less-more(dasc++)为了丢掉3dmodel而需要进行一个depth的大概初始化的问题.将重投影误差改成了角度误差,同时引入了multi-view的损失,也尝试了学光流用的photometric loss,L1范数+0.85SSIM.使用新的loss对初始化不敏感,在没有3dmodel的情况下结果更好.

### geometry-aware learning of maps for camera localization
> mapnet,李代数学习角度,同时相对和绝对pose,  mapnet+ 加入了unlabeled data, mapnet++ 加入了pgo

### exploiting points and lines in regression forests for RGB-D camera relocalization
> 使用回归森林模型，使用lsd提取了线段进行学习，并随机sample点,然后用ransac求定位.做了一个点应该在线段内的约束.

### are large-scale 3d models really necessary for accurate visual localization?
> 比较久的文章,但是刚中.讲不需要复杂的3d模型,做了个数据集,用vlad特征,spatial reranking和局部sfm的方法,达到了和3d模型近似的精度.

### netvlad
> vlad的深度版本

### posenet
> 开山之作

### Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network 
> NNnet,Siamese网络直接计算相对pose

### RelocNet: Continuous Metric Learning Relocalisation using Neural Nets 
> 这篇是加入了视角重合的比例作为一个额外的约束,camera pose retrieval更准.(NNnet基础上)

### Deep Auxiliary Learning for Visual Localization and Odometry 
> 这篇同时学习了相对和绝对pose,学习绝对pose的同时也保证其满足相对pose的约束关系,结果比前面两个高的多.

### VLocNet++: Deep Multitask Learning for Semantic Visual Localization and Odometry 
>加入语义分割同时训练,结果提升很多,这种加语义的paper去年有好多,因为语义特征更为稳定

### leveraging deep visual descriptors for hierarchical efficient localization
> 实现了相对pose的一个pipeline,看起来更像一个工程论文. pose聚类+pnp ransac,使用了netvlad层,用的sift特征.

### from coarse to fine: robust hierarchial localization at large scale
>　和上一篇差不多，2d-3d的匹配，加入了distillation学习更小的网络，参考文献可以参考，有几篇学习low-level特征的，比sift特征更好。（superpoint）

### Mask-SLAM: Robust feature-based monocular SLAM by masking using semantic segmentation
> 加入了segmentation一起学，segmentation提供了一个mask保证slam的特征点不选在天空和动态物体上。

### 2018  Practical Visual Localization for Autonomous Driving: Why Not Filter?
> 前端还是posenet或者mapnet，提出了一个后处理方案，比mapnet里的pgo结果好，主要就是根据车速有范围，以及时序上的一致性把预测结果平滑了。结果提升很多

### 2018  ACCURATE VISUAL LOCALIZATION IN OUTDOOR AND INDOOR ENVIRONMENTS EXPLOITING 3D IMAGE SPACES AS SPATIAL REFERENCE
> 比较水，基本是一个已有方法的工程实现。

### Project AutoVision Localization and 3D Scene Perception for an Autonomous Vehicle with a Multi-Camera System
> 比较水，基本是一个已有方法的工程实现。

### Semantic Visual Localization
> 使用语义地图，可以减少由于环境气候光线变化导致的定位不准的问题。用了一个3d网络对语义地图进行补充。3d-3dmatching

### inLoc: Indoor Visual Localization with Dense Matching and View Synthesis
> dense feature extraction and matching.用cnn提pixel-level的特征，先用高级语义特征作粗略匹配，再用低级纹理特征精确匹配,做了一个dense的matching.总体流程 netvlad提特征选取top10,然后dense匹配,ransac计算pose,然后做了一个合成视图用来验证pose准确性,最后根据这十个估计一个pose.需要用到depth,提出了一个inloc数据集.

### Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
> 白天黑夜 四季数据集  benchmark

### random forests versus neural networks -- what's best for camera localization?
> 将回归森林模型转化成了神经网络,并且进行端到端的训练,最后做了一个平均

### Semantic Match Consistency for Long-Term Visual Localization
> 用语义信息解决季节变化问题。建了一个3d语义地图，query的时候去匹配并计算语义的匹配程度。把这个匹配程度作为ransac的权重。

### Efficient 2D-3D Matching for Multi-Camera Visual Localization
> 多相机定位，也是基于ransac的迭代的方法，通过多相机可以过滤掉outlier，然后迭代优化。

### eccv18 VSO: Visual Semantic Odometry
> 在传统的VO领域加了语义分割。

### Night-to-Day Image Translation for Retrieval-based Localization
> 训了一个GAN， 夜晚转白天，然后再定位。

### Variational End-to-End Navigation and Localization
>定位+决策联合学习

### CVM-Net: Cross-View Matching Network for Image-Based Ground-to-Aerial Geo-Localization
> 卫星图定位，先用两个网络把卫星图和相机图投影到一个共同的空间，再去学习其相关性。

### 2018eccv  CPlaNet: Enhancing Image Geolocalization by Combinatorial Partitioning of Maps
> 也不是自动驾驶场景的定位，是在google地图上的那种定位，误差有几KM。思想是把地图分成不同组super pixel，每一块用于分类。结合多次分类的结果选取最可能的位置。

### on-the-fly adaptation of regression forests for online camera relocalisation
> 回归森林模型,2d-3d匹配,改成了online的版本,可以在一个场景finetune,另一个场景finetune

### real-time rgb-d camera pose estimation in novel scenes using a relocalisation cascade
> 上一篇回归森林模型的扩展,将最后ransac出的一个固定的pose改为概率输出,选择一个分最高的.提出了级联的结构,第一级精度差速度快,之后精度高速度慢,不同阶段做不同的取舍.

### scene coordinate and correspondence learning for image-based localization
> 用神经网络回归3d场景坐标,加了个平滑项.然后通过学习特征点为场景坐标预测权重,最后使用ransac.

### full-frame scene coordinate regression for image-based localization
> FCN回归场景坐标,然后ransac计算pose.通过微调pose反投影做了数据增强,被less more吊打了.

### learning to find good correspondences
> 使用sift提取n个特征点对,然后网络输入n\*4进行回归其权重,最后使用八点法进行定位,可微的loss.训练完又用了ransac

### efficient condition-based representations for long-term visual localization
> 解决robotcar-seasons不同环境场景的问题,用了多个encoder,根据不同场景选取不同的encoder,用了generalized mean pooling. retrive但并没有计算相对pose,感觉比较水.

### Learning to Navigate the Energy Landscape Julien
> The proposed method is essentially a discrete optimization technique that uses a hierarchical navigation graph and a retrieval forest. 有点理论,主要是树相关的方法,没看懂

### uncertainty-driven 6d pose estimation of objects and scenes from a single rgb image
> rgbpose.实际上是一篇只用rgb做object pose的文章,但也可以做localization.是先估算了一个大概的3d位置概率模型,然后使用ransac进行优化的方法.但是训练的时候也用到了depth,只是test的时候纯rgb.
