### DSAC - Differentiable RANSAC for Camera Localization
@inproceedings{brachmann2017dsac,
  title={DSAC-differentiable RANSAC for camera localization},
  author={Brachmann, Eric and Krull, Alexander and Nowozin, Sebastian and Shotton, Jamie and Michel, Frank and Gumhold, Stefan and Rother, Carsten},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  volume={3},
  year={2017}
}
> 经典方法RANSAC，每次随机初始化几个点，用这一部分点拟合，然后查看有多少点符合拟合结果。最后选取最多点符合的一次。是一种通过局部匹配得到全局模型的办法，或者说局部预测，全局拟合。不可微的原因是argmax，修改成可微有两种方案，第一种是softmax，第二种是基于概率的选择。作者后来发现基于概率的选择效果比较好。

### Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network 
> NNnet,Siamese网络直接计算相对pose

### RelocNet: Continuous Metric Learning Relocalisation using Neural Nets 
> 这篇是加入了视角重合的比例作为一个额外的约束

### Deep Auxiliary Learning for Visual Localization and Odometry 
> 这篇同时学习了相对和绝对pose,学习绝对pose的同时也保证其满足相对pose的约束关系,结果比前面两个高的多.

### VLocNet++: Deep Multitask Learning for Semantic Visual Localization and Odometry 
>加入语义分割同时训练,结果提升很多,这种加语义的paper去年有好多,因为语义特征更为稳定

### leveraging deep visual descriptors for hierarchical efficient localization
> 实现了相对pose的一个pipeline,看起来更像一个工程论文. pose聚类+pnp ransac,使用了netvlad层,用的sift特征.

### from coarse to fine: robust hierarchial localization at large scale
>　和上一篇差不多，2d-3d的匹配，加入了distillation学习更小的网络，参考文献可以参考，有几篇学习low-level特征的，比sift特征更好。（superpoint）
