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
