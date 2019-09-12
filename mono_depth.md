### Learning Depth from Monocular Videos Using Synthetic Data: A Temporally-Consistent Domain Adaptation Approach
> https://arxiv.org/pdf/1907.06882.pdf。学习单目depth，用了flow和camera pose，感觉还是老套路，不同之处是使用了合成数据集，加了个预测动态车的mask，然后结合了一下Domain Adaptation的方法，总体感觉是一个拼积木一样的方法，效果肯定好。

### Depth from videos in the wild: unsupervised monocular depth learning from unknown cameras
> 这篇文章也是通过video无监督学习depth，同时学了一个动态mask，学了pose。不同的是这篇文章学习了内参，说自己是第一个学习内参的，这样就可以把不同的数据集杂糅起来训练。提出一个很有意思的事情：就是训练depth的时候使用train mode比test mode效果要好，也就是使用样本的均值方差而不是统计的均值方差效果更好。作者说batch size越大效果越差，因此把BN替换成了LN效果更好，那么IN效果咋样呢，我很好奇。

### Learning Guided Convolutional Network for Depth Completion
> 2019 TIP.这篇做的是depth completion。是使用image feature作为卷积核用在了depthnet里面，叫做guided filter。但是这样特别费显存，所以作者为了节省显存，使用了两阶段的办法。第一阶段首先使用depth-wise的卷积，第二阶段才cross-channel来搞。
