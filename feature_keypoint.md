### NetVLAD: CNN architecture for weakly supervised place recognition
> 把VLAD层扩展到CNN，端到端学习，之后结果用PCA来获得特征表示。把Hard assignment改成了soft以用于反向传播。学到的是图片级别的特征。

### LIFT: Learned invariant feature transform
> 同时做了检测，方向估计，和特征提取。里面多次使用Spatial transform用来提取角度，使用softargmax替换了NMS。实际上这三部分是反着训练的，也是使用了sfm的特征进行监督，训好第三个再第二个再第一个。学到的是类似SIFT一样的稀疏特征表示。

### **NIPS2018**&nbsp;&nbsp; Neighbourhood Consensus Networks
> 采用neighbourhood consensus or more broadly as semi-local constraints.邻域共识，半局部约束，解决同样的纹理，比如墙面特征相同而无法选取特征点的问题。4D卷积，平移不变性,强局部性，the network will determine the quality of a match by examining only the information in a local 2D neighbourhood in each of the two images.交换前后帧得到两个匹配结果，把不匹配的干掉，再卷积。没用argmax匹配，而是使用了soft nearest score. While this filtering step has no trainable parameters, it can be inserted in the CNN pipeline at both training and evaluation stages, and it will help to enforce the global reciprocity constraint on matches.
>用softmax得到匹配程度，This probabilistic intuition allows us to model the match uncertainty using a probability distribution and will be also useful to motivate the loss used for weakly-supervised training.通过卡阈值可以学习到一对图像间稀疏的匹配点。
>实际上是学了一个h\*w\*h\*w的一个关联,通过4D卷积学习相关性,然后通过一个filter,除以最大的概率来得到一个soft的概率.如果卡阈值求argmax的话可以得到最合适的匹配点.利用半监督的pair loss来学习. 如果用作localization的话,文中是对于像素值做了特殊处理.

### **NIPS2016**&nbsp;&nbsp; Universal Correspondence Network
> 强监督，通过图像间的点匹配关系，利用metric learning和全卷积网络，学习了一个dense的特征提取,loss就是匹配的更近,不匹配的希望大于一个阈值,对于不匹配的点对用了hard negative mining,比较耗时。利用了conv spatial transformer层。

### spatial transformer networks
> 用在输入图片之后或者网络之中,显式的model图片或者featuremap的平移旋转和crop,基本思路是通过conv/fc层学习一个映射矩阵,然后把这个矩阵作用在原图上,并配合可微的双线性插值,来将图像或者featuremap进行一个变换,变换后的feature再用于分类效果就会比较好.

### learning correspondence from the cycle-consistency of time
> 这篇文章很棒棒,开篇点明了correspondence的重要性.文章基于cycle的思想.在video中取一个patch,forward几帧再backward回来,让位置尽可能的相同.开始不好学,可能不能forward太多次,所以文章是forward少次和多次同时学,同时还允许跳着学,以用来解决遮挡问题.进行patch 匹配的时候下采样了8倍,不过貌似默认不同patch都是一样大了,这样可能会有问题.文章使用softmax后的affinity function (数量积dot product,对于笛卡尔向量其实就是(Frobenius) Inner Product) 进行计算相似度,使用l2范树约束学习.学到的特征很有用,可以用于关键点光流跟踪等任务,配合k-nn方法使用.

### Learning Dense Correspondence via 3D-guided Cycle Consistency
> 这篇文章学习两个图片(比如两辆车)的语义相关性.借助CADmodel,将两张互相转,两张图片转到各自的cad,这三个结果通过两个cad模型得到的gt来进行学习.所以是4-cycle.文章使用Siamese网络,合并feature加了两个分支分别学习光流场,和可见区域(object区域).损失函数应该是用的l2范数.
