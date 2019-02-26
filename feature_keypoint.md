### NetVLAD: CNN architecture for weakly supervised place recognition
> 把VLAD层扩展到CNN，端到端学习，之后结果用PCA来获得特征表示。把Hard assignment改成了soft以用于反向传播。学到的是图片级别的特征。

### LIFT: Learned invariant feature transform
> 同时做了检测，方向估计，和特征提取。里面多次使用Spatial transform用来提取角度，使用softargmax替换了NMS。实际上这三部分是反着训练的，也是使用了sfm的特征进行监督，训好第三个再第二个再第一个。学到的是类似SIFT一样的稀疏特征表示。

### **NIPS2018**&nbsp;&nbsp; Neighbourhood Consensus Networks
> 采用neighbourhood consensus or more broadly as semi-local constraints.邻域共识，半局部约束，解决同样的纹理，比如墙面特征相同而无法选取特征点的问题。4D卷积，平移不变性,强局部性，the network will determine the quality of a match by examining only the information in a local 2D neighbourhood in each of the two images.交换前后帧得到两个匹配结果，把不匹配的干掉，再卷积。没用argmax匹配，而是使用了soft nearest score. While this filtering step has no trainable parameters, it can be inserted in the CNN pipeline at both training and evaluation stages, and it will help to enforce the global reciprocity constraint on matches.
>用softmax得到匹配程度，This probabilistic intuition allows us to model the match uncertainty using a probability distribution and will be also useful to motivate the loss used for weakly-supervised training.通过卡阈值可以学习到一对图像间稀疏的匹配点。
>实际上是学了一个h\*w\*h\*w的一个关联,通过4D卷积学习相关性,然后通过一个filter,除以最大的概率来得到一个soft的概率.如果卡阈值求argmax的话可以得到最合适的匹配点.利用半监督的pair loss来学习. 如果用作localization的话,文中是对于像素值做了特殊处理.

### **NIPS2016**&nbsp;&nbsp; Universal Correspondence Network
> 弱监督，通过图像间的匹配关系，利用metric learning和全卷积网络，学习了一个dense的特征提取。利用了conv spatial transformer层。

