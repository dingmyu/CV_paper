### NetVLAD: CNN architecture for weakly supervised place recognition
> 把VLAD层扩展到CNN，端到端学习，之后结果用PCA来获得特征表示。把Hard assignment改成了soft以用于反向传播。学到的是图片级别的特征。

### LIFT: Learned invariant feature transform
> 同时做了检测，方向估计，和特征提取。里面多次使用Spatial transform用来提取角度，使用softargmax替换了NMS。实际上这三部分是反着训练的，也是使用了sfm的特征进行监督，训好第三个再第二个再第一个。学到的是类似SIFT一样的稀疏特征表示。

### Universal Correspondence Network
> 弱监督，通过图像间的匹配关系，利用metric learning，学习了一个dense的特征提取。利用了conv spatial transformer层。

