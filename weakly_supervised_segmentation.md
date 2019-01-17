### Learning deep features for discriminative localization
>  CAM 根据featuremap激活值来提取分割区域
  
### learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentaiton
>  提CAM,使用两种不同阈值的crf提取前景背景,根据前景背景训练一个affinity网络,产生一个矩阵,然后进行random walk,产生真正能用的label,再训练分割网络.
  
### self-explanatory deep salient object detection
>  显著object检测.使用了Unet一样的结构,不过把decoder改成dense连接的了.提出了解释显著性区域的图.具体来说就是把一块遮挡上,观察产生的featuremap有多大的变化.
  
### weakly- and semi-supervised learning of a deep convolutional network for semantic image segmentation
>  使用em算法,一边优化label,一边优化分割.
  
### revisiting dilated convolution: a simple appproach for weakly- and semi-supervised semantic segmentation 
>  cam和分割联合学习.使用了四种不同的dilated conv提取cam最后融合

### object region mining with adversarial erasing: a simple classification to semantic segmentation approach
>  提取一小块cam,然后擦掉,再提取下一块cam,一直到提取了全部的区域,之后在refine
  
### weakly and semi supervised human body part parsing via pose-guided knowledge transfer
>  从关键点进行retrieve,找到最接近的几张图,然后根据这几张图生成分割图,然后refine,最后扩充数据分割.
