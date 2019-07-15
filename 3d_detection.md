## 单目

### 3D bounding box estimation using deep learning and geometry
> 开山之作?是使用现成的2d detector, 把图里的车抠出来学它的旋转和维度(长宽高), 然后根据2d和3d的关系, 算出3d bbox的位置(location).精度不高但是数学可以借鉴.可以考虑加可学习的权重, 将loss反传回detection model.

### joint monocular 3d vehicle detection and tracking
用faster-rcnn检测,并根据上篇论文将检测出的2d框转换成3d.faster rcnn同时估计一个3d中心投影到2d图中的位置.根据前后帧产生的框来计算一个联系矩阵,判断哪些框是同一个车.最后根据这些信息refine了一下.所以是3d detection和tracking一起做.

### Deep MANTA: a coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image
提出使用关键点来表示车的3d结构.使用CAD模型?做了各种车辆类型和大小的模板.文章用了cascade的结构对rcnn结果进行了refine.同时学习2d bbox和part坐标,part的可见性(我感觉这里part指的就是那些关键点),以及属于那个模板.学会了之后套模板再进行2d-3dmatching.

### GS3D: An efficient 3d object detection framework for autonomous driving
这篇文章计算了2d bbox和一个大概的方向，然后估计了3d bbox的一个大概位置，然后把三个面分别取出来，然后alignment提取特征。然后使用了基于分类的loss代替了回归loss。想到可以用deformable convolution？把0/1 label改成了一个soft的形式。

### Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction
这篇文章就很有意思很厉害了，对于每一个instance，文章是对其进行一个重建，相当于估计一个局部的3d点云出来。使用proposal-base的方法能够大大减小3d的search space。文章提了global和local的feature，然后估计出3d bbox的参数。同时使用估计出来的点云经过学到的位置放到3d空间，然后再投影到2d空间算一个l1 loss。因此是一个多任务学习。

### MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization
这篇文章不用额外data，单模型只用3d bbox作为监督。估了一个粗糙的depth，图片划分网格，每个网格一个depth，然后根据depth的大小关系处理了一下遮挡问题。模型输入单张rgb，使用浅层特征训练了delta location和corner的offset（个人觉得不是很make sense）。深层特征计算了instance depth和3d中心。深层特征输出大概的bbox位置后，浅层特征用来refine。

### 3d bounding boxes for road vehicles: a one-stage, localization prioritized approach using single monocular images
看起来像是一篇水文，全文的main contribution就是预测了一个关键点，这个关键点指的是车辆底部的中心。然后做了个投影回归的约束。

### Orthographic feature transform for monocular 3d object detection
这篇是搞了两个transform操作，把image feature投影到了voxel再投影到bird-eye view，然后直接在最后的feature上用高斯图进行学习，loss是l1。最后用了一个nms得到结果。结果不高，车的ap只有3.

## lidar

### Fast and Furious: Real Time End-to-end 3d objection, tracking and motion forecasting with a single convolutional net
雷达点云,多帧结合,统一坐标系,使用高度作为feature提取voxel.多帧预测bbox然后根据score和iou进行平均融合.提出前融合后融合这两种常见的融合方案.


## lidar+ rgb

### multi-view 3d object detection network for autonomous driving
百度的一篇很强很大的工作.使用lidar+rgb.这篇是利用了faster-rcnn的思想,把2d的proposal改成了3d的proposal,然后投影到三个面做roi pooling,一个是lidar bv, 一个是lidar front, 一个是image.这三个分别使用不同的conv层提取特征,提取之后三个roi pooling,然后将三个特征进行了deep fusion最后进行分类回归,从而完成3d det任务.


### Joint 3D proposal generation and object detection from view aggregation
这篇文章写得贼乱。同时用了原图和lidar俯视图和3d anchor grid。rpn后fusion然后经过nms得到检测结果。没仔细看
