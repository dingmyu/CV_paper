## 单目/双目

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

### Mono3D++：
这篇文章我觉得很solid的，首先related work写得很好，都可以用来参考。这篇文章是用pascal数据集构建了2d关键点，并且算了平均的3dshape，使用2d loss，2d 关键点，3d框和3dshape这些loss，通过重投影，利用了shape的先验信息。此外还学习了单目深度，貌似是通过一个patch来搞的，然后学了地平线的法线，论文的loss function看起来也合理，值得借鉴。

### multi-level fusion based 3d object detection from monocular images
武大的。这篇主要贡献应该是先学了个单目depth，然后把depth和rgb结合提取特征，通过depth还生成了电云，也直接pooling提取特征，然后和RGB的特征融合。

### ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape
2019.4.10.慕尼黑工业。深度好文。本文先使用一个2d detection网路提取bbox，然后结合单目depth 提出一个可微的ROI lifting操作，搞出了3d bbox的八个点。pose用四元数q加上whl来表示，根据bbox的位置x，y和深度z得到了八个点，然后学这八个点就可以了。用了warm up。文章指出了以自我为中心和非自我为中心的不同之处，没太看懂。然后还搞了学一个mesh的shape，并看了看shape空间内feature意义，貌似是这样？

### Shift r-cnn
这篇文章只有四页。related work写得比较中肯，可以参考。Monocular 3D object detection is the most difficult task since it
requires tackling the inverse geometrical problem of mapping 2D space into 3D space, in the absence of any true 3D information. Thus, top approaches rely on extra training data in order to make informed 3D estimations. Wang et al. [22] use monocular depth perception networks such as DORN [4] to generate pseudo-point clouds and then apply a state of the art LiDAR-based model [9]. ROI-10D [15] uses monocular depth estimation and lifts 2D detection into a 6D pose problem. Mono3D [1] enforces 3D candidates to lay on the ground plane, orthogonal to image plane. It also uses semantic segmentation, contextual information, object size, shape and location priors. Deep3DBox [16] uses geometric constraints by tightly fitting the 3D bounding box into the 2D box. Xu et al. [23] use multi-level fusion of Faster R-CNN and depth features for 3D object detection. Advances in stereo and monocular depth estimation could provide accurate 3D information, which could greatly improve non-LiDAR based 3D object detection systems. 这篇文章没用depth信息。采用了deep3dbox的思路通过两阶段算了一个解，但是position是算出来的，所以不准。这篇文章又通过前面结果回归出了一个新的position，并提出了一个loss，这个loss感觉是postion的三个维度通过角度投影过后的值的一个加权平均。

### Monocular 3d object detection and box fitting trained end-to-end using intersection-over-union loss
19年6月。这篇文章看起来有点厉害，kitti上13的结果，比monogr要高。思想是说直接回归3d参数是不准的，包括loss和梯度方向模型都不好学，因此2d上预测了bbox和3d的八个点以及depth，通过这些信息再预测3d box fitting。提出了几种loss/算梯度的方法，然后也利用了uncertainty，貌似是通过学习一个小的，下采样五倍的一个方框seg map来实现的。

### towards scene understanding with detailed 3D object representations
只有两页，文章讨论了几种3d形式，关键点啊，方框啊，不规则方框，视锥之类的。有一个思想可以借鉴，解决遮挡问题。主动的把车挡住一小块，让网络去预测被挡住的位置的关键点，不知道是不是可行。

### Eliminating the Blind Spot: Adapting 3D object detection and monocular depth estimation to 360 panoramic imagery
ECCV 2018。这篇文章主要就是把原来的前视摄像头给改成了全景图片，做了depth和3d detection两个任务。提出了一个数据集。讲解了一下全景摄像头的内参等等知识，根据这些，套用已有方法，还用了domain transter（cycle gan）来对全景图片做检测。

### 3D-RCNN: instance-level 3d object reconstruction via render-and-compare
cvpr。这篇是车和object都可以做。提了roi之后回归box和3d中心投影点，然后指出了同样大的人用不同姿势出来的roi feature形状是不一样的，因此学了一组crop参数并concat到了roi feature里，然后学习shape参数和pose参数。根据cad model训练了的shape参数，通过这个参数把车给渲染出来和原图算loss。这一步用的还是可微的渲染。

### 3d object detection and viewpoint estimation with a deformable 3d cuboid model
12年的文章.是基于DPM来做的,具体就是把一个车拆成前边侧边上面,叫做aspect.然后每一面使用share部分weight的模型来做.然后利用了deformation model (stitching point).

## 双目

### 3D object proposals using stereo imagery for accurate object class detection
17年的。这篇文章是想要尽量多利用信息，context信息，freespace等等。因此proposal之后还外扩一圈提取信息和proposal的信息结合。这篇文章是调研了很多传统方法，使用双目或者lidar（voxel）作为输入，尽量多的利用了特征，搞了个能量函数，提取3d proposal。和目前最新的方法都不是一个套路了。

### Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving
cvpr2019,这篇争议最大的点是，方法都是别人的，它真的只是“bridging”了一下。首先用stereo的方法得到了双目深度，然后用公式转成伪点云，再用lidar detection的方法跑个检测，走通了双目3d检测的一条新的pipeline。

### monocular 3d object detection with pseudo-lidar point cloud
2019,CMU的。这篇和pseudo lidar那篇有异曲同工之妙，和洪伟的paper也有点像。是提取了depth，然后根据depth生成pesudo lidar，然后根据instance segmentation从中抠出视锥来，根据视锥学习一个3d detection，这里用了二阶段，还学了一个残差，最后搞了一个2d-3d consistency，这篇文章一看流程图就懂了。

### Pseudo-Lidar++
2019，康奈尔的。这篇延续了pseudo lidar的套路。主要有两个创新点，第一是说通过disparity来估计depth，越远处偏差越大，这是系统误差不是随机误差，因此要直接估计depth，在PSMnet基础上改进的。第二是说做了一个depth的优化，通过便宜的4线lidar的稀疏点云，保留生成的pseudo-lidar的shape，通过knn的方式把一个点拉过来，周围的也跟着过来，大概是这种思想，得到了更好的depth。后面还是相同的套路了。


### Stereo R-CNN based 3D Object detection for autonomous Driving
cvpr2019.参考文献写得不错。双目搞了两个ROI，gt是取的左右框的交集，然后左右的特征进行concat学习3d box部分参数，后面也学了一下特征点（四个关键点，都在底边）。之后用了3d box estimator通过关键点和3d参数来解出position，然后是3d box alignment，是一个重投影误差用来求解最佳中心depth。


## lidar

### Fast and Furious: Real Time End-to-end 3d objection, tracking and motion forecasting with a single convolutional net
雷达点云,多帧结合,统一坐标系,使用高度作为feature提取voxel.多帧预测bbox然后根据score和iou进行平均融合.提出前融合后融合这两种常见的融合方案.

### OBject detection and classification in occupancy grid maps using deep convolutional networks
2018,将点云数据转换成occupancy grid maps，其实就是多种俯视图的特征，可以看论文上的图，然后将多种俯视图特征利用2d detector来检测，在kitti bird‘s eye view上测试。

### Capturing Object Detection Uncertainty in Multi-Layer Grid Maps
2019.1,这篇和上面那篇是一个套路的，搞了grid maps，此外说学习uncertainty是有用的。因此学了一个（角度上的？）uncertainty。（参考另一篇论文）通过最大和最小角度可以可视化出一个95%概率的3d壳，后面有些讨论。感觉这两篇都是水文。


## lidar+ rgb

### multi-view 3d object detection network for autonomous driving
百度的一篇很强很大的工作.使用lidar+rgb.这篇是利用了faster-rcnn的思想,把2d的proposal改成了3d的proposal,然后投影到三个面做roi pooling,一个是lidar bv, 一个是lidar front, 一个是image.这三个分别使用不同的conv层提取特征,提取之后三个roi pooling,然后将三个特征进行了deep fusion最后进行分类回归,从而完成3d det任务.


### Joint 3D proposal generation and object detection from view aggregation
这篇文章写得贼乱。同时用了原图和lidar俯视图和3d anchor grid。rpn后fusion然后经过nms得到检测结果。没仔细看

