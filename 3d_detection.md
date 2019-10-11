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
这篇文章就很有意思很厉害了，对于每一个instance，文章是对其进行一个重建，相当于估计一个局部的3d点云出来。使用proposal-base的方法能够大大减小3d的search space,还是先预测出了box大小和方向，然后结合2d box算出了position，depth通过相似三角形可以得到。文章提了global和local的feature，然后估计出3d bbox的参数。同时使用估计出来的点云经过学到的位置放到3d空间，然后再投影到2d空间算一个l1 loss。因此是一个多任务学习。文章说自己是的一个报行人和自行车成绩的，然后学习电云设置了三个loss，一个是点云之间的l1，一个是点云最后一维和depth的smooth l1，另外一个是点云投影回图像做一个smooth l1 loss，这个是不是可以利用起来，局部预测的点云和全局的depth生成点云结合，或者结合视锥等等做一个attention。

### MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization
这篇文章不用额外data，单模型只用3d bbox作为监督。估了一个粗糙的depth，图片划分网格，每个网格一个depth，然后根据depth的大小关系处理了一下遮挡问题。模型输入单张rgb，使用浅层特征训练了delta location和corner的offset（个人觉得不是很make sense）。深层特征计算了instance depth和3d中心。深层特征输出大概的bbox位置后，浅层特征用来refine。

### 3d bounding boxes for road vehicles: a one-stage, localization prioritized approach using single monocular images
看起来像是一篇水文，全文的main contribution就是预测了一个关键点，这个关键点指的是车辆底部的中心。然后做了个投影回归的约束。

### Orthographic feature transform for monocular 3d object detection
这篇是搞了两个transform操作，把image feature投影到了voxel再投影到bird-eye view，然后直接在最后的feature上用高斯图进行学习，loss是l1。最后用了一个nms得到结果。结果不高，车的ap只有3.

### Mono3D++：
这篇文章我觉得很solid的，首先related work写得很好，都可以用来参考。这篇文章是用pascal数据集构建了2d关键点，并且算了平均的3dshape，使用2d loss，2d 关键点，3d框和3dshape这些loss，通过重投影，利用了shape的先验信息。此外还学习了单目深度，貌似是通过一个patch来搞的，然后学了地平线的法线，论文的loss function看起来也合理，值得借鉴。现在没仔细看懂，有时间仔细看看。

### multi-level fusion based 3d object detection from monocular images
武大的。这篇主要贡献应该是先学了个单目depth，然后把depth和rgb结合提取特征，通过depth还生成了电云，也直接pooling提取特征，然后和RGB的特征融合。

### ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape
2019.4.10.慕尼黑工业。深度好文。本文先使用一个2d detection网路提取bbox，然后结合单目depth 提出一个可微的ROI lifting操作，搞出了3d bbox的八个点。pose用四元数q加上whl来表示，根据bbox的位置x，y和深度z得到了八个点，然后学这八个点就可以了。用了warm up。文章指出了以自我为中心和非自我为中心的不同之处，意思是先按非自我中心的来学习，学完了根据预测的position来纠正角度。然后还搞了学一个mesh的shape，通过一个autoencoder学习embedding，并看了看shape空间内feature意义。通过这个把原图投影到学到的mesh上，可以用原图给mesh上色，并通过这东西做数据增强，扩展车到不同的位置上，数据增强比较有用。

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

### Monocular 3d object detection for autonomous driving
2015,3dop同一个作者,这个就是把双目改成了单目.然后使用3d proposal,投影回2d,根据2d提特征,对proposal进行score并回归.这篇还是利用了地平面,shape,分割,等信息,follow 3dop.

### Are cars just 3d boxes? - jointly estimating the 3d shape of multiple objects
2012?,苏黎世理工.这篇是用线条组成的car模型来做,相当于利用了shape先验,先学coarse的3d box再refine,相比bbox描绘更精准,bird's eye view也更准了.主要参考[40]的一个方法,也用了cad model学shape,还学了一个遮挡mask,用来表示object之间的遮挡信息.文章通过概率模型来model,使用的多类别随机森林.

### Data-driven 3d voxel patterns for object category recognition
2015, 3dvp.这篇和上篇一样也是利用3d先验,不同的是使用3dvoxel,把原图和cad model对应起来,然后渲染3d投影到2d,根据前后顺序确定遮挡,然后搞一个2d遮挡mask,再搞3d遮挡voxel,然后学这个东西.测试的时候也是先2d detection,再出带有遮挡的segmentation mask,再出3d location.

### subcategory-aware convolutional neural networks for object proposals and detection
2016, 把3dvp那套搬到了fast rcnn上,说是利用subcategory,我感觉就是多任务的思想,rpn和最后分类里加了一个3dvp的任务,用这些任务来提取heatmap,结合起来用于更好的roi,这些任务在最后的feature也会用到,这样更好提取3d信息.

### Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Automonous Driving
ICCV2019. 欧阳万里。这篇是把depth转成了点云，然后把RGB值投到了电云上，这样每个点有三维位置和三维RGB，然后根据2d的ROI根据这部分点使用pointNet来学习3d detection。作者还用了一步背景分割，根据车的平均depth把更远的当做背景，这块设了一个阈值。最后结果很高很高。这篇说提升比较大的部分是segmentation，还要多加dropout防止过拟合。然后region branch实际上是没什么用的，RGB branch用处也没特别大。回归的时候分了两步，第一步是先算了中心点，应该是和Fpointnet是一样的？然后align之后再算了第二步的两个loss，回归RT和八个点。单目DORN最好，双目pointnet++好像比Fpointnet还好一点。

### monocular 3d object detection with pseudo-lidar point cloud
2019,CMU的。这篇和pseudo lidar那篇有异曲同工之妙，和洪伟的paper也有点像。是提取了depth，然后根据depth生成pesudo lidar，然后根据instance segmentation从中抠出视锥来，根据视锥学习一个3d detection，这里用了二阶段，还学了一个残差，最后搞了一个2d-3d consistency，这篇文章一看流程图就懂了。然后这篇文章重点参考了frustum pointnets这篇文章，还去掉了里面回归中心坐标的网络。另外在inference之后搞了个后处理，还是consistency来修改3d pose，随着深度的越大，3d参数可被更改的就越多，bound。

### M3D-RPN: Monocular 3D Region Proposal Network for Object Detection
单目的paper，目前仅次于欧阳万里学生那一篇。这篇是想用一阶段的办法，不需要额外的数据和额外的模型来训练单目3d detection，同时提取2d和3d的proposal。所以这篇文章的思想就是使用2d-3d rpn，定义了一些anchor，直接在这些anchor上回归2d-3d bbox。然后为了应对不同depth上的差别，提出了一个depth-aware的conv，意思就是将图片从上到下分成好多个部分，每个部分使用不同的卷积核，感觉还挺有效的，双边滤波器这里可以考虑一波了。然后作者最后通过2d-3d consistency对3d的角度进行了优化， 一般迭代8次会收敛。

## 双目

### 3D object proposals using stereo imagery for accurate object class detection
3DOP,pami17年的,原文是3d object proposals for accurate object class detection, nips15。这篇文章是想要尽量多利用信息，context信息，freespace等等。因此proposal之后还外扩一圈提取信息和proposal的信息结合。这篇文章是调研了很多传统方法，使用双目或者lidar（voxel）作为输入，尽量多的利用了特征，搞了个能量函数，提取3d proposal。和目前最新的方法都不是一个套路了。

### Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving
cvpr2019,这篇争议最大的点是，方法都是别人的，它真的只是“bridging”了一下。首先用stereo的方法得到了双目深度，然后用公式转成伪点云，再用lidar detection的方法跑个检测，走通了双目3d检测的一条新的pipeline。

### Pseudo-Lidar++
2019，康奈尔的。这篇延续了pseudo lidar的套路。主要有两个创新点，第一是说通过disparity来估计depth，越远处偏差越大，这是系统误差不是随机误差，因此要直接估计depth，在PSMnet基础上改进的。第二是说做了一个depth的优化，通过便宜的4线lidar的稀疏点云，保留生成的pseudo-lidar的shape，通过knn的方式把一个点拉过来，周围的也跟着过来，大概是这种思想，得到了更好的depth。后面还是相同的套路了。


### Stereo R-CNN based 3D Object detection for autonomous Driving
cvpr2019.参考文献写得不错。双目搞了两个ROI，gt是取的左右框的交集，然后左右的特征进行concat学习3d box部分参数，后面也学了一下特征点（四个关键点，都在底边）。之后用了3d box estimator通过关键点和3d参数来解出position，然后是3d box alignment，是一个重投影误差用来求解最佳中心depth。


## lidar

### PointNet++
非常solid啊。是把所有的点划分成有overlap的regions，然后提取局部特征。（类似CNN的share weights的思想。）采用了最远距离法选了一些中心点，然后尽量选了大范围的neighborhood（N个点，K个neighbor），选点提特征用mlp和pointnet一样。相当于用了层次的pointnet，每次都sampling & grouping，最后用pointnet+mlp分类或者，咋来的咋回去，从少的点插值回多的点，用来做分割。使用了multi-resolution grouping。

### Frustum pointnets
一张图经过2d detection，模型是固定的，是在coco上pretrain后到kitti上finetune的模型。把视锥取出来（需要用内参和depth），然后根据其类别对其中的点做了一个segmentation,使用point。然后取出剩下的点进行3d回归，其中使用了一个Tnet来回归中心点，其他的相当于学习了一个残差，这个东西为啥居然会很有用呢。loss上也加了八个角点回归的loss。

### Frustum ConvNet
和上一篇一样的思路，还是取了一个视锥出来，思想是视锥不同深度切块，都用pointnet来学了一个特征，然后把这些特征concat起来再用Fully Convolution Network和detect header进行回归。

### Fast and Furious: Real Time End-to-end 3d objection, tracking and motion forecasting with a single convolutional net
雷达点云,多帧结合,统一坐标系,使用高度作为feature提取voxel.多帧预测bbox然后根据score和iou进行平均融合.提出前融合后融合这两种常见的融合方案.

### OBject detection and classification in occupancy grid maps using deep convolutional networks
2018,将点云数据转换成occupancy grid maps，其实就是多种俯视图的特征，可以看论文上的图，然后将多种俯视图特征利用2d detector来检测，在kitti bird‘s eye view上测试。

### Capturing Object Detection Uncertainty in Multi-Layer Grid Maps
2019.1,这篇和上面那篇是一个套路的，搞了grid maps，此外说学习uncertainty是有用的。因此学了一个（角度上的？）uncertainty。（参考另一篇论文）通过最大和最小角度可以可视化出一个95%概率的3d壳，后面有些讨论。感觉这两篇都是水文。

### PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
少帅的paper，只用了lidar，先做了bin-based 3d box generation再refine，看图二。图一是和AVOD的对比，挺好。首先对于每个点过了point cloud autoencoder。然后bin-based box generation，并进行了前景的分割。然后把box取出来，前景的点利用其semantic information和其spatial information（转换到正则坐标下），concat起来经过box refinement和confidence prediction。


## lidar+ rgb

### multi-view 3d object detection network for autonomous driving
百度的一篇很强很大的工作.使用lidar+rgb.这篇是利用了faster-rcnn的思想,把2d的proposal改成了3d的proposal,然后投影到三个面做roi pooling,一个是lidar bv, 一个是lidar front, 一个是image.这三个分别使用不同的conv层提取特征,提取之后三个roi pooling,然后将三个特征进行了deep fusion最后进行分类回归,从而完成3d det任务.

### Joint 3D proposal generation and object detection from view aggregation
AVOD。用了3d anchor grid同时在原图和lidar俯视图上提特征并fusion回归box。fusion后rpn然后经过nms得到top k 3d proposal，再去前和俯视图中提取出featuremap再fusion，然后经过全连接再NMS得到最后的结果。没仔细看，但是不错。

### A General Pipeline for 3D Detection of Vehicles
ICRA18年的文章。这篇文章是同时利用了image和lidar的信息。contribution是说自己可以利用2d detection的框架，把RPN之前的部分fix住，只训练后面的部分，然后多回归一个dimension，同时把lidar点投影到图像上，把框里的部分抠出来放大一点。然后貌似是和三类车的voxel model做了对比提出了3d的proposal，根据这个又进行了一步refine。

### Deep Continuous Fusion for Multi-Sensor 3D Object Detection
同时用了lidar和rgb，把他们的特征都投到了俯视图上。具体来说，是俯视图上的每一个点，先找和它最近的一些lidar点，找到了之后把这些lidar点投到RGB平面上，然后可以把RGB平面上这些点的feature拿出来，和他们的3d位置一起使用mlp来作为俯视图的特征。所以图片每经过几个resblock就可以提取特征转到俯视图，和lidar投的俯视图特征结合（还有一个BEV net），最后俯视图上搞一个detection header FPN。训练时使用了dataaug，对图片和lidar都进行aug，并调整矩阵使其对齐。

### Multi-Task Multi-Sensor Fusion for 3D object Detection
上一篇工作的扩展，使用了multi-task，感觉coding挺厉害的。除了23D detection，还加了depth completion和mapping的任务。lidar online mapping估计地平面，然后投影出BEV图。利用BEV图和RGB image使用上文的方法进行fusion并第一阶段估计了一些3dbox。同时rgb做了depth completion并生成了plidar，这个也被用到了fusion里面。然后又第二阶段，同时利用rgb和fusion之后的feature提取roi并concat到一起，做一个2d和3d的refinement。我的方法可以借鉴他这个第二阶段啊，plidar的俯视图feature不知道如何。
