## 单目

### 3D bounding box estimation using deep learning and geometry
> 开山之作?是使用现成的2d detector, 把图里的车抠出来学它的旋转和维度(长宽高), 然后根据2d和3d的关系, 算出3d bbox的位置(location).精度不高但是数学可以借鉴.可以考虑加可学习的权重, 将loss反传回detection model.

### joint monocular 3d vehicle detection and tracking
用faster-rcnn检测,并根据上篇论文将检测出的2d框转换成3d.faster rcnn同时估计一个3d中心投影到2d图中的位置.根据前后帧产生的框来计算一个联系矩阵,判断哪些框是同一个车.最后根据这些信息refine了一下.所以是3d detection和tracking一起做.

### Deep MANTA: a coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image
提出使用关键点来表示车的3d结构.使用CAD模型?做了各种车辆类型和大小的模板.文章用了cascade的结构对rcnn结果进行了refine.同时学习2d bbox和part坐标,part的可见性(我感觉这里part指的就是那些关键点),以及属于那个模板.学会了之后套模板再进行2d-3dmatching.


## lidar

### Fast and Furious: Real Time End-to-end 3d objection, tracking and motion forecasting with a single convolutional net
雷达点云,多帧结合,统一坐标系,使用高度作为feature提取voxel.多帧预测bbox然后根据score和iou进行平均融合.提出前融合后融合这两种常见的融合方案.


## lidar+ rgb

### multi-view 3d object detection network for autonomous driving
百度的一篇很强很大的工作.使用lidar+rgb.这篇是利用了faster-rcnn的思想,把2d的proposal改成了3d的proposal,然后投影到三个面做roi pooling,一个是lidar bv, 一个是lidar front, 一个是image.这三个分别使用不同的conv层提取特征,提取之后三个roi pooling,然后将三个特征进行了deep fusion最后进行分类回归,从而完成3d det任务.
