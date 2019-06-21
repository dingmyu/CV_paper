## 3D multi-view oral paper:

### SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration Without Correspondences
> 是对RANSAC的一种改进方法，每次多取一些samples进行选取inliners。

### BAD SLAM: Bundle Adjusted Direct RGB-D SLAM
> 是用BA来做SLAM的一种方案。

### Revealing Scenes by Inverting Structure From Motion Reconstructions
> 稀疏点云 + RGB 重建真实场景，可以渲染出真实的frame 

### Strand-Accurate Multi-View Hair Capture
> micro-scale, 3D hair重建, 使用了一种新型的line-based的multi-view stereo

### DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
> 回归SDF，相比点云，可以得到continuous surface。用autoencoder学到了一个shape隐空间，用稀疏点来还原shape

### Pushing the Boundaries of View Extrapolation With Multiplane Images
> 多真渲染一个连续场景，用了MPI Fourier transformation。

### GA-Net: Guided Aggregation Net for End-To-End Stereo Matching
> Matching cost aggregation.把SGM变成了SGA layer，可微并且把min变成了max

### Real-Time Self-Adaptive Deep Stereo
code，refine，金字塔特征，backward只走一条路。

### LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation
Matching cost + Disparity + RGB -> confidence, 3 attention.

### NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences
local information, Feature aggregation module

### Coordinate-Free Carlsson-Weinshall Duality and Relative Multi-View Geometry
Linear internal constraints...没看懂

### Deep Reinforcement Learning of Volume-Guided Progressive View Inpainting for 3D point Scene Completion From a Single Depth Image
3D scene completion. Depth inpainting, 强化学习决定projection角度，完成3D重建。


## Scenes & Representation

### d-SNE: Domain Adaptation Using Stochastic Neighborhood Embedding
>

### Taking a Closer Look at Domain Shift: Category-Level Adversaries for Semantics Consistent Domain Adaptation
>

### ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation
> 对分割特征图进行minimize entropy，还加了GAN的loss进行训练。

### ContextDesc: Local Descriptor Augmentation With Cross-Modality Context
> 全局提取网络训练的特征描述符缺少local信息，因此选择加入keypoint信息，rgb content信息和keypoint geometry信息结合。

### Large-Scale Long-Tailed Recognition in an Open World
> 提出结合类别样本数目不均衡问题，结合分类，few-shot和noval class，训练了一个网络，bottom-up attention再top-down attention，记住原来的同时学习新的。attention in attention的思路。

### AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations Rather Than Data
> 是用auto-encoder对旋转平移等信息进行重建，给定一个图像和随机旋转评平移，通过重建无监督的学习更好的feature。

### SDC – Stacked Dilated Convolution: A Unified Descriptor Network for Dense Matching Tasks
> 使用dilated conv来学dense matching。kernel: [5,5,5,5], dilated rate: [1,2,3,4], 感受野为17，多个这样的module堆叠在一起。

### Learning Correspondence From the Cycle-Consistency of Time
> xiaolong wang的通过patch的cycle consistency学习dense matching的文章。

### AE2-Nets: Autoencoder in Autoencoder Networks
> 这篇做的是多任务？每个任务用autoencoder重建之后，为了证明隐变量是学到了相应的东西，再使用一个autoencoder对隐变量进行重建。

### Mitigating Information Leakage in Image Representations: A Maximum Entropy Approach
> 这篇是说把task无关的东西干掉，比如说superclass的分类不希望小类对其有影响。因此提出了Maximum Entropy，比直接最小化likelyhood要好。

### Learning Spatial Common Sense With Geometry-Aware Recurrent Networks
> 使用GRNN进行3D重建，这里说的主要就是传统的RNN连接的多个frame位置对不上，因此先基于geometry把2D都投影到3D空间，然后再连RNN就可以了

### Structured Knowledge Distillation for Semantic Segmentation
> 蒸馏网络，搞了pair-wise和pixel-wise的loss，认为需要蒸馏每个像素的概率。

### Scan2CAD: Learning CAD Model Alignment in RGB-D Scans
> release数据集，RGB和CAD model对于标注3D关键点.整体识别流程是寻找关键点，然后把某3dmodel关键点那一块丢进网络，会学到一个heat map出来，根据这个东西做align，最后基于ransac的方式得到9 dof的pose。

### Towards Scene Understanding: Unsupervised Monocular Depth Estimation With Semantic-Aware Representation
> 使用语义分割帮助优化单目depth，搞了一个左右眼的consistency，另外也可以帮助平滑。语义分割和depth网络共享一个主干网络，在一个位置concat不同的task identity会有不同的结果，这样就可以使用非pair的数据。

### Tell Me Where I Am: Object-Level Scene Context Prediction
> 使用部分的semantic segmentation map来生成完整的layout，比如给一个人，那么学习一个shape和一个bbox的proposal，可以是一个天空，把这个贴到bbox的部位就可以了，监督信息就是完整的segmentation map。搞了一堆gan loss

### Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
> 搞了一个NOCSmap，使用颜色来表示？没太看懂

### Supervised Fitting of Geometric Primitives to 3D Point Clouds
> 3D物体重建，每一个物体分解成了好多个类型的面，把它们融合起来学习，没太看懂。

### Do Better ImageNet Models Transfer Better?
> 做了好多好多实验，探究不同网络不同task，用不用imagenet预训练模型的效果差异。

## 3D Single View & RGBD

### 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans
用多帧的RGBD数据来解3d instance segmentation问题，同时也可以求解3d detection问题。主要是使用了多帧的geometry和color约束。
### Causes and Corrections for Bimodal Multi-Path Scanning With Structured Light
没看懂，选一个点A为前景，B为背景，AB为边界。然后根据频率...multi-path来求解。
### TextureNet: Consistent Local Parametrizations for Learning From High-Resolution Signals on Meshes
3. 提出了4-Rosy Field （quadriflow）来表示3D空间，大概是用角度来表示每一个点？最后做了3D Segmentation。
### PlaneRCNN: 3D Plane Detection and Reconstruction From a Single Image
4. 3d平面检测，重建3dmodel。解决了传统方法不鲁棒没有意义，深度学习的方法有数量限制的问题。使用mask-rcnn来检测一个区域，再计算depth的一个offset，然后得到一些mask，所有mask过网络进行refine。使用wrapping loss module，warp到相邻视角，然后约束一下。使用室内数据集训练可以扩展到室外数据集。
### Occupancy Networks: Learning 3D Reconstruction in Function Space
5. 在function space进行3d重建，3d重建的表示方法有：voxels，point cloud，mesh。本篇文章想把mesh表示成深度模型的连续决策边界。使用matching cubes算法。可以从rgb或点云或者进行3dmodel的超分辨率。作者使用了autoencoder进行重建，并对隐变量对应进行可视化，结果不错。
### 3D Shape Reconstruction From Images in the Frequency Domain	
6. 通过神经网络学习很多channel的表示，投影到3d fourior空间，然后再投影回来。说只需要很少的slices就可以表示3d shape...没看懂
### SiCloPe: Silhouette-Based Clothed People
7. 对穿衣服的人进行重建，voxel-based不能精确表示，本篇是silhouette-based的表示。提取3dpose投影到多个view，每个view使用一个mask来表示，使用GANloss来训练了一个2d silhouette的生成。又使用gan训练了一个不同view的rgb texture生成。
### Detailed Human Shape Estimation From a Single Image by Hierarchical Mesh Deformation
8. 人的重建可以分为两类，一种是参数模型，；另一种是非参的voxel表示。本文是经过了三步，每一步都有监督，最后生成了final mesh。
### Convolutional Mesh Regression for Single-Image Human Shape Reconstruction
9. 使用GCN对shape表示进行回归，有参数和非参数两种方案。
### H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions
10. 一个模型做了hand pose检测（3d hand skeleton），6dof物体pose检测，action识别等任务。多个RGB帧使用RNN连接起来，每一帧都做skeleton和3d bbox检测。
### Learning the Depths of Moving People by Watching Frozen People
11. release了一个数据集，单帧图像得到了一个depth，然后连续两帧使用光流，然后数据集里有静止人的pose和sfm，用这些做一个loss。最后做到了把一个动态的人插到一个场景里，还可以在场景里把人去掉。
### Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion
12. 传统方法提取特征点，然后匹配计算相对pose，用来解决小的或者没有重合的时候计算相对pose。首先通过一个transform和场景的complete，然后再计算相对pose。同时利用了color，normal和depth等来学习，感觉是有点悬学的。
### A Skeleton-Bridged Deep Learning Approach for Generating Meshes of Complex Topologies From Single RGB Images
13. 使用skeleton（curve和sheet）。image-> skeletal points-> skeleton volume-> base mesh-> final surface.
### Learning Structure-And-Motion-Aware Rolling Shutter Correction
14. rolling shutter cameras.相机可能导致拍出来电线杆歪，本文把这个扭过来。本文利用RS geometry，一路depth，一路velocity，使用了inverse depth loss，photometric和velocity loss，学习一个wrapping flow，修改后拍出来的图很正。和sfm还有点关系，没看懂。
### PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation
15. 学习object pose，是基于keypoint的方法，求解一个PnP问题。怎样找到robust的2d-3d correspondences。两个问题，occlusion和truncation，会造成invisible key point。本文使用vector-field来表示keypoint，就是方向vector，通过这个东西可以做dense voting。使用了uncertainty PnP。


## Recognition

### Panoptic Feature Pyramid Networks
1. 把带FPN的Mask-RCNN和普通分割结合在一起，共享FPN的feature map，提出Panoptic FPN。semantic FPN没有使用dilated conv。straightforward和efficient baseline，是以后方法的lower bound。
### Mask Scoring R-CNN
2. 传统使用分类score当作mask score，但是这俩关系不大。所以重新预测了mask score，并且refine。这就提升了一到两个点。
### Reasoning-RCNN: Unifying Adaptive Global Reasoning Into Large-Scale Object Detection
3. 搞了一个classifier weights，然后把这些类别放到了knowledge graph里，然后使用GCN，然后学到了更好的proposals。
### Cross-Modality Personalization for Retrieval
4. 现有的retrieve方法只使用了pixel像素，没有感情。放了一个数据集，在不同的位置看的图片描述也是不同的。同一张图的两种描述要尽量近（一正一反），不同图则要远。
### Composing Text and Image for Image Retrieval - an Empirical Odyssey
5. 同时使用文本和图片进行retrieval。baseline是直接使用feature fusion或者caption，vqa的架构。这篇文章使用lstm提取文本特征，并且把文本特征作为一个residual，使用gate function改变visual feature在空间中的位置。
### Arbitrary Shape Scene Text Detection With Adaptive Text Region Representation
6. 任意形状文字检测。大部分现有方法使用polygons，并且是固定数量的keypoint。本文使用配对的pairwise boundary points。adaptive的keypoints数量。使用proposal的roi特征，配合lstm，每次输出四个值，即配对的两个点坐标。
### Adaptive NMS: Refining Pedestrian Detection in a Crowd
7. Adaptive NMS。greedy nms和soft nms不能保证，crowd region的higher score，这是本文的用意。所以对于拥挤的位置就可以使用更高的threshold，反之也成立。所以本文额外预测了一个拥挤程度，并用这个东西进行adaptive的学习。
### Point in, Box Out: Beyond Counting Persons in Crowds
8. 数人头有两种方式。第一种是regression-based，使用高斯核来进行回归，另一种就是基于检测的方案。这篇文章使用pseudo gt初始化然后不断更新。惩罚了bbox大小不在正常范围内的。
### Locating Objects Without Bounding Boxes
9. 不用bbox定位。本文认为对于小的目标，用点定位更方便，相比bbox。本文设计了一个loss，对于gt point set和预测的point set做了个距离损失。然后往上加了个权重，把minmax去掉了，这样整个网络是可微的。
### FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery
10. 去掉了背景的影响，可以生成真实的精细类别图片。印度人完全听不懂啊。
### Mutual Learning of Complementary Networks via Residual Correction for Improving Semi-Supervised Classification
11. 好多公式，提出了一个CCN...没听懂。uncertainty什么鬼的。
### Sampling Techniques for Large-Scale Object Detection From Sparsely Annotated Objects
12. open images数据集，sparse annotation。如果一个proposal在bbox里，把这个loss忽略掉，用于part detection。经过统计93%这种都在图像框里。
### Curls & Whey: Boosting Black-Box Adversarial Attacks
13. 现有攻击方法有一步基于梯度的攻击，也有迭代的方法。本文是先假装梯度下降一步？然后计算cross-entropy，然后再反传更新图像。
### Barrage of Random Transforms for Adversarially Robust Defense
14. 使用random transform来抵抗攻击，BaRT。所以这篇文章就是搞了大量的transform，然后随机sample一些按照一定顺序来应用，相当于data argumentation？
### Aggregation Cross-Entropy for Sequence Recognition
15. 提出了ACE loss。解决了梯度消失的问题？
### LaSO: Label-Set Operations Networks for Multi-Label Few-Shot Learning
16. 一张图像里可能有好多objects，会影响分类，对这些东西取了交集并集差集，每个都去做了分类loss，然后又搞了一个重建loss，即减掉再加上？
### Few-Shot Learning With Localization in Realistic Settings
17. 好像是把数据集扩展了？使用batch folding训练。
### AdaGraph: Unifying Predictive and Continuous Domain Adaptation Through Graphs
18. 用raw view作为target metadata，根据这个训练meta的model，可以迁移到不同的角度分类。然后把这些东西搞到了一个graph里。propagate parameters，然后学习一个domain specific的params？然后用这个分类。


## Segmentation & Grouping

### UPSNet: A Unified Panoptic Segmentation Network
###	JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds With Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields
### Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth
### DeepCO3: Deep Instance Co-Segmentation by Co-Peak Search and Co-Saliency Detection
### Improving Semantic Segmentation via Video Propagation and Label Relaxation
### Accel: A Corrective Fusion Network for Efficient Semantic Segmentation on Video
### Shape2Motion: Joint Analysis of Motion Parts and Attributes From 3D Shapes
### Semantic Correlation Promoted Shape-Variant Context for Segmentation
### Relation-Shape Convolutional Neural Network for Point Cloud Analysis
### Enhancing Diversity of Defocus Blur Detectors via Cross-Ensemble Network
### BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames
### Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images
### Efficient Parameter-Free Clustering Using First Neighbor Relations
### Learning Personalized Modular Network Guided by Structured Knowledge
### A Generative Appearance Model for End-To-End Video Object Segmentation

## Deep Learning

### Practical Full Resolution Learned Lossless Image Compression
### Image-To-Image Translation via Group-Wise Deep Whitening-And-Coloring Transformation
### Max-Sliced Wasserstein Distance and Its Use for GANs
### Meta-Learning With Differentiable Convex Optimization
### RePr: Improved Training of Convolutional Filters
### Tangent-Normal Adversarial Regularization for Semi-Supervised Learning
### Auto-Encoding Scene Graphs for Image Captioning
### Fast, Diverse and Accurate Image Captioning Guided by Part-Of-Speech
### Attention Branch Network: Learning of Attention Mechanism for Visual Explanation
### Cascaded Projection: End-To-End Network Compression and Acceleration
### DeepCaps: Going Deeper With Capsule Networks
### FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
### APDrawingGAN: Generating Artistic Portrait Drawings From Face Photos With Hierarchical GANs
### Constrained Generative Adversarial Networks for Interactive Image Generation
### WarpGAN: Automatic Caricature Generation
### Explainability Methods for Graph Convolutional Neural Networks
### A Generative Adversarial Density Estimator
### SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates

