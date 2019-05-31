### DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion
> 融合了点云和RGB数据,相当于在posecnn上进行改进. https://www.jiqizhixin.com/articles/2019-01-17-16

### 6-dof object pose from semantic keypoints
> 使用单目2d图像,通过学习关键点而得到一个object的pose的方法.文章说如果有点云也能这样做,点云的话可以直接求解icp问题. 文章学习的关键点是有监督的,是通过channel-wise的高斯特征图来学习.两个encoder-decoder的结构 Stacked Hourglass Networks .学习完关键点之后可以求解pnp问题,但是这样不准.文章是把学到的关键点先匹配cad model,然后将匹配到的关键点再进行求pose
