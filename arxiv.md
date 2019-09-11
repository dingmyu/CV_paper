### Speed estimation evaluation on the KITTI benchmark based on motion and monocular depth information
https://arxiv.org/pdf/1907.06989.pdf
通过video计算本车的速度，感觉是比较鸡肋的一个功能。这篇文章使用optical flow和单目depth模型来得到现成的flow depth，然后计算速度，误差RMSE 1m/s.

### Boxy vehicle detection in large images
投iccv19的大型3d/2d车辆检测数据集，没中。标了一个矩形背框加一个梯形侧边框

### ShapeNet: An Information-Rich 3D model repository
提出了一个CADmodel数据集,很丰富全面,除了model还标注了word part correspondence等信息.

### Vehicle speed estimation using a monocular camera
固定摄像头测速。通过检测车牌，估计车辆距离地面的高度，然后用相似三角形就可以测出距离。使用连续三帧进行测速。

### new efficient solution to the absolute pose problem for camera with unknown focal length and redial distortion
这篇是一个优化问题，就像pnp问题一样。这篇是不知道焦距和径向畸变的情况下，分了平面和非平面的2d-3d correspondence来求解camera pose。

# making minimal solvers for absolute pose estimation compact and robust
iccv2017.解决的和上面的相同的问题，就是在上面那一篇的基础上进行的改进。

### a fast minimal solver for absolute camera pose with unknown focal length and radial distortion from four planar points
2018.这篇和上面一篇是同样的事情，不知道focal length和radial distortion的情况下通过平面上的匹配点来求解camera pose。

### monocular 3d object detection with fake 3d box for autonomous driving
据说是曾哥组用在senseauto中的算法，投了ECCV18，看起来没中。学习了12个关键点，然后生成了polygon，组成了一个fake 3d box。然后在这基础上根据投影关系找到了bird-eye view上的三个点，最后给一个人为规定的高度。。。。
