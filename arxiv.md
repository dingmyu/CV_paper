### Speed estimation evaluation on the KITTI benchmark based on motion and monocular depth information
https://arxiv.org/pdf/1907.06989.pdf
通过video计算本车的速度，感觉是比较鸡肋的一个功能。这篇文章使用optical flow和单目depth模型来得到现成的flow depth，然后计算速度，误差RMSE 1m/s.

### Boxy vehicle detection in large images
投iccv19的大型3d/2d车辆检测数据集，没中。标了一个矩形背框加一个梯形侧边框

### ShapeNet: An Information-Rich 3D model repository
提出了一个CADmodel数据集,很丰富全面,除了model还标注了word part correspondence等信息.

### Vehicle speed estimation using a monocular camera
固定摄像头测速。通过检测车牌，估计车辆距离地面的高度，然后用相似三角形就可以测出距离。使用连续三帧进行测速。
