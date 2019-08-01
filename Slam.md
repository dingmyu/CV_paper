### CubeSLAM: Monocular 3d object slam
shichao yang的paper,2019,深度好文.这篇文章是同时做了3d detection和slam两个任务.传统的slam是近乎静态的,这篇想要解决一个动态问题.首先指出每个3dbox都有三个消失点,通过消失点和一个顶点配合2d box就能解出3d的八个点的位置.
这篇文章不是采用回归预测的方式来预测vp点,而是使用sample加score的方式.然后这篇文章使用BA来对camera pose进行优化.这篇文章搞了几个loss和公式都很不错,值得借鉴.有重投影误差,方框误差,车上的点在3d框内等等.
然后这篇利用了前后帧的信息,远的检测不到,走近了说不定就可以了,加了对极约束和motion模型.有个地方说直接用svd可以解,后面也直接使用了一个tracking算法.可以仔细看下公式.
