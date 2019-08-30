### Characterizing and Improving Stability in Neural Style Transfer
这篇文章就是解决style transfer连续帧之间不稳定的一个问题。用了光流warp。主要两点：一是warp完的第一帧transfer结果加到第二帧输入里；二是使用光流，搞了一个对人的mask，然后temporal consistency loss。
