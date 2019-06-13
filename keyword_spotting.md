### Efficient keyword spotting using dilated convolutions and gating
> 在wavenet的基础上，加了跳跃连接和空洞卷积。提出了一种数据标注的方法，就是检测end-of-the-word，检测到了之后前后各取一个delta T作为该词进行标注，训练的时候把不是这个词的mask掉。

### Convolutional
> Google的文章，可能是第一篇用spectrogram和CNN来做语音分类的文章。
