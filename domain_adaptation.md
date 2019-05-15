## unsupervised

### adversarial discriminative domain adaptation
> 基础的方法,对之前的方法总结出了一个通式.source分类,然后gan loss.是否share weight,选用什么loss,是生成还是判别等等.

### M-ADDA
> 在上一篇的基础上加上了metric learning.就是triplet loss.相当于对source做了一个聚类,然后让target和最近的一个source中心类接近.

### CyCADA
> 搭积木.在DA的基础上加上了cyclegan.还加了semantic的一致性等内容用于语义分割的迁移.实际上像是cyclegan生成的图再用da的loss训练分类.

### CDAN: Conditional Adversarial Domain Adaptation
> 在对抗性的DA基础上把特征和概率图都用了起来.feature和prob做了一个bilinear,因为维度过大所以选用了random layer. CDAN+E貌似就是所有的loss用了一个自适应权重联合学习.
