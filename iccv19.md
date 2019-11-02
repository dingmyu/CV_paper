# Oral

## Oral1 Best paper

### low light, single photon camera.  (not in my field)

### object attributes and relations

GCN + object location + mask -> segmentation layout + appearance vectors -> Resmodule -> Image, GAN loss

(Mask feature matching loss, image feature matching loss, perceptual loss)

Can use object name and its position to generate images, and tune the hidden vector, the image will change.

### point-line minimal problems in complete multi-view visibility 

obtain 39 candidates by counting dimensions

remove that has little solutions, leave 30.

3d/2d dimension, how to calculate, how to use.

each minimal problem gives a system of polynomial equations, and solve a generic example and count of its solutions.

solve time (faster?, may future work)

### Singan: learning a generative model using a single image (Best!)

scale 1/8 or 1/16, mask part of image and generate, and upsample, and mask and generate... (multi-scale generator)

can change the dimension of the input, the output will change, preserve the texture.

can change the input image, do the same thing like cycleGan

can change the input style, cartoon or painting or skeleton.

can change the input image by mask (or add a small png in it), something like poison image editing.

Harmonisation, can change the local part of the add mask/png.

can use for super-resolution, becuz its multi-scale training. Gn -> Gn-1 -> G0 -> G0 (add a G0), cool!

Random walk in latent space can generate gif. (animation), can also change the intensity.

Code available.



## Oral2b multi-view geometry

### image query localization (privacy aware)

propose 3d line cloud map

hide query features

Image -> 2d point clouds -> 2d *random* lines (contribution: point to line)

### calibration

not sure a calibration is accurate

move camera to verify? choose the best next step? Guide user?

corner uncertainty in a novel manner

### Gated2Depth, gated camera

dataset

###  shape

Mask + crop + compose

fuse multi-frames

### event-based deep stereo

FIFO queue, CFC layer (indoor flying benchmark)

### point-based multi-view stereo network

Better, faster, smaller model size. point clouds can produce shape and complete 3D reconstruction.

### dynamic 3d reconstruction, discrete Laplace operator 

### non-rigid sfm

object category can be considerd as a non-rigid object, annotated imaged can be trained as rob;

### equivariant multi-view networks

combine CNN and equivalent 

```python
sleepy above
```

### 3d point cloud understanding

Previews: Voxel, multi-view projection, point processing

Interpolation, to unify them, use an norm term.

trilinear interpolation and gaussian interpolation are choices.

exp on modelnet40 and sharpener (classification) and S3DIS (segmentation)

maybe can used in 3d-det?

### revisiting point cloud classification

new dataset, scanobjectnn. 15 class, 2902 instances.

a benchmark on preivous methods.

how to handle partially, background.

seg branch is condition of class label.

### point cloud saliency maps

Extend this concept to point cloud, saliency score for each point, which point is the key for classification

dropping points: moving points to the point cloud center. saliency map based dropping

### shellconv, shellnet

neighborhood conv? extract feature from a local area (cycle of neighbor), then pooling.

conv points to less and less.

better and faster than pointnet++

### shape matching (nice)

previous, FMnet: supervised, point wise correspondence 

use an unsupervised loss

### quasi BNB

find local optimal 

### consensus maximization

nphard, ransac,

### quasi-globally, vanish point estimation

lines in calibrated image. or sampleing *ransac* or search *BNB*.

### two view relative pose

solve both the calibrated and semi-calibrated relative pose problem using homography with a common direction.

aligning the cameras with the gravity direction,

Rank 1 constraints.

### quasar,quaternion-based 

wahba problem, rotation search

### PLMP

all are math.

## Oral 3

### local descriptors 

image patch, triplet loss with margin.

static margin has problems.

1. static soft margin
2. dynamic hard margin
3. dynamic soft margin

hard negative mining

Hpatches benchmark

### Bayes-Factor-VAE

Unsupervised representation

Data fitting + disentangled factored + Discern relevant factors 

regularized loss

### multi-sampling for image transformation

Optimising to align images

Trans, rotaton, scale,

Bilinear sampling only backward four pixels

Generate several pixels. use forward-backward. using warping parameters for sampling.

### Adatransfrom

Increase data-varience for training while decrease for testing

can auto-zoom, result slight lower than auto-augmentation but 100x faster.

### CARAFE

pixel shuffle, limited kernel size

new upsampling method, Kernel prediction, norm, reassemble,

have large field of view, according the shape of object, not a box

### AFD-Net

Low-level features, compute their difference, aggregate block1,2,3,4.

### DJSRH: cross-modal hashing

joint semantics affinity matrix

### UNQ: Unsupervised neural quantization 

domain similarity search

### Siamese networks: the tale of two manifolds

Math, simple

FC with orthogonal columns

### PCA-GM: graph matching

Nphard,

image and key points -> cnn -> fconv (modified graph cn) -> cross conv (between two features)-> affinity metric -> sinkhorn -> match -> cross-entropy loss

code available

### fashion retrieval

global similarity use siamese network, is disturbed by occlusions, misalignments, etc.

Solution: multi-scale, each part of a clothing.

then build a graph for similarity reasoning.

classification loss. same/not



## Oral 10.31 morning

### Generative models by matching percptual features

Fixed discriminator

matching move avg, adam moving avg make it more stable, allowing small batchsize

MMD -> min/max optimisation; ours only min.

###Free-from image impainting with gated conv

user guidance 

naive conv, treat all pixels as valid ones, ill fitted. Solution: mask

benchmark: Places2, code available

### FiNet: fashion image inpainting 6

different shape, color, huge diversity.

Disentangle the appearance and shape. First, shape

use vae to encode diversity

Compatibility  encoder

then appearance, useing the shape/layout as guidance.

### InGan

single image, collection of batches. same patch distribution.

Input + transform -> G

generated image + inverse transform -> G to reconstruct.

can transform to any shape

Textures -> natural images.

### What a gan cannot generate

what is actually missing in GAN distribution

### coco-gan generation by parts

Crop and generate, never seen the whole image during training.

Test: Using multi-crops and concatenation. 16 * 16, 4 * 4 patch.

can extend image. Patch-guided generation

### neural turtle graphics for modeling city 

Node + edge, iterative process, capture unique style of cities. Demo is cool!

### Texture fields 11

gen textured 3d model

Voxels with color, no

Texture atlas, no

Texture field, yes. no discretization, no template.

learning a continuous 3d field, much better than nvs baseline

can also do singe-view construction

can change latent code and generate another, cool!

### point flow:3d point cloud generation with continuous 

neural ODE. point CNF.

first learn a shape latent space and use it as a guidance.

### meta-sim

MMD, reinforcement learning, test on the KITTI dataset.

### SinGAN

best paper, a girl present this work...😂 

Random samples.



## oral 10.31 afternoon

### IL

Combination of DC and IR.

### PR product

PR: projection and rejection, which is better than projection in cosine distance only.

### cutmix: robust image classification

Simple to implement, 20 lines in pytorch

Compare to cutout and mixup, more realistic and make full use of pixels.

### Interpretable object detection

Cool! Need to read carefully 

### transfering classification weights

### scale-aware trident nerworks for object detection

Scale-normlizaton.

Image pyramid is time consuming 

Feature pyramid; multi-head, multi dilation, share weights.

Multi-branch training and single-branch testing.

This paper is worth to read more,

### object-aware instance labeling for weakly supervised object detection

Previous, only label part and recognised other parts as background.

ignore little overlapping regions

### generative modeling for small data object detection

### Transductive for zero-shot object detection

### Self-training and adversarial backgrounders regularization

Enhancing FG/BG discrimination

### Memory-based

currently, ignored neighborhood information.

memory module and neighboorhood graph

### self-similarity grouping



## oral 11.1 morning

### RIO

dataset: 1482 scans, 48k object instances. NB

Method: triplet, multi-scale.

### Pixel2pose:

3d reconstuction loss, Transformer loss

Gan loss, for occlusion 

PNP problem

### CDPN

Disentangled trans rand rot.

rotation is different to train, first rotation then fix, then trans.

dynamic zoom in and what loss

### C3DPO: non-rigid sfm

use 2d keypoints as supervision.

predict camera matrix, get 3d shape. then reverse projection. Shape is weighted-avg of some basic shape.

for shape rotation: prevented with canonicalization

first deep method for nrsfm

### learning to reconstruct 3d manhattan wireframes from a single image

Use wireframe representation

a synthetic dataset, 24k images.

### Soft Rasterizer (cool)

differentiable rendering

can handle occlusion

## learnable triangulation for human pose

Algebraic 

3D CNN

2d confidence

Arbitrary views

### xR-EgoPose

Ego-centric estimation

synthetic dataset.

heatmap -> loss, pose loss.

### DeepHuman: 3d human construction from a single image

related work is good.

clothing layers

use VFT layer between 3d voxel net and 2d net.

### detailed human depth estimation

clothing wrinkles, body shapes.

Segmentation, skeleton, rgb -> 5 channels.

predict base shape, detail shape.

refine by normal.

### denserac

exsiting, direct regression

ours consider IUV image and dense pose,then 3d body estimation.

differencial rendering

large-scale syntheic dataset MOCA



## oral 11.1 afternoon

### explaining Neural networks semantically and quantitatively

Manual defined semantic concepts 

Prior loss. distillation, distill knowledge from pretrained model

### PANET: few shot segmentation

extend prototypical to segmentation

### shapemask: sement noval objects

Shape prior base, cluster and choose the centroid

Box - > shape prior -> coarse segmentation -> refine

### sequence level semantics aggregation for video object detection

optical flow is trival

each frame proposal, then aggregation

### video object segmentation using space-time (cool)

which frames.

Memory embedding

Memory encoder

### zero-shot segmentaion

graph, edge:relations

### meteornet

dynamic 3d point cloud sequence

Local patio-temporal neighbouring 

### 3d instance segmentation via multi-task learning 

Feature space (learned an embedding and mean shift)

extend previous 2d work to 3d

### deep gcns: can gcns go as deep as cnns? (Faust)

social network, point clouds and 3d meshes.

resGCNs...plainGCN + identity (residual block)

denseGCN (densenet)

dilated graph convolutions

point clouds are unsorted...

### deep hough voting for 3d detection

Input->pointnet++->seeds->votes->vote clusters->output

Votenet

no image information

### M3D-RPN

2d and 3d monocular detection

### semanticKITTI

dataset and benchmark?

semantic scene completion

### woodscape

dataset

### Scalable place recognition under appearance change for AD

faster and accurate than mapnet cause its temporal information

### exploring the limitations of behaviour cloning for AD

### habitat

dataset -> simulators -> tasks

for navigation

highly performant 3d simulation

learned vs classical navigation agents

# poster section

two threshold paralleled? 

point base net? point and image feature fusion (cuhk hongsheng xiaogang)?

Anchor-free 3d detector?
