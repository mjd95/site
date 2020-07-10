---
title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
date: 2020-07-02T09:23:35+01:00
draft: true
---

Around the mid-2010s, it seems there were a few papers on a topic called R-CNNs, geared towards object detection problems. The basic setup is that you start with some "region proposals" (parts of the image which you think there might be an object) and then apply some CNN in the locality. For many applications you want the whole pipeline to be fast, but it seems that now the bottleneck is the time taken to make the region proposals.

How slow are we talking? The author mentions a couple of methods: one which generates high quality proposals (Selective Search) but is extremely slow (2s on the CPU), another which trades off quality for speed (EdgeBoxes) but still takes 0.2s on the CPU. The author addresses the obvious question of why not speed these up by using the GPU (I guess the methods would lend themselves to it) given that speed of the next step will definitely rely on the GPU. Actually, the author's method _will_ use the GPU (so it's not an apples-to-apples comparison), but thinks a new method is justified because the region proposal algorithm can be adapted to use not only the GPU but the exact convolutions the downstream detection network is already computing. To this end, he introduces _region proposal networks_ (RPNs), which will have a small marginal cost for computing proposals (10ms per image).

Of course the RPN is going to have to do some other stuff than just reapplying the convolution maps. There are two additional convolutional layers: one that econdes each convolutional map position in to a short feature vector and a second that, at each convolutional map position, outputs an objectness score and some other stuff that I can't parse before having another coffee. The RPNs can be trained alongside the Fast R-CNNs by alternating between fine-tuning for the region proposal task and fine-tuning for object detection, while keeping the region proposals fixed.

The method is evaluated on PASCAL VOC, where it performs slightly better than Selective Search + Fast R-CNNs, but more importantly has an overall rate of 5 fps. (This is the same speed as the fastest region proposal method (EdgeBoxes) without the Fast R-CNN, but remember that region proposal was happening on the CPU and could presumably be sped up, perhaps significantly.)

## Region Proposal Networks

An RPN takes an image (of any size) as input and outputs a set of rectangular object proposals with an objectness score (likelihood of there being a box in that rectangular region). To generate region
