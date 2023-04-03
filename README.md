# SSL for Image Representation
This repository is ```SSL for Image Representation```, one of the OpenLab's PseudoLab.

[Introduce page](https://www.notion.so/chanrankim/SSL-for-Image-Representation-0574c45b4674428b94149c41cd724f30?pvs=4) <br/>
Every Monday at 10pm, [PseudoLab Discord](https://discord.gg/sDgnqYWA3G) Room YL!
## Contributor
- _Wongi Park_ | [Github](https://github.com/kalelpark) | [LinkedIn](https://www.linkedin.com/in/wongipark/) |
- _Jaehyeong Chun_ | [Github](https://github.com/jaehyeongchun) | [LinkedIn](https://www.linkedin.com/in/jaehyeong-chun-95971b161/) |

| idx |    Date    | Presenter | Review or Resource(Youtube) | Paper / Code |
|----:|:-----------|:----------|:-----------------|:------------ |
| 1   | 2023.03.20| _Wongi Park_    | OT | OT |
| 2   | 2023.03.27| _Wongi Park_    | [Youtube](https://youtu.be/7xUZA9X78x0) / [Resource](https://www.notion.so/chanrankim/SSL-for-Image-Representation-0574c45b4674428b94149c41cd724f30?pvs=4) | [Paper](https://arxiv.org/abs/2301.03580) / [CODE](https://github.com/keyu-tian/SparK)|
| 3   | 2023.04.03| _Jaehyeong Chun_    | [Resource](https://www.notion.so/chanrankim/SSL-for-Image-Representation-0574c45b4674428b94149c41cd724f30?pvs=4) | [Paper](https://arxiv.org/abs/2105.04906) / [CODE](https://github.com/facebookresearch/vicreg)|

## Table of Contents
- [Survey and Analysis](Survey-and-Analysis)
- [Contrastive Learninig](Contrastive-Learninig)
- [Masked Auto Encoder](Masked-Auto-Encoder)
- [Clustering](Clustering)
- [Blog and Resource](Blog-and-Resource)

### Survey and Analysis
- **[ Analysis ]** Revisiting self-supervised visual representation learning **(CVPR 2019)** [[Paper](https://arxiv.org/abs/2005.10243)] [[CODE](https://github.com/google/revisiting-self-supervised)]
- **[ Analysis ]** What Makes for Good Views for Contrastive Learning? **(NIPS 2020)** [[Paper](https://arxiv.org/abs/2005.10243)]
- **[ Analysis ]** A critical analysis of self-supervision, or what we can learn from a single image **(ICLR 2020)** [[Paper](https://openreview.net/forum?id=B1esx6EYvr)]
- **[ Analysis ]** How Well Do Self-Supervised Models Transfer? **(CVPR 2021)** [[Paper](https://arxiv.org/abs/2011.13377)]


### Contrastive Learninig
- **[ MoCo ]** Momentum Contrast for Unsupervised Visual Representation Learning **(CVPR 2019)** [[Paper](https://arxiv.org/abs/1911.05722)] [[CODE](https://github.com/facebookresearch/moco)]
- **[ MoCoV2 ]** Improved Baselines with Momentum Contrastive Learning **(ArXiv 2020)** [[Paper](https://arxiv.org/abs/1911.05722)] [[CODE](https://github.com/facebookresearch/moco)]
- **[ SimCLR ]** A Simple Framework for Contrastive Learning of Visual Representations **(ICML 2020)** [[Paper](https://arxiv.org/abs/2002.05709)] [[CODE](https://github.com/google-research/simclr)]
- **[ SimCLR v2 ]** Big Self-Supervised Models are Strong Semi-Supervised Learners **(NIPS 2020)** [[Paper](https://arxiv.org/abs/2006.10029)] [[CODE](https://github.com/google-research/simclr)]
- **[ SwAV ]** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments **(NIPS 2020)** [[Paper](https://arxiv.org/abs/2006.09882)] [[CODE](https://github.com/facebookresearch/swav)]
- **[ SimSiam ]** Exploring Simple Siamese Representation Learning. **(CVPR 2021)** [[Paper](https://arxiv.org/abs/2011.10566)] [[CODE](https://github.com/PatrickHua/SimSiam)]
- **[ BYOL ]** Bootstrap Your Own Latent A New Approach to Self-Supervised Learning **(NIPS 2020)** [[Paper](https://arxiv.org/abs/2006.07733)] [[CODE](https://github.com/deepmind/deepmind-research/tree/master/byol)]
- **[ RoCo ]** Robust Contrastive Learning Using Negative Samples with Diminished Semantics **(NIPS 2021)** [[Paper](https://arxiv.org/abs/2110.14189)] [[CODE](https://github.com/SongweiGe/Contrastive-Learning-with-Non-Semantic-Negatives)]
- **[ ImCo ]** Improving Contrastive Learning by Visualizing Feature Transformation **(ICCV 2021)** [[Paper](arxiv.org/abs/2108.02982)] [[CODE](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)]
- **[ Barlow Twins ]** Barlow Twins: Self-Supervised Learning via Redundancy Reduction **(ICML 2021)** [[Paper](https://arxiv.org/abs/2103.03230)] [[CODE](https://github.com/facebookresearch/barlowtwins)]
- **[ VICReg ]** VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning **(ICLR 2022)** [[Paper](https://arxiv.org/abs/2105.04906)] [[CODE](https://github.com/facebookresearch/vicreg)]


### Masked Auto Encoder
- **[ MAE ]** Masked Autoencoders Are Scalable Vision Learners **(CVPR 2020)** [[Paper](https//arxiv.org/abs/2111.06377)] [[CODE](https://github.com/facebookresearch/mae)]
- **[ SimMiM ]** SimMIM: A Simple Framework for Masked Image Modeling **(CVPR 2021)** [[Paper](https://arxiv.org/abs/2111.09886)] [[CODE](https://github.com/facebookresearch/mae)]
- **[ iBOT ]** iBOT 🤖: Image BERT Pre-Training with Online Tokenizer **(ICLR 2022)** [[Paper](arxiv.org/abs/2111.07832)] [[CODE](https://github.com/bytedance/ibot)]
- **[ BEiT ]** BEiT: BERT Pre-Training of Image Transformers **(ICLR 2022)** [[Paper](https://arxiv.org/abs/2106.08254)] [[CODE](https://github.com/microsoft/unilm/tree/master/beit)]
- **[ DMAE ]** Denoising Masked AutoEncoders Help Robust Classification **(ICLR 2023)** [[Paper](https://arxiv.org/abs/2210.06983)] [[CODE](https://github.com/quanlin-wu/dmae)]
- **[ SparK ]** Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling **(ICLR 2023)** [[Paper](https://arxiv.org/abs/2301.03580)] [[CODE](https://github.com/keyu-tian/SparK)]

### Clustering
- **[ RUC ]** Improving Unsupervised Image Clustering With Robust Learning **(CVPR 2021)** [[Paper](https://arxiv.org/abs/2012.11150)] [[CODE](https://github.com/deu30303/RUC)]
- **[ MICE ]** MiCE: Mixture of Contrastive Experts for Unsupervised Image Clustering **(ICLR 2021)** [[Paper](https://openreview.net/forum?id=gV3wdEOGy_V)] [[CODE](https://github.com/TsungWeiTsai/MiCE)]
- **[ GATCluster ]** GATCluster: Self-Supervised Gaussian-Attention
Network for Image Clustering **(ECCV 2020)** [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700732.pdf)] 
### Blog and Resource
