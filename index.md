# Abstract

<img src="./assets/gifs/kiara.gif" width="350" title="Kiara" alt="Qualitative result for the Bunny model in an in-the-wild panorama." align="right"/>

Estimating a scene’s lighting is a very important task when compositing synthetic content within real environments, with applications in  mixed reality and post-production.
In this work we present a data-driven model that estimates an HDR lighting environment map from a single LDR monocular spherical panorama.
In addition to being a challenging and ill-posed problem, the lighting estimation task also suffers from a lack of facile illumination ground truth data,  a fact that hinders the applicability of data-driven methods.
We approach this problem differently, exploiting  the  availability  of  surface  geometry  to  **employ image-based relighting as a data generator and supervision mechanism**.
This relies on a global Lambertian assumption that helps us overcome issues related to pre-baked lighting.
We relight our training data and complement the model’s supervision with a photometric loss, enabled by a **differentiable image-based relighting technique**.
Finally, since we predict spherical spectral coefficients, we show that by imposing a **distribution prior on the predicted coefficients**, we can greatly boost performance

<p align="center">
<iframe width="720" height="480" src="https://www.youtube.com/embed/M7c69qxVzXY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
___

# Overview
We use the uncoupled limited Laval HDR lighting dataset (\[[1](#Laval)\]) and the larger [3D60](https://vcl3d.github.io/3D60) color and normal dataset (\[[2](#HyperSphere)\]) jointly, coupling them through relighting in order to learn a single-shot HDR lighting estimator from a single LDR spherical panorama. 

<img src="./assets/images/Introduction.png" width="850" title="Overall Concept" alt="Our concept couples uncoupled datasets for learning the lighting estimation task."         align="center"/>

# Relighting-based Supervision
![Supervision](./assets/gifs/Training_scheme.gif)

# Results
Qualitative results for virtual object rendering in real scenes with the lighting estimated by our model.

<img src="./assets/images/hotel_room.jpg" width="49%" title="Hotel Room Panorama"/>
<img src="./assets/gifs/hotel.gif" width="49%" title="Hotel Room"/>

<img src="./assets/images/wooden_lounge.jpg" width="49%" title="Wooden Lounge Panorama"/>
<img src="./assets/gifs/wooden_gif.gif" width="49%" title="Wooden Lounge"/>

<img src="./assets/images/anniv_lounge.jpg" width="49%" title="Anniv Lounge Panorama"/>
<img src="./assets/gifs/anniv270.gif" width="49%" title="Anniv Lounge"/>
<img src="./assets/images/anniv_lounge2.jpg" width="49%" title="Anniv Lounge Panorama"/>
<img src="./assets/gifs/anniv70.gif" width="49%" title="Anniv Lounge"/>

<img src="./assets/images/colorful_studio.jpg" width="49%" title="Colorful Panorama"/>
<img src="./assets/gifs/colorful.gif" width="49%" title="Colorful"/>

<img src="./assets/images/lythwood_lounge.jpg" width="49%" title="Lythwood Lounge Panorama"/>
<img src="./assets/gifs/lythwood.gif" width="49%" title="Lythwood Lounge"/>

<!--

![wooden_lounge_panorama](./assets/images/wooden_lounge.jpg)
![wooden_lounge](./assets/gifs/wooden_gif.gif) 

![anniv_lounge_panorama](./assets/images/anniv_lounge.jpg)
![anniv270](./assets/gifs/anniv270.gif)
![anniv90](./assets/gifs/anniv70.gif)

![colorful_panorama](./assets/images/colorful_studio.jpg)
![colorful](./assets/gifs/colorful.gif)

-->

> Images are in-the-wild samples from [HDRiHaven](https://hdrihaven.com/). Three materials are used: a conductor(reflecting mirror), rough plastic and another conductor(gold). On each row, the leftmost image is the panorama, while the rightmost show perspective render (viewport denoted within the panorama). 

# Publication
<a href="https://arxiv.org/abs/2005.08000"><img src="./assets/images/paper_image.png" width="900" title="arXiv paper link" alt="arXiv"/></a>

## Authors
[Vasilis Gkitsas](https://github.com/VasilisGks) __\*__, [Nikolaos](https://github.com/zokin) [Zioulis](https://github.com/zuru) __\*__, [Federico Alvarez](https://www.researchgate.net/profile/Federico_Alvarez3), [Dimitrios Zarpalas](https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html), and [Petros Daras](https://www.iti.gr/iti/people/Petros_Daras.html)

## Citation
If you use this code and/or data, please cite the following:
```
@inproceedings{gkitsas2020deep,
  title={Deep lighting environment map estimation from spherical panoramas},
  author={Gkitsas, Vasileios and Zioulis, Nikolaos and Alvarez, Federico and Zarpalas, Dimitrios and Daras, Petros},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={640--641},
  year={2020}
}
```


# Acknowledgements
This project has received funding from the European Union’s Horizon 2020 innovation programme [Hyper360](https://hyper360.eu/) under grant agreement No 761934.

 We would like to thank NVIDIA for supporting our research with GPU donations through the NVIDIA GPU Grant Program.

![eu](./assets/images/eu.png){:width="150px"} ![h360](./assets/images/h360.png){:width="150px"} ![nvidia](./assets/images/nvidia.jpg){:width="150px"}

# Contact
Please direct any questions related to the code & models to gkitsasv “at” iti “dot” gr or post an issue to the code [repo](https://github.com/VCL3D/DeepPanoramaLighting).

# References
<a name="Laval"/>__\[1\]__ Gardner, M. A., Sunkavalli, K., Yumer, E., Shen, X., Gambaretto, E., Gagné, C., & Lalonde, J. F. (2017). [Learning to predict indoor illumination from a single image.](https://arxiv.org/pdf/1704.00090.pdf) ACM Transactions on Graphics (TOG), 36(6), 1-14.

<a name="HyperSphere"/>__\[[2](https://vcl3d.github.io/HyperSphereSurfaceRegression/)\]__ Karakottas, A., Zioulis, N., Samaras, S., Ataloglou, D., Gkitsas, V., Zarpalas, D., and Daras, P. (2019). [360<sup>o</sup> Surface Regression with a Hyper-sphere Loss](https://arxiv.org/pdf/1909.07043.pdf). In Proceedings of the International Conference on 3D Vision (3DV).

