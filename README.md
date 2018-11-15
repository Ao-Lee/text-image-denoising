
#### Introduction
three models are used and assembled together in this implementation, there are:
* an auto encoder covnet is used to remove noise
* a traditional 2d median filter algorithm is used to remove very dark noise
* an unet is used as the 'assembling' network

#### notes
* all models are pixel-wise models, so there is no resize operation. In prediction, the target image is cropped into small blocks which fit the network input.

#### Results
left column shows the target images, ground truth images are in the middle, the right column shows the prediction of the model

![Alt text](https://github.com/Ao-Lee/text-image-denoising/raw/master/show/86.png)
![Alt text](https://github.com/Ao-Lee/text-image-denoising/raw/master/show/87.png)
![Alt text](https://github.com/Ao-Lee/text-image-denoising/raw/master/show/98.png)
![Alt text](https://github.com/Ao-Lee/text-image-denoising/raw/master/show/99.png)






