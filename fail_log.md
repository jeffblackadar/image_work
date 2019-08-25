
Gitbub pages
https://bookdown.org/yihui/rmarkdown/rmarkdown-site.html

yaml validator
https://codebeautify.org/yaml-validator

Binder


https://gke.mybinder.org/


https://github.com/binder-examples/r

https://hub.gke.mybinder.org/user/shawngraham-bin-with-tensorflow-php6a9xn/tree

https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb



# OCRR

https://cran.r-project.org/web/packages/tesseract/vignettes/intro.html

# Arch-I-Scan
Exploring Automated Pottery Identification [Arch-I-Scan]
Ivan Tyukin*ORCID identifier1, Konstantin Sofeikov1, Jeremy Levesley1, Alexander N. Gorban1, Penelope AllisonORCID identifier2 and Nicholas J. Cooper3
http://intarch.ac.uk/journal/issue50/11/index.html

Tyukin, I., Sofeikov, K., Levesley, J., Gorban, A.N., Allison, P. and Cooper, N.J. 2018 Exploring Automated Pottery Identification [Arch-I-Scan], Internet Archaeology 50. https://doi.org/10.11141/ia.50.11


# Google Colab

Setting up Google colab seems to be a place to access opencv, tensorflow and Python write code that can be reproduced.

## Instructions
https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c
I followed these instructions.

Made a folder in drive.google.com
clicked + New, more..., Connect More Apps...
searched for colaborate
mapped it

Mounted Google Drive.
from google.colab import drive
drive.mount('/content/gdrive')

In notebook settings, set hardware accelleration to GPU

I used some instructions from here
https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/
These instructions seem to have some typos

!pip3 install torch torchvision

import PIL
print(PIL.PILLOW_VERSION)

I get 4.3.0


import torch
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:    print('Bummer!  Training on CPU ...')
else:    print('You are good to go!  Training on GPU ...')

## Google Deep Learning Containers
### Introducing Deep Learning Containers: 
I followed these instructions:
https://cloud.google.com/blog/products/ai-machine-learning/introducing-deep-learning-containers-consistent-and-portable-environments

I installed this
### Google CLOUD SDK
Command-line interface for Google Cloud Platform products and services
https://cloud.google.com/sdk/
I installed this for just user jblackad locally

Worry - can I and others use this inexpensively?

In Google Cloud SDK Shell I listed:

gcloud container images list --repository="gcr.io/deeplearning-platform-release"

XXXXXX - ok it looks like I need Docker to pull this container in locally. That likely can't happen on my work machine which is locked down.
## Image classification
https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb

## Google Colab (Colaboratory)

I can run R in Colab and install
install.packages('ggmap')
install.packages('codetools')


### Problem installing keras in Google colab with R.
when I run 
```
Sys.which('virtualenv')
file.exists('/usr/local/bin/virtualenv')
#install.packages("keras")
library(keras)
install_keras()
```
I get:
```
virtualenv: ''
FALSE
Error: Prerequisites for installing TensorFlow not available.

Execute the following at a terminal to install the prerequisites:

$ sudo apt-get install python-virtualenv


Traceback:

1. install_keras()
2. install_tensorflow(method = method, conda = conda, version = tensorflow, 
 .     extra_packages = c(paste0("keras", version), extra_packages))
3. stop("Prerequisites for installing TensorFlow not available.\n\n", 
 .     install_commands, "\n\n", call. = FALSE)
 ```
 I did see this: 
 https://github.com/rstudio/tensorflow/issues/201
 
 But I am unable to make this work in Google colab at this time so I am switching to Python
 
 ### Installing keras 
 per Setting up GPU support P317  A.3.1 Installing CUDA
 I did:
 ! sudo apt-get install cuda-8-0
 and got this error:
 E: Unable to locate package cuda-8-0
 
 Per this note https://github.com/tensorflow/tensorflow/issues/16214
 I ran this
 ! sudo apt-cache search cuda-command-line-tool
 
 got this output:
 cuda-command-line-tools-10-0 - CUDA command-line tools
cuda-command-line-tools-10-1 - CUDA command-line tools

and so ran:
 ! sudo apt-get install cuda-*10*-0

 and it worked.
 
 Made account at
 https://developer.nvidia.com/cudnn
 
 Selected
 Download cuDNN v7.6.1 (June 24, 2019), for CUDA 10.0
 downloaded
 cuDNN Developer Library for Ubuntu18.04 (Deb)
 
 ! sudo dpkg -i /libcudnn7*.deb
 
 I am stil lhaving trouble with R and Google Colab
 - But I am still going to work with it since it seems easier to make something that can be explained and reused.
 
 # Creating a new set of images for training a model.
 I wanted to create an image processor on Colab using R, but I still have problems importing libraries like imager into Colab, so I will use Python
 
 ### Old Weather
 https://www.oldweather.org/naval_rendezvous.html
 
 I was thinking of using numbers from the ships' logs of this project, but that does not seem that appropriate for machine learning and traditional OCR is likely a better solution.
 
 I took pictures of coins.
 What I learned from that so far:
 * nickels are too shiny for good photographs. I am using old copper pennies
 * taking photographs from too high does not give a good result.
 * note a row number on the background
 * a picture of 6 coins (2 rows of 3) seems to work
 
 # Image inpainting
 
 I want to work with Mathias Gruber's PConv-Keras:
 https://github.com/MathiasGruber/PConv-Keras  
 
 I have had trouble cloning it to Colab
 So I am following Eduardo Rosas' instruction
 https://lalorosas.com/blog/github-colab-drive
 
 ## Steps to clone the repo
 https://colab.research.google.com
 new Python 3 Notebook
 
 Mounted the drive
 from google.colab import drive
drive.mount('/content/drive')

### Make a repos directory
import os

new_dir = "/content/drive/My Drive/repos"
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

new_dir = "/content/drive/My Drive/repos/MathiasGruber"
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

%cd $new_dir
!pwd

#Clone the repo
!git clone https://github.com/MathiasGruber/PConv-Keras

#now: "open drive.google.com, navigate to that folder, select the file, and click to open with>Colab." - Eduardo Rosas

#### Step1 - Mask Generation.ipynb
Opened it from google Drive

##### Standard for these notebooks
Changed runtime to Runtime | Change runtime... Runtime type: Python3, hardware accelerator: GPU

Mounted the drive

#Set up the directory
new_dir = "/content/drive/My Drive/repos/MathiasGruber/PConv-Keras"
%cd $new_dir
!pwd


#### Step2 - Partial Convolution Layer.ipynb
Opened it from google Drive
Did the standard steps of changing the runtime, mounting the drive and setting up the directory

#### Step 3
Did the standard steps of changing the runtime, mounting the drive and setting up the directory
Had to change model - I am not sure if this worked
Per here: https://github.com/MathiasGruber/PConv-Keras/issues/38
"the pytorch_vgg16.h5 can be downloaded from https://drive.google.com/file/d/1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0/view "
I moved this to /content/drive/My Drive/repos/MathiasGruber/PConv-Kera/data/logs

 Instantiate model
#model = PConvUnet(vgg_weights='./data/logs/pytorch_vgg16.h5')
model = PConvUnet(vgg_weights='./data/logs/pytorch_to_keras_vgg16.h5')

#### Step 4
Did the standard steps of changing the runtime, mounting the drive and setting up the directory
!pip install keras-tqdm

can't mess wth 512,512
Got out of resrouce errors - rest runtime - that fixed it

### using different weights
# Load weights from previous run
model = PConvUnet()
model.load(
    r"/content/drive/My Drive/repos/MathiasGruber/PConv-Keras/data/logs/imagenet_phase2/weights.02-4.26.h5",
    train_bn=False,
    lr=0.00005
)
