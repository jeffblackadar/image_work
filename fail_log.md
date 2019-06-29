
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
