# Steps used to set up Mathias Gruber's PConv-Keras in Google Colab
 
https://github.com/MathiasGruber/PConv-Keras  
 
Thanks to Eduardo Rosas' instructions here:
 
https://lalorosas.com/blog/github-colab-drive
 
## Steps to clone the PConv-Keras repo
 
### Open Google Colab
https://colab.research.google.com

new Python 3 Notebook
 
### Mount Google Drive drive

```
from google.colab import drive
drive.mount('/content/drive')
```

### Make a repos directory

```
import os

new_dir = "/content/drive/My Drive/repos"
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

new_dir = "/content/drive/My Drive/repos/MathiasGruber"
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

%cd $new_dir
!pwd
```

### Clone the repo

```
!git clone https://github.com/MathiasGruber/PConv-Keras
```

### Open the notebooks in Colab from Drive
Open drive.google.com

navigate to the notebooks folder

/content/drive/My Drive/repos/MathiasGruber/PConv-Keras/notebooks

select the file

Step1 - Mask Generation.ipynb

right-click | open with>Google Colaboratory

### Step1 - Mask Generation.ipynb


#### Standard instructions for these notebooks
Open it from Google Drive

right-click | open with>Google Colaboratory

Change runtime to Runtime | Change runtime... Runtime type: Python3, hardware accelerator: GPU

Mounted the Google Drive

Run:

```
#Set up the directory
new_dir = "/content/drive/My Drive/repos/MathiasGruber/PConv-Keras"
%cd $new_dir
!pwd
```

### Step2 - Partial Convolution Layer.ipynb
Follow standard steps above of changing the runtime, mounting the drive and setting up the directory.

### Step3 - UNet Architecture.ipynb
Follow standard steps above of changing the runtime, mounting the drive and setting up the directory.

I had to change model - I am not sure if this worked
Per here: https://github.com/MathiasGruber/PConv-Keras/issues/38
"the pytorch_vgg16.h5 can be downloaded from https://drive.google.com/file/d/1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0/view "
I moved this to /content/drive/My Drive/repos/MathiasGruber/PConv-Kera/data/logs

I changed

```
Instantiate model
#model = PConvUnet(vgg_weights='./data/logs/pytorch_vgg16.h5')
model = PConvUnet(vgg_weights='./data/logs/pytorch_to_keras_vgg16.h5')
```

### Step4 - Imagenet Training.ipynb
Follow standard steps above of changing the runtime, mounting the drive and setting up the directory.

```
!pip install keras-tqdm
```

# This is not finished.  This is where I am at this time.
