# Visually-Grounded Dialogue State Encoder (GDSE)
This repository contains code for the NAACL 2019 paper

**[Beyond task success: A closer look at jointly learning to see, ask, and GuessWhat](https://arxiv.org/abs/1809.03408)**  
Ravi Shekhar, Aashish Venkatesh, Tim Baumgärtner, Elia Bruni, Barbara Plank, Raffaella Bernardi and Raquel Fernández

# Setup
All code was developed and tested on Windows 10, Python3.8 and Pytorch 1.12+cu116. 

# Code Organisation
The code is organised in the following main directories: 


```
---bin
---config 
---data
---logs
---models
---train
---utils
    |---datasets
```

```
---bin
```
This folder contains the PyTorch binary files for the models. This folder is further has three subfolders SL, CL and Oracle. SL folder already has the best pretrained SL models, **SL**. 

```
---config
```
This folder contains JSON files which define hyperparameters, experiment configurations, etc.

```
---data
```
This folder should have all the data files like training data files, vocabulary etc. Place the datasets here. They can be downloaded using the following commands.

```
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```
You also need to place the raw MSCOCO images from which the image features need to be extracted. Get the images from here:

```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/img/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/img/

unzip data/img/train2014.zip -d data/img/raw
unzip data/img/val2014.zip -d data/img/raw
```

Make sure to place all images in a single folder since the unzip will create additional folders.

```
---logs
```
This folder will contain all the tensorboard and text logging from the model. This further has three subfolders namely CL, GamePlay and SL.

```
---models
```
This folder contains the definition of all models, i.e. parameters and forward-pass.
```
---train
```
This folder contains the scripts for training and inference of the model. This further has four subfolders namely CL, GamePlay(inference code), Oracle and SL. Oracle folder contains the code for training the baseline model. This is where all comes together: The experiment configuration is loaded from the respective JSON file (config); the model is instantiated with the defined hyperparameters; a dataset iterator is defined and eventually the training is performed.

```
---utils
```
This contains various functions which are used throughout the code (mainly in training and data preprocessing).

```
---utils/datasets
```
Contains dataset subclasses for each training method, inference and Oracle. Every dataset has a function to perform preprocessing on the original GuessWhat?! data and store the result in a JSON file. Thereby, preprocessing has only to be done once. This further has four subfolders namely CL, GamePlay, Oracle and SL. 

# Training

The procedure for training your own models [is described here](TRAINING.md)


# Reference

If you find this code useful, consider citing our work:

```
@inproceedings{shekhar2019beyond,
  title = {Beyond task success: A closer look at jointly learning to see, ask, and GuessWhat},
  author={Ravi Shekhar and Aashish Venkatesh and Tim Baumgärtner and Elia Bruni and Barbara Plank and Raffaella Bernardi and Raquel Fernández},
  year={2019},
  booktitle = {NAACL-HLT}
}
```

# License

BSD

