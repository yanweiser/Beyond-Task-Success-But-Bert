# Training
Here we will walk through the process of training your own model. 

## Preprocessing

The image features have to be extracted using the ResNet. For that use the following command: `python3 -m utils.ExtractImgfeatures`.

## Training
Only supervised learning is used in this approach.

### Supervised Learning
The SL model is trained using modulo-epoch.

To train model with modulo-epoch use the command:

`python3 -m train.SL.train -modulo 7`


### GamePlay(Inference)
GamePlay is interaction between the Questioner and Oracle to figure out the target object. To do this interplay the code in the GamePlay folder is used. 
Be sure to have downloaded the binary for the oracle model (see models dir).

To run the Game use the command:

`python3 -m train.GamePlay.inference load_bin_path bin/SL/SL`

