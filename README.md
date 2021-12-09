# AppleHand

This repository contains all files related to using recurrent neural networks on apple picking data.

This includes:

  Collected data  - located in the "Deep Learning Data" folder

  Recurrent Neural Networks and training code - located in the main folder

  

#Motivation

During an apple pick, we want to determine the stem orientation and likelyhood of pick success, both features which benefit from knowing both the current state of the system and the previous states.

#Installation

ensure you have the following packages installed. An easy way is to install python through anaconda, then pip install pytorch

​	numpy

​	pytorch

​	pandas

once you have these installed, clone the repo

#How to use

pick_classifier_main.py is the file to run for basically all operation. Desired features are selected with arguments. A basic command to train a network and plot the acc over the epochs is given below

python pick_classifier_main.py --plot_acc=True

