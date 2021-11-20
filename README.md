# What Causes a Driver’s Attention Reallocation? A Driver’s Attention-Guided Abnormal Driving Event Recognition Model on Untrimmed Driving Videos
# Abstract
It is important to research the types of abnormal driving event that cause a shift in a driver’s spatial attention.  We built an abnormal driving event dataset (ADED) based on the interaction between driver’s attention allocation and other traffic participants or elements. We relabeled and redivided BDD-A, a driver attention dataset in critical traffic situations, into six different semantic categories ofdriving events. The new ADED is introduced for abnormal driving event recognition. In addition, we proposed an abnormal driving event recognition model (called ADER-Net) with driver’s attention guidance to recognize what causes a driver’s attention reallocation. In ADER-Net, a driver’s attention-guided (DAG) branch is constructed to consider the driver’s spatial attention information. 

# Architecture

![fig1](https://github.com/10Messiah/ADED-ADER/blob/main/images/fig_2.png) 
The architecture of the proposed ADER-Net. (a) The overall architecture of the proposed ADER-Net. (b) The detailed architecture of the driver’s attention predicting model.

# Experiment Results
  * Comparison to State-of-the-Art Models
  ![fig2](https://github.com/10Messiah/ADED-ADER/blob/main/images/fig_3.png) 
  * Results of the classification
  ![fig3](https://github.com/10Messiah/ADED-ADER/blob/main/images/fig_3.png) 
# Abnormal Driving Event Dataset(ADED)
   * You can find the annotation in './annotation/annotation.xls'.
   ![fig4](https://github.com/10Messiah/ADED-ADER/blob/main/images/fig_11.png) 
   We divided all abnormal driving events into following six categories：
   driving normally (DN), avoiding crossing pedestrians (ACP), waiting for vehicles ahead (WVA), stopped by the red lights (SRL), stopped by the stop signs (SSS) and avoiding a lane-changing car (ALC).  
   The training set and test set also can be found in './annotation/annotation.xls'.
   * To obtain ADED dataset, follow these steps:
   1. Get BDD-A dataset from [the website](https://bdd-data.berkeley.edu/), and unzip the file.
   2. Download this project.
   3. Rename videos and split the vidoes to frames. ---Add your dir of unzipped file in './tools/rename_split.py'. And run
```python
    python tools/rename_split.py
 ``` 
   You may get several file folders. '../dataset/training_set_img' and '../dataset/test_set_img' are the file folders that contain images that are used to train and test model for recognizing the abnormal driving events. '../dataset/training_att_set_img' and '../dataset/test_att_set_img' are the file folders for training and testing model to predict the driver attention maps.
   
   4. Create the label file. --- Run
```python
    python tools/label.py
 ``` 
   You may get four txt files. '../dataset/16_frames_training.txt' and  '../dataset/16_frames_test.txt' can be used to load sequences and corresponding labels to train and test models for abnormal driving event recognization. '../dataset/attention_train.txt' and  '../dataset/attention_test.txt' can be used to load images and corresponding attention maps to train and test models for predicting the driver attention maps. 

# Abnormal Driving Event Recognition Net(ADER-Net)
 * You can find the codes in './ADER-net'.
 1. Add the dir of the txt files in './ADER-Net/data_load.py'  and  add the dir of the image file folders in './ADER-Net/options.py'
 2. To train the ADER-Net model, run
 ```python
    python ADER-Net/train.py
 ``` 
 3. To test the ADER-Net model, update the dir of the weight and run
 ```python
    python ADER-Net/test.py
 ``` 
 4. To evaluate the ADER-Net model, run
 ```python
    python ADER-Net/evalution.py
 ``` 
