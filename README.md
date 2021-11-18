# Abnormal Driving Event Dataset(ADED)
   * You can find the annotation in './annotation/annotation.xls'.
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
