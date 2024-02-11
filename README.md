# Terrastack: Satellite Agri-Modeling

## Problem Statement:
### Dataset Overview:
Input Data: Clipped rasters for 568 GIDs
Temporal Coverage: 12 months
### Part-1:
The given dataset is unlabelled. So first task is to apply image processing techniques to classify each image in to three different class. This part does not require any learning/ predictive model as it is a part of known dataset preparation.
### Part-2:
In this part, we develop a model and train it on the above prepared dataset and try to test it in new data.

## Image Annotation:

From visual inspection, for class <lush>, the colour of the image is dark green, for <growing>, the colur is light green and for no_crop, the images contain barren land and are mostly brown.
So, we took the average values of R, G and B and compared them. If G_avg is greater than B_avg by a huge margin, the image is predominantly green and so lush. If the difference between G_avg and B_avg is small, then growing and for other cases, it is no_crop.

## Model:

We used a very simple and light weight model, namely Softmax Multiclass Classifiaction.

## Result:

Epoch 0, Loss: 7.6560

Epoch 10, Loss: 6.6759

Epoch 20, Loss: 5.9268

Epoch 30, Loss: 5.5285

Epoch 40, Loss: 5.3395

Epoch 50, Loss: 5.1969

Epoch 60, Loss: 5.0654

Epoch 70, Loss: 4.9397

Epoch 80, Loss: 4.8194

Epoch 90, Loss: 4.7041

Test Accuracy: 70.01466275659824%


 
