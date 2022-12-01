# Summary
This repository is to host models and satasets for Heart Transplant rejection prediction using longitudanal ECG data.
There are 3 main types of Models that we are experimenting with:

<img src="Screenshot 2022-12-01 at 10.42.47 AM.png" width="550" title="hover text">

## Model 1
Model 1 is just a simple 1d CNN to try and predict transplant rejection from a single ECg sample

## Model 2
Model 2 employs the Model 1 architecture as a twin feature extraction layer for 2 longitudanal ECG samples for a single patient
These samples are passed onto a feed forward network alognside normalized time information, and Biopsy results for sample 1.
### 2.0
One version of this model just concatenates the features extracted from the CNN layer with the time difference and biopsy label.
### 2.1
The second version of this model concatenates (sample2-sample1)/dt, sample 1 and the biopsy results for sample 1.


## Model 3
Not implemented yet
Goal of model 3 is to use a transformer to learn if past time points of ECGs can lead to a better prediction of rejection at the current time point.