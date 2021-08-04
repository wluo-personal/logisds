# logisds
## Modeling Task
Given the continuous history events including event timestamp, longitude, latitude
1. Predict predict the number of events occurring in a half hourly block for the next 7 days following the last date in the data
2. Predict x and y coordinate for question 1.

## Modeling Idea
This is a time-series prediction problem. The problem can be defined as given the timestamp T and the history infomration before T, and we need to predict what will happend after T, eg from T+1 to T+N. 
There are serveral ways to modeling this problem:
1. Given the data between T-x and T, predict T+1
2. Use sequence to sequence prediction.  

Since solution 2 requires much more data than 1 and the modeling is much more complicated than 1, and thus I will choose 1 for this project. To predict T+1, there are two ways:
1. Use 1-D features, eg, the count of occurence in last X minutes. Linear regression, SVM, lightGBM, etc can be used.
2. use 2-D features, where one dimention stands for the timestamp, and another dimension stands for the feature of that timestmap. RNN models can be used for this solution.
Since RNN has the natural ability to reveal context dependencies/relationship. And thus, I will use RNN to solve this problem. This will be
a sequence-to-one model. During inference stage, in order to predict the next event, the previous prediction will be append to the input and the oldest input timestamp will be popped from the input.
### What to Predict
The RNN model will output o1, o2, o3. Where 
- o1 is the time difference in seconds between predicted event and the last event. 
- o2 is the x, which is the x coordinate associated with the predicted event
- o3 is the y, which is the y coordinate associated with the predicted event
### Model Input Data
Regardless of the batch dimension, the input data will have such shape (N_steps, N_features)
- N_steps: Consider each event as a step, N_steps means N_steps continuous events data
- N_features: For each step, the data will be N_features
### Labeling
Given the original data as data_ori, the model input as data_input and the output as data_output
- data_input_i will be constructed from data_ori[i-N_steps: i+1]
- data_output_i will be (o1_i, o2_i, o3_i) and the ground truth will be obtained from data_ori[i+1]

## Feature Engineering  
There is no missing data,
### time
time features contains the belows:
- hour: one-hot encoding. Even in same/nearby hours may shapre some similarities. 
- weekday: one-hot encoding. Even in the same weekday may shapre some similarities
- delta in seconds with previous event timestamps. There are some outliers, however those outliers can not be simply discarded and thus I will leave it unchanged.

### x,y corrdinates
The corrdinates has very small standard-deviation. In order to make the output/input features at the same scale, I will do "z" normalization
to make x, y has 0 mean and 1 std.  

## Modeling
The 

![alt text](https://github.com/wluo-personal/logisds/blob/main/image.jpg?raw=true)
