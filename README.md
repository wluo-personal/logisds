# logisds
## Modeling Task
Given the continuous history events including event timestamp, longitude, latitude
1. Predict the number of events occurring in a half hourly block for the next 7 days following the last date in the data
2. Predict x and y coordinate for question 1.

## Modeling Idea
This is a time-series prediction problem. The problem can be defined as given the timestamp T and the history information before T, and we need to predict what will happen after T, eg from T+1 to T+N. 
There are several ways to modeling this problem:
1. Given the data between T-x and T, predict T+1
2. Use sequence to sequence prediction.  

Since solution 2 requires much more data than 1 and the modeling is much more complicated than 1, and thus I will choose solution 1 for this project. To predict T+1, there are two ways:
1. Use 1-D features, eg, the count of occurence in last X minutes, etc. Linear regression, SVM, lightGBM, etc can be used.
2. Use 2-D features, where the first dimension stands for the timestamp, and the 2nd dimension stands for the feature of that time. RNN has the natural ability to reveal context dependencies/relationship. And thus, I use RNN to solve this problem. This will be
a sequence-to-one RNN model. During inference stage, in order to predict the next event, the previous prediction will be appended to the input and the oldest input timestamp will be popped from the input.
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
- hour: one-hot encoding. Event in same/nearby hours may share some similarities. 
- weekday: one-hot encoding. Event in the same weekday may share some similarities
- time delta in seconds (current - previous ). There are some outliers, however those outliers can not be simply discarded and thus I will leave it unchanged.

### x,y corrdinates
The coordinates have very small standard-deviation. In order to make the output/input features are at the same scale, I will do "z" normalization
to make x, y has 0 mean and 1 std.  

## Modeling
The model defines as below, where input is 300 * 34 and output shape is 1 * 3. 
- input layer: The input take 300 time steps and 34 features for each timestamp.
- bi-directional-lstm layer: This layer is a bi-directional LSTM and all hidden state will be captured
- flatten-layer: This layer flattens all the hidden states of the previous layer
- dropout layer: This layer has a dropout ratio default to 0.3
- time delta output layer: This layer connects to dropout layer and will output time delta
- x,y coordinates output layer: This layer connects to dropout layer and will output x,y coordinates.
- final concatenate layer: This layer concatenates the time,x,y layers to be an output layer  

The optimizer uses Adam, and the learning rate is set to 0.001.
![alt text](https://github.com/wluo-personal/logisds/blob/main/jupyter/model.png?raw=true)

## Training
Data is randomly split into 60% training, 40% validation. The batch size is set to 100 and total epoch is set to 100 with early stopping for 10 rounds.

## Inference
Given that the last timestamp in the training data is T, now we need to predict all the events from T to T + 7 days.
- predict T + 1, by using T - 30 to T
- predict T + 2, by using T - 29 to T + 1
- predict until T + x, where predict T + x is greater than T + 7 days

# Project Detail
## setup
1. git clone
```shell
git clone https://github.com/wluo-personal/logisds.git
```

2. install dependencies (3.7.1 <= python < 3.8 ). Run below code under project root dir. (tensorflow + pandas + numpy)
```shell
conda create --name newenv python=3.7.10 --no-default-packages
conda activate newenv
pip install poetry
poetry install
```

## project directory
1. "pyproject.toml" defines the package dependencies
2. jupyter folder contains all demo code
3. test folder contains all pytest cases
4. data folder contains:
    - data.csv: the raw data
    - inference.csv: future 7 days inference which includes event_time, x, y
    - inference_halfhour.csv: futures 7 days inference which includes time beginning for each half-hour block and the number of occurrence
    - model.* is the RNN model weights
5. logisds is the python package folder
    - data.py provide the method to do feature engineering, provide training, validation data
    - model.py define the RNN model structure
    - train.py define how to train the RNN (uncomment the main method and run below. Please refer to jupyter/train.ipynb)
      ```shell
        python logisds/train.py 
      ```
    - inference.py defines how to inference the futures events and the result will be saved into data folder. Please refer to jupyter/inference.ipynb
      ```shell
        python logisds/inference.py 
      ```
    - utils.py defines the python log.
    

