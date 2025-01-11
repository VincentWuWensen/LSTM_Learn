# LSTM_Learn
Just some records of learning the LSTM, it contains some sectiosn by AI modifications. The code integrates a number of technologies, including attention, CNN and KAN etc. Although this goes against the principle of Occam's Razor which make the prediction result not so well, but LSTM do learns the trend of future. By the way, how to solve the infomation leakage is still a preblem.
https://www.kaggle.com/code/yunsuxiaozi/lstm-predict-mean-temp is the code blueprint.

2025.1.11
The problem of information leakage is addressed by partitioning the dataset and applying normalization techniques to the test set, validation set and test set respectively. This makes the model less capable. We additionally added L1 regularization, but regularization did not improve the model's capability. This hints at the limitations and conflicts of integrating multiple techniques.
