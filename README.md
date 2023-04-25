Introduction
This repository contains code for two different machine learning tasks: federated learning and centralized machine learning. The code for each task is located in a separate directory.

Federated-Learning-For-Healthcare/Federated Deep Learning/ & Federated-Learning-For-Healthcare/Federated Logistic Regression/

The following files are included:
server.py: This file is used to aggregate the model weights from multiple clients and run federated learning.

client1.py and client2.py: These files are used to fetch the dataset, train the model, and update the weights and parameters. 
They communicate with the server to get aggregated scores.

utils.py: This file contains helper functions, such as loading the dataset and model parameters.

To run the code, follow these steps:
Download the data from this link https://www.kaggle.com/datasets/vinayakshanawad/heart-rate-prediction-to-monitor-stress-level, and place it in the same directory as the code.
Open the code in an IDE.
Run the following commands simultaneously in separate terminal windows:
Copy code
python3 server.py
python3 client1.py
python3 client2.py
Centralized Machine Learning and Deep Learning
The Centralized_ML_DL directory contains Jupyter notebook files for implementing centralized machine learning and deep learning. The code can be run on Google Colab, Kaggle, or locally using Jupyter notebook. The dataset for this task is the same as the one used in the previous task.

To run the code, follow these steps:

Download the data from this link https://www.kaggle.com/datasets/vinayakshanawad/heart-rate-prediction-to-monitor-stress-level.
Upload the Jupyter notebook file to your desired platform or run it locally using Jupyter notebook.
Run the code in the notebook.
