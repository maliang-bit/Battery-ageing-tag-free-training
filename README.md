1. 'train.py' and 'test.py' are used for model training and test.
2. Change the exclude_rate, cycle_number and start_index parameters in 'module->config.py' to set the training data size, cycle number and starting voltage, respectively.
3. Change the mode parameter in 'train.py' and 'test.py' to select dataset.
4. The trained model used in the paper is in 'log' folder, the test results are stored in 'figure' folder, the four datasets are in 'dataset' folder.
5. Required python package:
Python 3.8
keras==2.6.0
matplotlib==3.5.3
numpy==1.19.5
pandas==1.4.0
scikit-learn==1.3.2
scipy==1.7.1
tensorflow-gpu==2.6.2
