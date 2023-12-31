# Description

Three experiments are included in this program: _pretraining & linear classification_, _direct supervised learning on CIFAR-100_, and a supplementary experiment of _pretraining & complete downstream task learning_. 

# Experiment

* Run ```pretrain.py```  for pretraining.
* Run ```cifarTrain.py``` for training the first two experiments simultaneously.
* Run ```Fulltrain.py``` for the supplementary experiment.

# Test

* Run ```test.py``` for testing.
  
# Remark

* Model results and parameters will be saved in the __save__ folder.
* Please download the pre-training dataset from the following link [ImageNet](https://pan.baidu.com/s/1JWDda58yk0jmopLyUVKhBw?pwd=2acc) and place it under __../data/ImageNet__.
* If you want to test the model without training, please download the model parameters from [Parameters](https://pan.baidu.com/s/1JM8wK5astEGJzevowssxVA?pwd=9gqh) and place them under __save__.


