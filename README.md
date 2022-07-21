# Daily Gesture Recognition During Human-Robot Interaction Combining Vision and Wearable Systems

It is a classification of daily gesture data using LSTM.

## Prerequisites
* Ubuntu 16.04

* Python 3

* pytorch

## run
```
python3 k_fold_lstm_final.py
```
# Daily Gesture Recognition During Human-Robot Interaction Combining Vision and Wearable Systems
The recognition of human gestures is crucial for improving the quality of human-robot cooperation.This article presents a system composed of a Pepper robot that mounts an RGB-D camera and an inertial device called SensHand. The system acquired data from twenty people who performed five daily living activities (i.e. Having Lunch, Personal Hygiene, Working, House Cleaning, Relax). The activities were composed of at least two “basic” gestures for a total of 10 gestures. The data acquisition was performed by two cameras positioned laterally and frontally to mimic the real conditions. The acquired data were off-line classified considering different combinations of sensors to evaluate how the sensor fusion approach improves the recognition abilities. Specifically, the article presents an experimental study that evaluated four algorithms often used in computer vision, i.e. three classical machine learning and one belonging to the field of deep learning, namely Support Vector Machine, Random Forest, K-Nearest Neighbor and Long Short-Term Memory Recurrent Neural Network. The comparative analysis of the results shows a significant improvement of the accuracy when fusing camera and sensors data, i.e. 0.81 for the whole system configurationwhen the robot is in a frontal positionwith respectto the user (0.79 if we consideronly the index finger sensors) and equal to 0.75 when the robot is in a lateral position. Interestingly, the system performs well in recognising the transitions between gestures when these are presented one after the other, a common event in the real-life that was often neglected in the previous studies.

@article{fiorini2021daily,
  title={Daily gesture recognition during human-robot interaction combining vision and wearable systems},
  author={Fiorini, Laura and Loizzo, Federica G Cornacchia and Sorrentino, Alessandra and Kim, Jaeseok and Rovini, Erika and Di Nuovo, Alessandro and Cavallo, Filippo},
  journal={IEEE Sensors Journal},
  volume={21},
  number={20},
  pages={23568--23577},
  year={2021},
  publisher={IEEE}
}
