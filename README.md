# deep-learning-training-template-pytorch
This repo is my most used training code in pytorch. I decomposed functions into different files, which is more readable, and it can push notification to mobile phone, show the train and eval figure in realtime.
## For pushover
I used the service provided by pushover, which can push notifications to mobile phone.
## Usage
jsut change the reader and net to fit your task, and change the data realted code from loader in both `train_model` and `eval_model` func of the file `fit.py`.
