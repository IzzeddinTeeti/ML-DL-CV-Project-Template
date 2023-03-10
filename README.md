# Machine Learning, Deep Learning, and Computer Vision Project Template
This is a generic template for Machine Learning, Deep Learning, and Computer Vision projects, it illustrats the main elements and the pipeline connecting these elements.

## Machine Learning Pipeline
### 1. Configuration and Parameters
* This template puts all the parameters, hyperparmeters, and configurations in one Yaml file inside the configs folder. This puts all the parameters related to dataset, model, schedular, optimizor, devices, etc in one place, and I found it easier, neater, and better than defining the parameters in the command line (ending with a very long command e.g `CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /road-dataset/ /road-dat /3D-RetinaNet/kinetics-pt --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041`.

* It uses `yaml` and `munch` packages to convert yaml config file into nested dictionaries. This will enable easy access to the parameters, for example `batch_zise = cfg.Train.batchsize`.

### 1. Datasets and DataLoaders
* 

### 2. Train
model
device
loss
optimisor
schedular

### 1. test
no grad
loss 
metric

### 1. helpers
timers
loggers
Progress bar

## bonus
### To Automatically generate a requierments.txt
1. install `pipreqs` package using `pip3 install pipreqs`
2. in the target directory run `python3 -m  pipreqs.pipreqs`
3. then install requirements using `pip install -r requirements.txt`

### To automatially format the code to python standards
1. Install `black` package using `pip install git+https://github.com/psf/black`
2. In the terminal, run `black python_file_name.py`

