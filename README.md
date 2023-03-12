# Machine Learning, Deep Learning, and Computer Vision Project Template
This is a generic template for Machine Learning, Deep Learning, and Computer Vision projects. It illustrates the principal elements and the pipeline connecting these elements. It also contains helper functions and decorations to time, log, and see the progress of processes. 

## Machine Learning Pipeline
### 1. Configuration and Parameters
* This template puts all the parameters, hyperparameters, and configurations in one place, which is a Yaml file inside the configs folder. I found it more manageable, neater, and better than defining the parameters in the command line (ending with a very long command e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /road-dataset/ /road-dat /3D-RetinaNet/kinetics-pt --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041`). Instead, you can change the parameters in the config file and then run `python main.py --config configs/Simple_MNIST.yaml`.

* It uses `yaml` and `munch` packages to convert a yaml config file into nested dictionaries. This will enable easy access to the parameters, for example, `batch_size = cfg.Train.batchsize`.

### 2. Datasets and DataLoaders
* After reading the relevant hyperparameters, parameters, and variables. The next step is to prepare the dataset for the training process. It starts by defining the dataset itself, then batchifying it using the Pytorch built-in dataloader. 

### 3. Train
In the `train.py`, the following will be defined:
* **Model:** The training starts by choosing the model that will be used to process the data (for example, it could be a simple NN, CNN, or a transformer), knowing that the model's parameters will be defined in the config file.
* **Device/es:** The template contains a helper function to choose the available device/es (CPU or GPU) to process the data and whether multiple GPUs will be used. 
* **Loss/es:** used to measure the model's performance and judge its progression/degradation through training (might be classification, regression, or multitask loss).
* **Optimiser:** that will drive the training towards convergence, enhance the model's performance, and reduce loss.
* **Scheduler:** that will change the learning rate according to a defined trend (e.g. every certain number of steps).

### 4. Test
During the test/validation process, there will be no grad, meaning the model does not learn, train, or change the values of its parameters. The model will be used mainly for processing the test/val data, then output a certain metric value to judge the model performance. Usually, the testing/validation process has the following attributes:
* **model.eval() and no_grad:** This will shut down the model's training process.  
* **Loss/es:** having the same loss for the training and testing makes sense to know if the model is over/underfitting.
* **Metric/s:** which is necessary to measure the model performance on the specific task. Usually, the metric is not differentiable, unlike losses.

### 5. Helpers
The main advantage of this template, other than providing a general structure, is the helper functions it contains, which are listed below:
* **Timers:** Used to calculate the processing time of a function. The template defines the timer function as a decorator, which can be attached to any function. For example, I used to record the taken for training and validation.
* **Loggers:** The template contains a punch of functions to 
  * Create an experiment name.
  * Log the version of different used modules and frameworks, like Python version, PyTorch version, Cuda version, etc. and 
  * Show the file and line that produced the output, like:
   ```
  [INFO:    main.py:   43]: User name: izzeddin
  [INFO:    main.py:   44]: Python Version: 3.9.12 (main, Jun  1 2022, 11:38:51)
  [INFO:    main.py:   45]: PyTorch Version: 1.12.1+cu113
  [INFO:    main.py:   46]: Experiment Name: ROAD-e50
  ```
* **Progress bar:** The template uses `tqdm` package to show the progress of the training and validation processes, and it further shows the loss and metric values for each iteration and epoch. 
```
Epoch [2/50]: 100%|██████████████████████████████████████████████████████| 221/221 [00:03<00:00, 69.48it/s, acc=0.27, loss=0.434]
```

* **Metric and W&B:** The template record the value of the metric at each iteration and calculates the average of iterations for each batch. This will be shown on the graph on W&B website. 

* **Model's parameters, FLOPs, and MACs** The template calculates the model's number of parameters, number of Multiply-Add cumulation (MACs), and Floating Point Operations (FLOPs).

## Bonus
### To automatically generate a requierments.txt file
1. Install `pipreqs` package using `pip3 install pipreqs`
2. In the target directory, run `python3 -m  pipreqs.pipreqs`
3. Then install requirements using `pip install -r requirements.txt`

### To automatically format the code to python standards
1. Install `black` package using `pip install git+https://github.com/psf/black`
2. In the terminal, run `black python_file_name.py`

