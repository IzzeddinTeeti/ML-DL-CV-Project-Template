
# from datetime import datetime
import time
from utils.logger import get_logger
import torch
import os
import numpy as np

logger = get_logger(__name__)

def set_seed(man_seed: int):
    # initialise torch, cuda, np seed for reproducibility
    torch.manual_seed(man_seed)
    torch.cuda.manual_seed(man_seed)
    torch.cuda.manual_seed_all(man_seed)
    np.random.seed(man_seed)

def set_device(cfg_devices, batch_size):
    """
    Set up the training device and the number of GPUs
    see https://github.com/ultralytics/yolov5/blob/72cad39854a7d9ebbd4d58994cefa966b0da8fc1/utils/torch_utils.py#L108
    """
    
    if cfg_devices == -1:
        device = torch.device('cpu')
        device_list = []
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg_devices if isinstance(cfg_devices, str) else str(cfg_devices)
        device = torch.device("cuda")
        cfg_devices = cfg_devices.split(',') if isinstance(cfg_devices, str) else cfg_devices
        device_list = [int(i) for i in cfg_devices] if isinstance(cfg_devices, list) else [cfg_devices]
        num_device = len(device_list)
        torch.cuda.set_device(device_list[0])

        assert torch.cuda.is_available() and torch.cuda.device_count() >= num_device, \
            f"Invalid CUDA '--device {cfg_devices}' requested, use 'cpu' or pass valid CUDA device(s)"
        
        if num_device > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % num_device == 0, f'batch-size {batch_size} not multiple of GPU count {num_device}'

    
    logger.info("Device: %s", device.type)
    if device.type == 'cuda': 
        logger.info("Num of GPUs: %d", len(device_list))
        for idx, device_name in enumerate(device_list):
            p = torch.cuda.get_device_properties(idx)
            logger.info("CUDA: %d (%s, %dMiB)", device_name, p.name, p.total_memory / (1 << 20))  # bytes to MB

    return device, device_list

def timeit(func):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.perf_counter() # time.time() time.perf_counter() datetime.now()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time
        # seconds = duration.total_seconds()
        minutes, seconds = divmod(duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        if func.__name__ == "train":
            logger.info("Training time: {:02.0f}d:{:02.0f}h:{:02.0f}m:{:02.0f}s ".format(days, hours, minutes, seconds))
        else:
            logger.info("Time taken for {:s}: {:02.0f}d:{:02.0f}h:{:02.0f}m:{:02.0f}s".format(func.__name__, days, hours, minutes, seconds))
        
        return result

    return timed