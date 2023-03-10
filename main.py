
import argparse
import sys
import yaml
import munch
from utils.logger import create_exp_name, get_logger
from utils.misc import set_seed, set_device
import socket
import getpass
from train import train
import torch
from torchvision import datasets, transforms

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer models.")
    parser.add_argument(
        "--config_file",
        default="configs/ViT_ROAD.yaml",
        type=str,
        help="path to config file.",
    )
    args = parser.parse_args()

    # Load config yaml file as nested object
    cfg = yaml.safe_load(open(args.config_file, "r"))
    cfg = munch.munchify(cfg)
    
    # start logging
    name = create_exp_name(cfg)
    logger.info("Host name: %s", socket.gethostname())
    logger.info("User name: %s", getpass.getuser())
    logger.info("Python Version: %s", sys.version)
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("Experiment Name: %s", name)

    # Set up the training device and the number of GPUs
    device, device_list = set_device(cfg.TRAIN.DEVICE, cfg.TRAIN.BATCH_SIZE)
    cfg.update(device = device)
    cfg.update(device_list = device_list)
    
    # set manual seed for reproducibility
    set_seed(cfg.TRAIN.SEED)

    # train transform and dataset
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    logger.info('Done Loading Training Dataset')

    # Val_transform and dataset
    val_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    val_dataset = datasets.MNIST('../data', train=False, download=True, transform=train_transform)
    logger.info('Done Loading Validation Dataset')
    
    # run training
    train(cfg, train_dataset, val_dataset)