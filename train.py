"""
This is the main file for training the model.
"""

from models.demomodel import SimpleModel
from data.ROAD import ROAD_dataset
from utils.logger import get_logger
from utils.misc import timeit
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import AverageMeter
from utils.losses import loss_ego_cls
from thop import profile, clever_format
from utils.metrics import mAP_ego
from utils.metrics import f1_ego



# from torch.utils.tensorboard import SummaryWriter
import wandb
wandb.init()
# to show multiple curves on the same plot, use
# wandb.log({"Train loss": value, "Val loss": value, "x": epoch})
logger = get_logger(__name__)



@timeit
def train(cfg, train_dataset, val_dataset):
    """
    Train the model
    """
    # tensorboard
    # writer = SummaryWriter()

    # Define the training device and the number of GPUs
    device, device_list = cfg.device, cfg.device_list
    num_device = len(device_list)
    
    # Training hyperparameters
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_epoch = cfg.TRAIN.NUM_EPOCH
    workers_single = cfg.TRAIN.NUM_WORKERS

    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=workers_single * num_device,
        shuffle=False,
        pin_memory=True,
        collate_fn=ROAD_dataset.custum_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size,
        num_workers=workers_single * num_device,
        shuffle=False,
        pin_memory=True,
        collate_fn=ROAD_dataset.custum_collate,
        drop_last=True,
    )
    
    # define loss function
    loss_cls = F.nll_loss()

    # define model 
    model = SimpleModel()

    # print model stats
    input = torch.empty((1, input_dim), dtype=torch.float32)
    macs, params = profile(model, inputs=(input, ))
    macs, flops, params = clever_format([macs, macs*2, params], "%.2f") # multiply macs by 2 for flops
    # model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model MACs: {:s}".format(macs))
    logger.info("Model FLOPs: {:s}".format(flops))
    logger.info("Model Parameters: {:s}".format(params))

    # parallelize model
    if num_device > 1:
        model = nn.DataParallel(model, device_ids = device_list)
    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.W_DECAY)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    total_iters = (len(train_dataset) // batch_size) * num_epoch
    itr_num = 0
    logger.info("Start training")
    for epoch in range(num_epoch):

        model.train()
        itr_num = train_one_epoch(cfg, epoch, train_loader, model, optimizer, loss_cls, itr_num)
        
        # validate 
        model.eval()
        val_loss = val_one_epoch(cfg, epoch, val_loader, model, loss_cls, itr_num)

        # scheduler 
        scheduler.step(val_loss)

        # save model
        
    
    logger.info("End training")

   

def train_one_epoch(cfg, epoch, train_loader, model, optimizer, loss_cls, itr_num):

    device = cfg.device
  
    epoch_loss = AverageMeter()

    loop = tqdm(enumerate(train_loader), total= len(train_loader)) 
    for batch_idx, (data, targets) in loop:
        itr_num += 1

        # data, target to device
        data = data.to(device)
        targets = targets.to(device)

        # prediction using model
        pred_labels = model(data)

        # loss calculation
        cls_loss = loss_cls(pred_labels, targets)

        # zero_grad, backpropagation, and step
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()
        
        # log
        epoch_loss.update(cls_loss.item())
        
        loop.set_description(f"Epoch [{epoch+1}/{cfg.TRAIN.NUM_EPOCH}]")
        loop.set_postfix(loss=epoch_loss.val, acc=1)

    wandb.log({'Train/Ego_loss': epoch_loss.avg}, step=epoch) # for wandb
    # writer.add_scalar('Train/Ego_loss', epoch_loss.avg, epoch) # for tensorboard

    return itr_num
    
def val_one_epoch(cfg, epoch, val_loader, model, loss_ego, itr_num):
    
    device = cfg.device
 
    epoch_loss = AverageMeter()
    f1_metric = AverageMeter()

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total= len(val_loader)) 
        for batch_idx, (data, targets) in loop:
            itr_num += 1

            # data, target to device
            data = data.to(device)
            targets = targets.to(device)

            # prediction using model
            pred_labels = model(data)
            
            # loss calculation
            cls_loss = loss_ego(pred_labels, targets)
            
            # Evaluation metrics
            f1 = f1_ego(pred_labels, targets)

            # log
            epoch_loss.update(cls_loss.item())
            f1_metric.update(f1.item())
            
            loop.set_description(f"Val epoch [{epoch+1}]")
            loop.set_postfix(loss=epoch_loss.val, F1=f1_metric.val)

    # val_metrics['Ego_loss'] = epoch_ego_loss.avg
    wandb.log({'Val/Ego_loss': cls_loss.avg}, step=epoch)
    wandb.log({'Metrics/F1_ego': f1_metric.avg}, step=epoch)

    return epoch_loss.avg

