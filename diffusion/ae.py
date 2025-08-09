import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *

 
def get_ae():
    #ckpt = '/home/qq/2/diff/AE_airplane.pt'
    ckpt = '/home/qq/2/diff/AE_all.pt'
    # Checkpoint
    ckpt = torch.load(ckpt)
    seed_all(ckpt['args'].seed)

    model = AutoEncoder(ckpt['args']).cuda()
    model.load_state_dict(ckpt['state_dict'])
    return model

#如果是自己的预训练的
def my_ae(str):
    if str == 'ModelNet40':
        ae = torch.load('/home/qq/2/diff/my/my_modelnet40_ae_all_2.pt').cuda()
        #ae = torch.load('/home/qq/2/diff/my/my_modelnet40_ae_chair_3.pt').cuda()
    elif str == 'ShapeNetPart':
        ae = torch.load('/home/qq/2/diff/my/my_ae_chair.pt').cuda()
    return ae


def ae_opt(model):
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=0
                                 )
    scheduler = get_linear_scheduler(
        optimizer,
        #start_epoch=150*THOUSAND,
        #end_epoch=300*THOUSAND,
        start_epoch=1000,
        end_epoch=2000,
        start_lr=1e-3,
        end_lr=5e-6
    )
    return optimizer, scheduler

