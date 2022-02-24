#%%
import sys
sys.path.append('../')

from utils.utils import *
from utils.inference import *
from utils.data_loaders import *
from modelArchitecture import *
from train_test import *

import os
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import datetime
import json
import argparse
from datetime import timedelta
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = 14


#%%
def main(rank, metadata):
    
    setup(rank, metadata["num_gpu"])
    train_loader, train_len, test_loader, test_len, val_loader, val_len = default_distributed_data_loader(rank, metadata)


    model = DenseNet121_Mixed(N_CLASSES, metadata['weightsPath']).cuda()
    device = torch.device('cuda:{}'.format(rank))
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    BETAS = [float(x) for x in metadata['BETAS'].split(',')]
    optimizer = optim.Adam(model.parameters(),lr=metadata['LR'], betas=BETAS)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=metadata['PatiencePlateau']) 
    
    if rank == 0:
        writer = SummaryWriter(metadata['TensorBoardFold'])
        with open(os.path.join(metadata['TensorBoardFold'], 'config.json'), 'w+') as out_conf:
            json.dump(metadata, out_conf)
    else:
        writer = None

    EPOCHS = metadata['EPOCHS']

    if rank == 0:
        start_execution = time.time()
        Max_AUROC_avg = 0
    for epoch in range(EPOCHS):
        if rank == 0:
            print('Epoch:', epoch)
            epoch_start = time.time()
        train_Mixed_distributed(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, writer, device, rank)
        model.eval()
        AUROC_avg, loss = val_Mixed_distributed(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, writer, device, rank)
        model.train()
        scheduler.step(loss)
        if rank == 0:
            print_remaining_time(epoch_start, epoch, EPOCHS)
            Max_AUROC_avg = checkPoint(metadata, model, Max_AUROC_avg, AUROC_avg)
    if rank == 0:
        end_execution = time.time()
        elapsed_time = end_execution - start_execution
        elapsed_time = timedelta(seconds=elapsed_time)
        print("Elapsed Time for all epochs", elapsed_time)

        model.eval()
        test_Mixed(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, writer)



#%%
if __name__ == '__main__':
    rank = torch.cuda.device_count()

    name = sys.argv[0][:-3].split('_')
    name = '_'.join(name[:-1] + [str(torch.cuda.device_count()) + name[-1]])
    TensorDir,CheckPointDir = create_folders(name)
    print(name)
    f = open('../../hparams/' + name + '.json')
    metadata = json.load(f)
    f.close()

    metadata["TensorBoardFold"] = TensorDir
    metadata["CheckPointDir"] = CheckPointDir
    metadata["name"] = name

    assert metadata['num_gpu'] == torch.cuda.device_count()

    print(metadata)

    mp.spawn(
        main,
        args=(metadata,),
        nprocs=metadata["num_gpu"],
        join=True
    )