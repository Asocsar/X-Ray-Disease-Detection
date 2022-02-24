import os
from datetime import datetime
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score
import time
from datetime import timedelta
from os import listdir
from pathlib import Path

import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def create_folders(name, rank=None):
    if rank == 0 or rank == None:
        checkpoints = os.path.join('./', 'checkpoint')
        if not os.path.isdir(checkpoints):
            os.mkdir(checkpoints)
        
        checkpoints_dir = os.path.join(checkpoints, name)
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)



        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").replace('/','_').replace(' ', '_')

        name_date = name + '_Fast_LastTrials' + '[' + dt_string + ']'

        name_folder = name + 'Train_Val_Test'


        path = os.getcwd()
        path = '/'.join(path.split('/')[:-2])

        #dire_tmp = [os.path.join(path, 'TensorBoard', f) for f in listdir(os.path.join(path, 'TensorBoard')) if os.path.isdir(os.path.join(path, 'TensorBoard', f)) if name_folder in f][0]

        dire = os.path.join(path, 'TensorBoard', 'torch_1.9_2', name_date)
        print("TensorBoardId:", dire)
        Path(dire).mkdir(parents=True, exist_ok=True)

        return dire, checkpoints_dir
    
    else:
        return '', ''

def compute_AUCs(gt, pred, N_CLASSES):

    AUROCs = []
    gt_np = gt.cpu().numpy().astype('float32')
    pred_np = pred.cpu().detach().numpy()
    for i in range(N_CLASSES):
        gt_np_class = gt_np[:, i]
        pred_np_class = pred_np[:, i]
        AUROCs.append(roc_auc_score(gt_np_class, pred_np_class))
    return AUROCs


def print_remaining_time(epoch_start, epoch, EPOCHS):
    epoch_time = time.time() - epoch_start
    epoch_timef = timedelta(seconds=epoch_time)
    time_left = epoch_time*(EPOCHS - (epoch + 1))
    time_left = timedelta(seconds=time_left)
    print('Epoch time:', epoch_timef)
    print('Remaining estimated time', time_left)




def checkPoint(metadata, model, Max_AUROC_avg, AUROC_avg):
    if Max_AUROC_avg < AUROC_avg:
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f)) and metadata["name"] == f.split('_AUROC')[0]]
        if len(onlyfiles) > 0:
            os.remove(onlyfiles[0])

        name_checkpoint = os.path.join(metadata["CheckPointDir"], metadata["name"] + '_AUROC:' + str(AUROC_avg)+'.pkl')
        metadata["CheckPointLoad"] = name_checkpoint
        torch.save(model.state_dict(), name_checkpoint)
        return AUROC_avg
    
    else:
        return Max_AUROC_avg
