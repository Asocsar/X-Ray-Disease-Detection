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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime
import json
from efficientnet_pytorch import EfficientNet
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = 14


#%%
def main(metadata):

    train_loader, train_len, test_loader, test_len, val_loader, val_len = Sound_Flip_data_loader_NC(metadata)

    model = Efficient(metadata['NameEfficient'], metadata['weightsPath'], N_CLASSES).cuda()

    BETAS = [float(x) for x in metadata['BETAS'].split(',')]
    optimizer = optim.Adam(model.parameters(),lr=metadata['LR'], betas=BETAS)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=metadata['PatiencePlateau']) 
    

    writer = SummaryWriter(metadata['TensorBoardFold'])
    with open(os.path.join(metadata['TensorBoardFold'], 'config.json'), 'w+') as out_conf:
        json.dump(metadata, out_conf)

    EPOCHS = metadata['EPOCHS']

    Max_AUROC_avg = 0

    start_execution = time.time()
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        epoch_start = time.time()
        train_NC(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, writer)
        model.eval()
        AUROC_avg, loss = val_NC(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, writer)
        model.train()
        scheduler.step(loss)
        print_remaining_time(epoch_start, epoch, EPOCHS)
        Max_AUROC_avg = checkPoint(metadata, model, Max_AUROC_avg, AUROC_avg)
    end_execution = time.time()
    elapsed_time = end_execution - start_execution
    elapsed_time = timedelta(seconds=elapsed_time)
    print("Elapsed Time for all epochs", elapsed_time)

    model.eval()
    test_NC(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, writer)



#%%
if __name__ == '__main__':
    TensorDir,CheckPointDir = create_folders(sys.argv[0][:-3])
    f = open('../../hparams/' + sys.argv[0][:-3] + '.json')
    metadata = json.load(f)
    f.close()

    metadata["TensorBoardFold"] = TensorDir
    metadata["CheckPointDir"] = CheckPointDir
    metadata["name"] = sys.argv[0][:-3]
    main(metadata)


# %%
