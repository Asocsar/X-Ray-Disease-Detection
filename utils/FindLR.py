#%%

import sys
sys.path.append('../')
from data_loaders import *
from modelArchitecture import *

import torch.nn as nn
import torch.optim as optim
import json
from apex import amp

from LrFinder import *

#%%
def create_plot(metadata, model, trainloader, TenCrop):
    BETAS = [float(x) for x in metadata['BETAS'].split(',')]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=metadata['LR'], betas=BETAS)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda", TenCrop=TenCrop)
    lr_finder.range_test(trainloader, end_lr=10, num_iter=300, step_mode='exp')
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state


#%%
N_CLASSES = 14

f = open('FindLR.json')
metadata = json.load(f)
f.close()

BATCH_SIZE_D = metadata["BATCH_SIZE_D"]
BATCH_SIZE_B0 = metadata["BATCH_SIZE_B0"]
BATCH_SIZE_B1 = metadata["BATCH_SIZE_B1"]
BATCH_SIZE_B2 = metadata["BATCH_SIZE_B2"]
BATCH_SIZE_B3 = metadata["BATCH_SIZE_B3"]

TEN_CROP_D = metadata["TEN_CROP_D"]
TEN_CROP_B0 = metadata["TEN_CROP_B0"]
TEN_CROP_B1 = metadata["TEN_CROP_B1"]
TEN_CROP_B2 = metadata["TEN_CROP_B2"]
TEN_CROP_B3 = metadata["TEN_CROP_B3"]

RESIZE_D = metadata["RESIZE_D"]
RESIZE_B0 = metadata["RESIZE_B0"]
RESIZE_B1 = metadata["RESIZE_B1"]
RESIZE_B2 = metadata["RESIZE_B2"]
RESIZE_B3 = metadata["RESIZE_B3"]

modelD = DenseNet121_Mixed(N_CLASSES, metadata['weightsPathDense']).cuda()
model0 = Efficient_Mixed(metadata['NameEfficientB0'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
model1 = Efficient_Mixed(metadata['NameEfficientB1'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
model2 = Efficient_Mixed(metadata['NameEfficientB2'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
model3 = Efficient_Mixed(metadata['NameEfficientB3'], metadata['weightsPathEfficient'], N_CLASSES).cuda()

#modelD = DenseNet121_Pretrained(N_CLASSES, metadata['weightsPathDense']).cuda()
#model0 = Efficient(metadata['NameEfficientB0'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
#model1 = Efficient(metadata['NameEfficientB1'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
#model2 = Efficient(metadata['NameEfficientB2'], metadata['weightsPathEfficient'], N_CLASSES).cuda()
#model3 = Efficient(metadata['NameEfficientB3'], metadata['weightsPathEfficient'], N_CLASSES).cuda()




metadata["BATCH_SIZE"] = BATCH_SIZE_D
metadata["Resize"] = RESIZE_D
metadata["TenCrop"] = TEN_CROP_D
trainloader, _, _, _, _, _ = default_data_loader(metadata)
create_plot(metadata, modelD, trainloader, True)

metadata["BATCH_SIZE"] = BATCH_SIZE_B0
metadata["Resize"] = RESIZE_B0
metadata["TenCrop"] = TEN_CROP_B0
trainloader, _, _, _, _, _ = default_data_loader(metadata)
create_plot(metadata, model0, trainloader, True)

metadata["BATCH_SIZE"] = BATCH_SIZE_B1
metadata["Resize"] = RESIZE_B1
metadata["TenCrop"] = TEN_CROP_B1
trainloader, _, _, _, _, _ = default_data_loader(metadata)
create_plot(metadata, model1, trainloader, True)

metadata["BATCH_SIZE"] = BATCH_SIZE_B2
metadata["Resize"] = RESIZE_B2
metadata["TenCrop"] = TEN_CROP_B2
trainloader, _, _, _, _, _ = default_data_loader(metadata)
create_plot(metadata, model2, trainloader, True)

metadata["BATCH_SIZE"] = BATCH_SIZE_B3
metadata["Resize"] = RESIZE_B3
metadata["TenCrop"] = TEN_CROP_B3
trainloader, _, _, _, _, _ = default_data_loader(metadata)
create_plot(metadata, model3, trainloader, True)



# %%
