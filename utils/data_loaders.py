from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import os
import torch
import numpy as np
import PIL
import torchvision
import math


def TenCrop_to_Tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def Normalize_TenCrop(crops):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    return torch.stack([normalize(crop) for crop in crops])


def default_data_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)



    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def default_data_loader_FiveCrop(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.FiveCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.FiveCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)



    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.FiveCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def default_data_loader_NDA(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)



    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def default_distributed_data_loader(rank, metadata):



    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (TenCrop_to_Tensor),
                                        transforms.Lambda
                                        (Normalize_TenCrop)
                                    ]))
    
    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=metadata['num_gpu'],
                                        shuffle=True,
                                        rank=rank
                                        )
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=metadata['BATCH_SIZE'],
                             shuffle=False, num_workers=metadata['training_dataset_workers'], pin_memory=True, persistent_workers=True)
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (TenCrop_to_Tensor),
                                        transforms.Lambda
                                        (Normalize_TenCrop)
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True, persistent_workers=True)



    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (TenCrop_to_Tensor),
                                        transforms.Lambda
                                        (Normalize_TenCrop)
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    


    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Flip_data_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Flip_data_loader_NC(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=False, prefetch_factor=metadata['BATCH_SIZE'])
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def RandomFlip_data_loader_NC(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomHorizontalFlip(p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=False)
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    print("Len Train", len(train_dataset), "Len Val", len(val_dataset), "Len Test", len(test_dataset))
    print("Total Length", len(train_dataset) + len(test_dataset) + len(val_dataset))
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)



def RandomFlip_data_loader_NC_distributed(rank, metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomHorizontalFlip(p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))

    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=metadata['num_gpu'],
                                        shuffle=True,
                                        rank=rank
                                        )
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=metadata['BATCH_SIZE'],
                             shuffle=False, num_workers=metadata['training_dataset_workers'], pin_memory=True, persistent_workers=True)



    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True, persistent_workers=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    

    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)

def RandomFlip_Rot_data_loader_NC(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomHorizontalFlip(p=metadata['prob']),
                                        transforms.RandomApply([transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)



def RandomFlip_Sound_data_loader_NC(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomHorizontalFlip(p=metadata['prob']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)



def Crop_Flip_data_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    rotation = transforms.RandomRotation(metadata['angle'])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.RandomApply([
                                        transforms.Lambda
                                        (lambda crops: [rotation(crop) for crop in crops])], p=metadata['prob']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Flip_distributed_data_loader(rank, metadata):

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([torchvision.transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=metadata['num_gpu'],
                                        shuffle=True,
                                        rank=rank
                                        )
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=metadata['BATCH_SIZE'],
                             shuffle=False, num_workers=metadata['training_dataset_workers'], pin_memory=True)



    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)

    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Sound_data_loader(metadata):
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)

def Crop_Sound_data_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    salt_and_pepper = SaltAndPepperNoise(treshold=metadata['treshold'])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.RandomApply([
                                        transforms.Lambda
                                        (lambda crops: [salt_and_pepper(crop) for crop in crops])], p=metadata['prob']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)


    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)

def Sound_distributed_data_loader(rank, metadata):
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=metadata['num_gpu'],
                                        shuffle=True,
                                        rank=rank
                                        )
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=metadata['BATCH_SIZE'],
                             shuffle=False, num_workers=metadata['training_dataset_workers'], pin_memory=True)
    

    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Sound_Flip_data_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                         transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.RandomApply([torchvision.transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Sound_Flip_data_loader_NC(metadata):

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                         transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.RandomApply([torchvision.transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['training_dataset_workers'], pin_memory=True)


    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


def Sound_Flip_distributed_data_loader(rank, metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TRAIN_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.RandomApply([SaltAndPepperNoise(treshold=metadata['treshold'])], p=metadata['prob']),
                                        transforms.RandomApply([torchvision.transforms.RandomRotation(metadata['angle'])], p=metadata['prob']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=metadata['num_gpu'],
                                        shuffle=True,
                                        rank=rank
                                        )
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=metadata['BATCH_SIZE'],
                             shuffle=False, num_workers=metadata['training_dataset_workers'], pin_memory=True)
            
    val_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['VAL_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    val_loader = DataLoader(dataset=val_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['val_dataset_workers'], pin_memory=True)
    
    test_dataset = ChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=transforms.Compose([
                                        transforms.Resize(metadata['Resize']),
                                        transforms.TenCrop(metadata['TenCrop']),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))


    test_loader = DataLoader(dataset=test_dataset, batch_size=metadata['BATCH_SIZE'],
                             shuffle=True, num_workers=metadata['test_dataset_workers'], pin_memory=True)
    
    return train_loader, len(train_dataset), test_loader, len(test_dataset), val_loader, len(val_dataset)


class SaltAndPepperNoise(object):
    def __init__(self,
                 treshold = 0.005,
                 imgType = "PIL",
                 lowerValue = 5,
                 upperValue = 250,
                 noiseType = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img_o):
        if self.imgType == "PIL":
            img = np.array(img_o)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        
        

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            P = PIL.Image.fromarray(img)
            return P


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):

        epsilon = 1e-10
        image_names = []
        labels = []
        self.probs = [0 for _ in range(14)]
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                image_name = image_name[:-3] + 'jpg'
                label = items[1:]
                label = [int(i) for i in label]
                
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                self.probs = [e1+e2 for e1,e2 in zip(self.probs,label)]
        
        self.probs = [math.log10(k/len(labels) + epsilon) for k in self.probs]
        positivos = [sum(x) for x in zip(*labels)]
        num_pos = sum(positivos)
        negativos = [len(labels) - x for x in positivos]
        num_neg = sum(negativos)


        self.pos_weight = len(labels)/(num_pos + len(labels))
        self.neg_weight = num_neg/(num_neg + len(labels))

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        label_inverse = [1 - x for x in label]
        weights = []
        for (L,LI) in zip(label,label_inverse):
            weights.append(max(L*self.pos_weight, LI*self.neg_weight))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.FloatTensor(weights)

    def __len__(self):
        return len(self.image_names)



