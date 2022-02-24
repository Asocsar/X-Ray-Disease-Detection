import time
import torch
import numpy as np
import torch.nn as nn
from datetime import timedelta
import datetime
import os
import time
from os import listdir
from utils.utils import *
from pathlib import Path
from torch.autograd import Variable
from torch.cuda.amp import autocast

def train(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog):
    loss = 0.0
    train_start = time.time()
    print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 500 == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')


        

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        outputs = model(input_var)
        outputs = outputs.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.cuda()
        target = target.cuda()
        loss_batch = loss_func(outputs, target)
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch.item() * inp.size(0)
        batch_end = time.time()
        elapsed_time = batch_end - batch_start
        elapsed_time = timedelta(seconds=elapsed_time)
        #print("Elapsed Time 1 batch", elapsed_time)


    
    loss = loss / train_len
    train_time =  time.time() - train_start
    train_time = timedelta(seconds=train_time)
    print('Train finished in', train_time)
    print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

    TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/train', loss , epoch)



def val(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog):
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Validating Model...')
    for i, (inp, target, weights) in enumerate(val_loader):

        if i % 250 == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        with torch.no_grad():
            output = model(input_var).view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        torch.cuda.empty_cache()

    
    loss = loss / val_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Validation finished in', test_time)
    print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/val', loss , epoch)

    print('[%d] loss: %.3f' % (epoch + 1, loss  ))

    return AUROC_avg, loss



def test(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, TensorLog):
    if not "CheckPointLoad" in metadata.keys():
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f))]
        if len(onlyfiles) > 0:
            model.load_state_dict(torch.load(onlyfiles[0]), strict=False)
    
    else:
        model.load_state_dict(torch.load(metadata["CheckPointLoad"]), strict=False)


    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Testing Model...')
    for i, (inp, target, weights) in enumerate(test_loader):

        if i % 250 == 0:
            p = int((i/len(test_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        with torch.no_grad():
            output = model(input_var).view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        torch.cuda.empty_cache()
    
    loss = loss / test_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Test finished in', test_time)
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_hparams({"Test_Accuracy": AUROC_avg, "Test_Loss": loss}, {"Test_Accuracy": AUROC_avg, "Test_Loss": loss})
    #paths = sorted(Path(metadata['TensorBoardFold']).iterdir(), key=os.path.getmtime)[-2]
    #os.remove(paths)




def train_NC(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog):
    loss = 0.0
    
    train_start = time.time()
    print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 500 == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        outputs = model(inp.cuda())
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.cuda()
        target = target.cuda()
        loss_batch = loss_func(outputs, target)
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch.item() * inp.size(0)
        batch_end = time.time()
        batch_time =  batch_end - batch_start
        train_time = timedelta(seconds=batch_time)
        #print('Batch finished in', train_time)
    
    loss = loss / train_len
    train_time =  time.time() - train_start
    train_time = timedelta(seconds=train_time)
    print('Train finished in', train_time)
    print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

    TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/train', loss , epoch)



def val_NC(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog):
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Validating Model...')
    for i, (inp, target, weights) in enumerate(val_loader):

        if i % 250 == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        with torch.no_grad():
            output = model(inp.cuda())
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, inp
        torch.cuda.empty_cache()
    
    loss = loss / val_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Validation finished in', test_time)
    print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/val', loss , epoch)

    print('[%d] loss: %.3f' % (epoch + 1, loss))

    return AUROC_avg, loss



def test_NC(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, TensorLog):
    if not "CheckPointLoad" in metadata.keys():
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f))]
        if len(onlyfiles) > 0:
            model.load_state_dict(torch.load(onlyfiles[0]), strict=False)
    
    else:
        model.load_state_dict(torch.load(metadata["CheckPointLoad"]), strict=False)


    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Testing Model...')
    for i, (inp, target, weights) in enumerate(test_loader):
        if i % 250 == 0:
            p = int((i/len(test_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCELoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        with torch.no_grad():
            output = model(inp.cuda())
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, inp
    
    loss = loss / test_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Test finished in', test_time)
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_hparams({"Test_Accuracy": AUROC_avg, "Test_Loss": loss}, {"Test_Accuracy": AUROC_avg, "Test_Loss": loss})



def train_distributed(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog, device, rank):
    loss = 0.0
    train_start = time.time()
    if rank == 0:
        print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    train_loader.sampler.set_epoch(epoch)
    for i, (inp, target, weights) in enumerate(train_loader):
        if i % 500 == 0 and rank == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        

        loss_func = nn.BCELoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).to(device)
        outputs = model(input_var)
        outputs = outputs.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.to(device)
        target = target.to(device)
        loss_batch = loss_func(outputs, target)
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch.item() * inp.size(0)
    
    if rank == 0:
        loss = loss / train_len
        train_time =  time.time() - train_start
        train_time = timedelta(seconds=train_time)
        print('Train finished in', train_time)
        print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

        TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/train', loss , epoch)





def val_distributed(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog, device, rank):
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    loss = 0.0
    test_start = time.time()
    if rank == 0:
        print('Validating Model...')

    for i, (inp, target, weights) in enumerate(val_loader):
        if i % 250 == 0 and rank == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCELoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).to(device)
        with torch.no_grad():
            output = model(input_var).view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        torch.cuda.empty_cache()
    
    loss = loss / val_len

    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    
    if rank == 0:
        test_time = time.time() - test_start
        test_time = timedelta(seconds=test_time)
        print('Validation finished in', test_time)
        print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
        
        
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
        
        TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/val', loss , epoch)
        print('[%d] loss: %.3f' % (epoch + 1, loss  ))

    return AUROC_avg, loss


def train_NC_Mixed(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog):
    loss = 0.0
    
    train_start = time.time()
    print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    scaler = torch.cuda.amp.GradScaler()

    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 500 == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inp.cuda())
            loss_batch = loss_func(outputs, target)
        
        outputs = torch.sigmoid(outputs)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.cuda()
        target = target.cuda()
        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        loss += loss_batch.item() * inp.size(0)
        
        #batch_end = time.time()
        #batch_time =  batch_end - batch_start
        #train_time = timedelta(seconds=batch_time)
        #print('Batch finished in', train_time)
    
    loss = loss / train_len
    train_time =  time.time() - train_start
    train_time = timedelta(seconds=train_time)
    print('Train finished in', train_time)
    print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

    TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/train', loss , epoch)


def val_NC_Mixed(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog):
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Validating Model...')
    for i, (inp, target, weights) in enumerate(val_loader):
        if i % 250 == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        with autocast():
            with torch.no_grad():
                output = model(inp.cuda())
                loss_batch = loss_func(output, target)
        
        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss += loss_batch.item() * inp.size(0)
        del output, inp
        torch.cuda.empty_cache()
    
    loss = loss / val_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Validation finished in', test_time)
    print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/val', loss , epoch)

    print('[%d] loss: %.3f' % (epoch + 1, loss))

    return AUROC_avg, loss

def test_NC_Mixed(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, TensorLog):
    if not "CheckPointLoad" in metadata.keys():
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f))]
        if len(onlyfiles) > 0:
            model.load_state_dict(torch.load(onlyfiles[0]), strict=False)
    
    else:
        model.load_state_dict(torch.load(metadata["CheckPointLoad"]), strict=False)


    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Testing Model...')
    for i, (inp, target, weights) in enumerate(test_loader):
        if i % 250 == 0:
            p = int((i/len(test_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        with autocast():
            with torch.no_grad():
                output = model(inp.cuda())
                loss_batch = loss_func(output, target)
        
        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss += loss_batch.item() * inp.size(0)
        del output, inp
    
    loss = loss / test_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Test finished in', test_time)
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_hparams({"Test_Accuracy": AUROC_avg, "Test_Loss": loss}, {"Test_Accuracy": AUROC_avg, "Test_Loss": loss})



def train_Mixed(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog):
    loss = 0.0
    train_start = time.time()
    print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    scaler = torch.cuda.amp.GradScaler()

    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 500 == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        with autocast():
            outputs = model(input_var)
            outputs = outputs.view(bs, n_crops, -1).mean(1)
            loss_batch = loss_func(outputs, target)

        outputs = torch.sigmoid(outputs)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.cuda()
        target = target.cuda()

        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()


        loss += loss_batch.item() * inp.size(0)
        batch_end = time.time()
        elapsed_time = batch_end - batch_start
        elapsed_time = timedelta(seconds=elapsed_time)
        #print("Elapsed Time 1 batch", elapsed_time)
    
    loss = loss / train_len
    train_time =  time.time() - train_start
    train_time = timedelta(seconds=train_time)
    print('Train finished in', train_time)
    print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

    TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/train', loss , epoch)



def val_Mixed(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog):
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Validating Model...')
    for i, (inp, target, weights) in enumerate(val_loader):

        if i % 250 == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        with autocast():
            with torch.no_grad():
                output = model(input_var).view(bs, n_crops, -1).mean(1)

        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        torch.cuda.empty_cache()
    
    loss = loss / val_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Validation finished in', test_time)
    print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
    TensorLog.add_scalar('Loss/val', loss , epoch)

    print('[%d] loss: %.3f' % (epoch + 1, loss  ))

    return AUROC_avg, loss



def test_Mixed(metadata, model, test_loader, test_len, CLASS_NAMES, N_CLASSES, TensorLog):
    if not "CheckPointLoad" in metadata.keys():
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f))]
        if len(onlyfiles) > 0:
            model.load_state_dict(torch.load(onlyfiles[0]), strict=False)
    
    else:
        model.load_state_dict(torch.load(metadata["CheckPointLoad"]), strict=False)


    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    loss = 0.0
    test_start = time.time()
    print('Testing Model...')
    for i, (inp, target, weights) in enumerate(test_loader):

        if i % 250 == 0:
            p = int((i/len(test_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.cuda())
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).cuda()
        with autocast():
            with torch.no_grad():
                output = model(input_var).view(bs, n_crops, -1).mean(1)

        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        torch.cuda.empty_cache()
    
    loss = loss / test_len
    test_time = time.time() - test_start
    test_time = timedelta(seconds=test_time)
    print('Test finished in', test_time)
    
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    
    TensorLog.add_hparams({"Test_Accuracy": AUROC_avg, "Test_Loss": loss}, {"Test_Accuracy": AUROC_avg, "Test_Loss": loss})


def train_Mixed_distributed(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog, device, rank):
    loss = 0.0
    train_start = time.time()
    if rank == 0:
        print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)


    scaler = torch.cuda.amp.GradScaler()

    train_loader.sampler.set_epoch(epoch)
    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 500 == 0 and rank == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        

        loss_func = nn.BCEWithLogitsLoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).to(device)
        with autocast():
            outputs = model(input_var)
            outputs = outputs.view(bs, n_crops, -1).mean(1)
            loss_batch = loss_func(outputs, target)

        outputs = torch.sigmoid(outputs)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.to(device)
        target = target.to(device)

        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()


        loss += loss_batch.item() * inp.size(0)
        batch_end = time.time()
        elapsed_time = batch_end - batch_start
        elapsed_time = timedelta(seconds=elapsed_time)
        #print("Elapsed Time 1 batch", elapsed_time)
    
    if rank == 0:
        loss = loss / train_len
        train_time =  time.time() - train_start
        train_time = timedelta(seconds=train_time)
        print('Train finished in', train_time)
        print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

        TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/train', loss , epoch)

    del gt, pred


def val_Mixed_distributed(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog,  device, rank):
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    loss = 0.0
    test_start = time.time()
    if rank == 0:
        print('Validating Model...')

    for i, (inp, target, weights) in enumerate(val_loader):

        if i % 250 == 0 and rank == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = inp.view(-1, c, h, w).to(device)
        with autocast():
            with torch.no_grad():
                output = model(input_var).view(bs, n_crops, -1).mean(1)

        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss_batch = loss_func(output, target)
        loss += loss_batch.item() * inp.size(0)
        del output, input_var
        #torch.cuda.empty_cache()
    
    loss = loss / val_len

    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    
    if rank == 0:
        test_time = time.time() - test_start
        test_time = timedelta(seconds=test_time)
        print('Validation finished in', test_time)
        print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
        
        
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
        
        TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/val', loss , epoch)
        print('[%d] loss: %.3f' % (epoch + 1, loss  ))

    del gt, pred

    return AUROC_avg, loss



def train_Mixed_distributed_NC(model, CLASS_NAMES, N_CLASSES, epoch, train_loader, train_len, optimizer, TensorLog, device, rank):
    loss = 0.0
    
    train_start = time.time()
    if rank == 0:
        print('Training Model...')

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    scaler = torch.cuda.amp.GradScaler()

    train_loader.sampler.set_epoch(epoch)
    for i, (inp, target, weights) in enumerate(train_loader):
        batch_start = time.time()
        if i % 100 == 0 and rank == 0:
            p = int((i/len(train_loader))*100000)/1000
            print('Progress', p, '%')
        
        #print(inp.size())
        

        loss_func = nn.BCEWithLogitsLoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inp.to(device))
            loss_batch = loss_func(outputs, target)


        outputs = torch.sigmoid(outputs)
        pred = torch.cat((pred, outputs), 0)
        outputs = outputs.to(device)
        target = target.to(device)

        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()


        loss += loss_batch.item() * inp.size(0)
        batch_end = time.time()
        batch_time =  batch_end - batch_start
        train_time = timedelta(seconds=batch_time)
        #print('Batch finished in', train_time)
    
    if rank == 0:
        loss = loss / train_len
        train_time =  time.time() - train_start
        train_time = timedelta(seconds=train_time)
        print('Train finished in', train_time)
        print('Epoch [%d] Train loss: %.3f' % (epoch + 1, loss ))
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

        TensorLog.add_scalar('Accuracy/train', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/train', loss , epoch)

    del gt, pred

def val_Mixed_distributed_NC(model, CLASS_NAMES, N_CLASSES, epoch, val_loader, val_len, optimizer, TensorLog,  device, rank):
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    loss = 0.0
    test_start = time.time()
    if rank == 0:
        print('Validating Model...')
    for i, (inp, target, weights) in enumerate(val_loader):
        if i % 100 == 0 and rank == 0:
            p = int((i/len(val_loader))*100000)/1000
            print('Progress', p, '%')

        loss_func = nn.BCEWithLogitsLoss(weights.to(device))
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        with autocast():
            with torch.no_grad():
                output = model(inp.to(device))
                loss_batch = loss_func(output, target)
        
        output = torch.sigmoid(output)
        pred = torch.cat((pred, output), 0)
        loss += loss_batch.item() * inp.size(0)
        del output, inp

    
    loss = loss / val_len
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    
    if rank == 0:
        test_time = time.time() - test_start
        test_time = timedelta(seconds=test_time)
        print('Validation finished in', test_time)
        print('Epoch [%d] Validation loss: %.3f' % (epoch + 1, loss  ))
        
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
        
        TensorLog.add_scalar('Accuracy/val', AUROC_avg, epoch)
        TensorLog.add_scalar('Loss/val', loss , epoch)

        print('[%d] loss: %.3f' % (epoch + 1, loss))

    del gt, pred

    return AUROC_avg, loss
