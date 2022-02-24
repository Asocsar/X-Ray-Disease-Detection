from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import cv2


def inference(metadata, model, CLASS_NAMES):

    visual_loader =  create_visual_loader(metadata)

    if not os.path.exists(metadata['RESULT_IMAGES']):
        os.makedirs(metadata['RESULT_IMAGES'])

    model.eval()
    heatmap_output = []
    image_id = []
    output_class = []
    gcam = GradCAM(model=model, cuda=True)
    for i, (imageT, raw_image, image_name) in enumerate(visual_loader):
        with torch.no_grad():
            n_crops, c, h, w = imageT.size()
            input_img = torch.autograd.Variable(imageT.view(-1, c, h, w).cuda())

        probs = gcam.forward(input_img)
        activate_classes = np.where((probs > [0.6]*14)[0]==True)[0]


        for activate_class in activate_classes:
            with torch.no_grad():
                gcam.backward(idx=activate_class)
                #efficient._conv_head.static_padding <- E0/E3
                #densenet121.features.denseblock4.denselayer16.conv2 <- Dense
                output = gcam.generate(target_layer="densenet121.features.denseblock4.denselayer16.conv2")
                name = 'Image {} disease {}.png'.format(image_name[0].split('/')[-1].split('.')[0], CLASS_NAMES[activate_class])
                if np.sum(np.isnan(output)) == 0:
                    gcam.save(os.path.join(metadata['RESULT_IMAGES'], name), output, raw_image)
        
        #del output, probs, input_img
        torch.cuda.empty_cache()


#%%
def create_visual_loader(metadata):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
    visual_dataset = VisualChestXrayDataSet(data_dir=metadata['DATA_DIR'],
                                    image_list_file=metadata['TEST_IMAGE_LIST'],
                                    transform=(transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ]),
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                    ])))
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1,
                             shuffle=False, num_workers=metadata['visualization_dataset_workers'], pin_memory=True)
    return visual_loader


#%%
class VisualChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):

        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)

        self.image_names = image_names
        self.transform1 = transform[0]
        self.transform2 = transform[1]

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform1 is not None:
            imageT = self.transform1(image)

        if self.transform2 is not None:
            raw_image = self.transform2(image)
        return imageT, raw_image, image_name

    def __len__(self):
        return len(self.image_names)


#%%
class PropagationBase(object):

    def __init__(self, model, cuda=True):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(image)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


#%%
class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()


        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data
        
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_RAINBOW)
        raw_image = raw_image.float().numpy()[0]
        shape = raw_image.shape
        raw_image = np.ascontiguousarray(raw_image.transpose(1,2,0))

        im1 = Image.fromarray(np.uint8(raw_image*255))
        im2 = Image.fromarray(np.uint8(gcam))

        Im =  Image.blend(im1, im2, 0.4) 
        Im.save(filename)