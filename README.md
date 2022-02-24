# X-Ray-Disease-Detection

This repositori contains all the code needed to train, evaluate and test a convolutional neural network combined with different types of data augmentation.


# Folders


|        Fichero/Carpeta        | Función                          |
|----------------|-------------------------------|
|`DenseNet` | This folder contains different script that train DenseNet model taking advantage from the application of transfer learning. All the script present train the same model but applying different types of data augmentation.
|`DenseNet_GPUs` |This folder contains different script that train DenseNet model making use of multiple GPUs and taking advantage from the application of transfer learning. All the script present train the same model but applying different types of data augmentation.      |
|`EfficientBX`          | All the folders which name start with EfficientB contains different script that train EfficientNet model taking advantage from the application of transfer learning. The name of the folder selected indicates which type of EfficientNet is going to be used in the training, being the options B0, B1, B2 and B3. All the script inside a folder train the same model but applying different types of data augmentation.            |
|`EfficientBX_GPUs`          |All the folders which name start with EfficientB and have the ending _GPUs contains different script that train EfficientNet model taking advantage from the application of transfer learning while making use of multiple GPUs. The name of the folder selected indicates which type of EfficientNet is going to be used in the training, being the options B0, B1, B2 and B3. All the script inside a folder train the same model but applying different types of data augmentation. |
|`hparams`          |This folder contains different JSON that will configure how is the training executed. |
|`utils`          |This folder contains scripts that allow the model to save checkpoints, make inference, define all the dataloaders depending on the type of data augmentation selected and LRFinder to analyze our model and the best Learning Rate for the training. |
|`train_test.py`          |This script contains the functions to train, validate and test the model. The diferent functions are apdapted for training in sequential model, training with multiple GPUs, training when making use of TenCrop as data augmenter and training in mixed precision mode.|


# Entrenar modelo
In order to train a model:
1. Create a JSON that determines how has to be the training process 
2. Execute the desired script (for example `RayXDense.py`)  in the following way `python RayXDense.py Config.json` 

The JSON that will determine how will bethe training performed has to have the following parameters:
 ```yaml
{
"weightsPath": Path to weight from trained model,

"DATA_DIR" : Where is the data located,

"TEST_IMAGE_LIST" : test_list.txt or similar file that gives 
information of the images for test pruposes and the diseases 
presented in those images,

"TRAIN_IMAGE_LIST" : train_list.txt or similar file that gives 
information of the images for train pruposes and the diseases 
presented in those images,

"VAL_IMAGE_LIST" : val_list.txt or similar file that gives 
information of the images for train pruposes and the diseases 
presented in those images,

"RESULT_IMAGES" : Path to store the images generated 
when doing inference,

"BATCH_SIZE" : Batch,

"LR" : Learning Rate,

"Resize": Resize of the image,

"TenCrop": Size of the images produced by TenCrop,

"BETAS" : Betas for the optimizer,

"EPOCHS" : Epochs,

"PatiencePlateau": Epoch Patience on Plateau,

"visualization_dataset_workers": number of workers for loading data 
when doing inference,

"training_dataset_workers": number of workers for loading data 
when doing training,

"test_dataset_workers": number of workers for loading data 
when doing test,

"val_dataset_workers": number of workers for loading data 
when doing validation,

"num_gpu":Number of GPUs,

"angle": Angle of rotation of an Image when applying horizontalFlip,

"prob": Probability of apply the selected data augmenation,

"treshold": Threshold applied when applying gausiand sound noise as data augmentation,
}
```
