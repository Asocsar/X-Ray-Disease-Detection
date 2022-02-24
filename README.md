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
1. Crear un JSON que determine como ha de ser el entrenamiento del modelo. 
2. Ejecutar el script Main.py de la forma `python Main.py Config.json` 

El JSON que determinará la ejecución debe contener las siguientes palabras claves:
 ```yaml
{
"Parallel": 1 para indicar que se ejecutará de forma paralelizada 0 
para indicar ejecución secuencial,

"Version": 1 para indicar que se quiere hacer uso de los modelos que
no implementan capas Densas 2 para indicar que se quiere hacer uso
de los modelos que haces uso de capas densas,

"BertCalls": Numero de veces que se va a llamar a BERT,

"AdLR_SLTR": 1 para aplicar Adaptative Learning Rate y Slanded 
Triangular Schedule y 0 para no aplicar,

"num_pages": Numero de páginas a introducir al modelo,

"name_efficient": nombre del modelo de efficientnet que se quiere
usar siendo este des de efficientnet-b0 a efficientnet-b7 (el codigo intenta cargar los pesos de efficientnet por lo que será necesario que estos se encuentren presentes en una carpeta llamada Weights),

"num_GPUs": Número de GPUs que se van a usar para entrenar el modelo,

"LR_BERT": Learning Rate que se aplica a BERT,

"LR_EFFICIENT": Learning Rate que se aplica a EfficientNet,

"LR_BASE": Learning Rate que se aplica a todas las capas que no 
pertenezcan ni a BERT ni a EfficientNet,

"directory_h5df_files": Directorio donde se encuentran las carpetas 
train, validation y test que contienen ficheros h5df de BigTobacco,

"directory_tobacco800": Directorio donde se encuentran las carpetas 
train, validation y test que contienen ficheros h5df de Tobacco800,

"BATCH": Batch que se desea usar,

"workers": Workers usados en la carga del dataset,

"EPOCH": Número de Epochs,

"decay_factor": Decay factor aplicado al LR de BERT (la formula
aplicada es LR_BERT*decay_factor^(k) entendiendo k como la 
profundidad en capas)

"BETAS": Betas aplicadas al optimizador,

"num_features": Numero de neuronas densas para procesar la salida
obtenida de EfficientNet (solo usado en modelos que hacen uso de capa 
Densa),

"feature_concatenation": Numero de neuronas densas para procesar
la concatenación de las llamadas a BERT y EfficientNet(solo usado 
en modelos que hacen uso de capa Densa).
}
```


# Testear modelo con Tobacco800
Para hacer el test de un modelo con Tobaco800 los pasos ha seguir han de ser los siguientes:

1.  Seleccionar el mismo JSON que se ha usado para entrenar el modelo llamando al fichero Main.py.
2.  Ejecutar el script  test_tobacco800.py  de la forma  `python test_tobacco800.py [args]`. Los argumentos que se pueden introducir en la llamada son los siguientes:
	* `tobacco800_conf`: Indicar el fichero JSON usado para el entrenamiento (NECESARIO) [string].
	* `select_epoch`: Indicar la epoch de la cual se desea cargar el checkpoint, de no estar presente cogera la epoch con mejor resultado [integer].
	* `filtered`: Indica si el checkpoint que se quiere cargar es de un modelo obtenido de entrenar con BigTobacco después de filtrar los datos, por defecto esta en falso [bool].
	* `fine_tune`: Indica si se quiere usar una porcion de Tobacco800 para entrenar y validar el modelo, por defecto esta en falseo [bool].
	* `full_train`: Indica si se quiere coger un modelo y sin cargar ningun checkpoint hacer un entrenamiento, validación y test en Tobacco800. Los parametros usados serían escogidos a partir del JSON, por defecto esta en falseo [bool].


# Crear un dataset
En caso de que fuera necesario los scripts dentro de la carpeta llamada CreateDataset sirven para crear un dataset dadas imagenes y ficheros .txt distribuidas en diversas carpetas.

Para crear un dataset los scripts se deben ejecutar en el siguiente orden:
1. creation_sublist.py: Para ejecutar este fichero se debe llamar de la forma `python create_sublist.py [tobacco800]`. La llamada registrará todos los documentos, entendiendose como estos todas las imagenes dentro de una unica carpeta, y lo dividirá en 4 listas separadas, esto se hace para más tarde tratar estas 4 listas de forma paralela, una vez creadas las 4 listas se guardaran en 4 ficheros .txt que mas tarde se podrán leer. El argumento tobacco800 cuando se indica como 1, estamos indicando que no queremos crear 4 sublistas, sino una unica lista, esto sirve si el dataset no es muy grande.
2. CreationOCR_parallel.py: Para ejecutar este fichero se debe llamar de la forma `python create_sublist.py [agrs]` y los argumentos de entrada son los siguientes:
	* iden: Indica la lista que se debe leer, este valor puede oscilar de 0 a 3 [string] (NECESARIO).
	* BigTobacco: Indicamos si estamos creando ficheros para BigTobacco [bool].
	* Tobacco800: Indicamos si estamos creando ficheros para Tobacco800 [bool].
3. create_H5DF.py: Para ejecutar este fichero se debe llamar de la forma `python createHDF5.py [agrs]` y los argumentos de entrada son los siguientes:
	* `filtering`: Indica si debe o no filtrar los datos de entrenamiento [bool].
	* `visualize_data`: Indica si crea o no plots mostrando la distribución de los datos de entrenamiento [bool].
	* `create_json_information`: Indica si debe o no crear un JSON con toda la información de los documentos respecto al balanceo de clases [bool].
	* `mode`: Indica si aplica modo "Tail" o "Head" en el momento de filtrar los datos. Es necesario solo si se indica `filtering`.
	* `tobacco800`: Indica si se estan creando ficheros H5DF relacionados con tobaco800 [bool]. 
	* `splitTobacco800`: Indica si debe o no dividir los datos del dataset de tobacco800 entre entrenamiento validación y test [bool].
	* `trainT800`: Indica porcion de entrenamiento de tobacco800 en caso de dividir el dataset [bool].
	* `valT800`: Indica porcion de validación de tobacco800 en caso de dividir el dataset [bool].
	* `testT800`: Indica porcion de test de tobacco800 en caso de dividir el dataset [bool].
