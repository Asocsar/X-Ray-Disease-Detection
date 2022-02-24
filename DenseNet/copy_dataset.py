from os import listdir
from os.path import isfile, join
import os
from shutil import copyfile
import subprocess
import os

def copy_dataset(metadata):
    data_directory = metadata["DATA_DIR"]

    bashCommand = "echo $NVME1DIR"
    new_directory = subprocess.check_output(bashCommand, shell=True, universal_newlines=True)[:-1]


    new_directory = os.path.join(new_directory, "bsc31168_temp")

    if not os.path.isdir(new_directory):
        os.mkdir(new_directory)

    for f in listdir(data_directory):
        if isfile(os.path.join(data_directory, f)):
            copyfile(os.path.join(data_directory, f), os.path.join(new_directory, f))
    
    metadata["DATA_DIR"] = new_directory
