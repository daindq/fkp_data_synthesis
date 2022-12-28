import opendatasets as od
import argparse
import os
import shutil
from PIL import Image
import requests


def remove_folder_content(folder):
    if os.listdir(folder) == []:
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
    return None    


def get_args():
    parser = argparse.ArgumentParser(description="Build dataset.")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["MNIST", "PolyUHKV1"])
    parser.add_argument("-p", "--phase", type=str, required=True, choices=["train", "evaluate"])
    parser.add_argument("-e", "--extension", type=str, default="bmp", choices=["jpg", "bmp"])
    return parser.parse_args()


def build_PolyUHKV1(args, dirs):
    data_raw_dir, train_path, dev_path, test_path = dirs
    phase = args.phase
    od.download('https://www.kaggle.com/datasets/nguyendqdai/fkp-polyu-hk-ver1'
                , data_dir=data_raw_dir)
    if args.extension == "jpg":
        dirformatted = 'fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major_jpg'
        if not(os.path.exists(f'{data_raw_dir}/{dirformatted}')):
            os.makedirs(f'{data_raw_dir}/{dirformatted}')
        for item in os.listdir(f'{data_raw_dir}/fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major'):
            img = Image.open(f'{data_raw_dir}/fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major/{item}')
            filename = item.split('.')[0]
            img.save(f'{data_raw_dir}/{dirformatted}/{filename}.jpg')            
    else:
        dirformatted = 'fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major'
    if phase == "evaluate":
        for item in os.listdir(f'{data_raw_dir}/{dirformatted}'):
            shutil.copyfile(f'{data_raw_dir}/{dirformatted}/{item}', f'{test_path}/{item}')
    else:
        for item in os.listdir(f'{data_raw_dir}/{dirformatted}'):
            idx = item.split('_')[0]
            if not(os.path.exists(f'{train_path}/index_{idx}')):
                os.makedirs(f'{train_path}/index_{idx}')
            if not(os.path.exists(f'{dev_path}/index_{idx}')):
                os.makedirs(f'{dev_path}/index_{idx}')
            shutil.copyfile(f'{data_raw_dir}/{dirformatted}/{item}', f'{train_path}/index_{idx}/{item}')    
    return None


def download_MNIST(filename, data_raw_directory):
    r = requests.get(f'http://yann.lecun.com/exdb/mnist/{filename}', allow_redirects=True)
    open(f'{data_raw_directory}/train-images-idx3-ubyte.gz', 'wb').write(r.content)
    
    return None


def build_MNIST(args, data_raw_directory):
    
    r = requests.get('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', allow_redirects=True)
    open(f'{data_raw_directory}/train-images-idx3-ubyte.gz', 'wb').write(r.content)
    return None


# nguyendqdai - f56dd07d7a52ae4dab0008b690945b87
if __name__ == "__main__":
# Get args...
    args = get_args()
    
# Pre-proccess data directory
    data_path = "./data/processed"
    data_raw_dir = f'./data/raw'
    train_path = f'{data_path}/train'
    dev_path = f'{data_path}/dev'
    test_path = f'{data_path}/test'
    for _, dir in enumerate([train_path, dev_path, test_path]):
        if os.path.exists(dir):
            for item in os.listdir(dir):
                remove_folder_content(f'{dir}/{item}')
                shutil.rmtree(f'{dir}/{item}')
            remove_folder_content(dir)                
        else:
            os.mkdir(dir) 
            
# Build dataset...
    if args.dataset == "PolyUHKV1":
        dirs = (data_raw_dir, train_path, dev_path, test_path)
        build_PolyUHKV1(args, dirs)
        
    

