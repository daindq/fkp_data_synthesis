import opendatasets as od
import argparse
import os
import shutil
from PIL import Image
import requests
import cv2
import numpy as np
import os


def sharp(input_image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(input_image, kernel_size, sigma)
    sharpened = float(amount + 1) * input_image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(input_image - blurred) < threshold
        np.copyto(sharpened, input_image, where=low_contrast_mask)
    return sharpened


def crop_image(row, img):
    img = np.array(img, dtype=np.uint8)
    img_crop = img[row - 300: row, :]
    return img_crop


def step_2(img_path):
    # Get ox axis of local co-ordinate (y=y0)
    img = cv2.imread(img_path, 0)   # Grayscale
    origin = img
    y0 = img.shape[0]
    img_crop = img  # no crop
    # img_crop = crop_image(y0 - 15, img)
    return img_crop, y0, origin


def scale(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val)  # 0-1
    return new_img


def step_3(img, min_val, k_size=3):
    # 3. Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k_size)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k_size)

    edge_gradient = np.sqrt(Gx * Gx + Gy * Gy)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    # round angle to 4 directions
    angle = np.abs(angle)
    for x in range(angle.shape[0]):
        for y in range(angle.shape[1]):
            if (angle[x, y] <= 22.5) or (angle[x, y] >= 157.5):
                angle[x, y] = 0
            elif (angle[x, y] > 22.5) and (angle[x, y] < 67.5):
                angle[x, y] = 45
            elif (angle[x, y] >= 67.5) and (angle[x, y] <= 112.5):
                angle[x, y] = 90
            else:
                angle[x, y] = 135  # 4. Non-maximum Suppression
    canny_raw = np.zeros(img.shape, np.uint8)
    for x in range(1, edge_gradient.shape[0] - 1):
        for y in range(1, edge_gradient.shape[1] - 1):
            if angle[x, y] == 0:
                if edge_gradient[x, y] <= max(edge_gradient[x, y - 1], edge_gradient[x, y + 1]):
                    edge_gradient[x, y] = 0
            elif angle[x, y] == 45:
                if edge_gradient[x, y] <= max(edge_gradient[x + 1, y - 1], edge_gradient[x - 1, y + 1]):
                    edge_gradient[x, y] = 0
            elif angle[x, y] == 90:
                if edge_gradient[x, y] <= max(edge_gradient[x - 1, y], edge_gradient[x + 1, y]):
                    edge_gradient[x, y] = 0
            elif angle[x, y] == 135:
                if edge_gradient[x, y] <= max(edge_gradient[x - 1, y - 1], edge_gradient[x + 1, y + 1]):
                    edge_gradient[x, y] = 0
    # 5. Hysteresis Thresholding
    canny_mask = np.zeros(img.shape, np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if edge_gradient[x, y] >= min_val:
                canny_mask[x, y] = 255
    return scale(canny_mask), canny_mask


def step_4(ie):
    img = ie
    y_mid = img.shape[0] / 2
    icd = np.zeros((img.shape[0], img.shape[1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if (int(img[i, j]) == 0) or (int(img[i + 1, j - 1]) == 1 and int(img[i + 1, j + 1] == 1)):
                icd[i, j] = 0
            elif ((int(img[i + 1, j - 1]) == 1) and (i <= y_mid)) or (int(img[i + 1, j + 1]) == 1 and i > y_mid):
                icd[i, j] = 1
            elif ((int(img[i + 1, j + 1]) == 1) and (i <= y_mid)) or (int(img[i + 1, j - 1]) == 1 and i > y_mid):
                icd[i, j] = -1
    return icd


def calc_con_mag(x, icd):
    res = np.absolute(np.sum(icd[:, x - 17:x + 18]))    # window shape = n_height x 35 pixels
    return res


def step_5(icd):
    conMag = np.ones(icd.shape[1])
    conMag *= 1000
    for i in range(17, icd.shape[1] - 18):
        conMag[i] = calc_con_mag(i, icd)
    x_0 = np.argmin(conMag)
    return x_0, conMag


def step_6(origin, x_0, y_0, width):
    if (x_0 - width/2) < 0:
        x_0 = int(width/2)
    if (x_0 + width/2) > origin.shape[1]:
        x_0 = int(origin.shape[1] - width/2)
    IROI = origin[:, int(x_0 - width/2):int(x_0 + width/2)]
    return IROI


def extract_roi(raw_path, roi_path):
    image = cv2.imread(raw_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = sharp(image)
    cv2.imwrite(roi_path, image)
    img_crop, y0, origin = step_2(roi_path)
    ie, canny_mask = step_3(img_crop, 10)
    icd1 = step_4(ie)
    x0, con_mag = step_5(icd1)
    roi = step_6(origin, x0, y0, image.shape[0])
    roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(roi_path, roi)
    return None


def extract_roi_all(inputfolder, outputfolder):
    for item in os.listdir(inputfolder):
        extract_roi(f'{inputfolder}/{item}', f'{outputfolder}/{item}')
    return None





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
    parser.add_argument("-c", "--crop", type=bool, default=False, choices=[True, False])
    parser.add_argument("-e", "--extension", type=str, default="bmp", choices=["jpg", "bmp"])
    return parser.parse_args()


def build_PolyUHKV1(args, dirs):
    data_raw_dir, train_path, dev_path, test_path = dirs
    square_dir = 'fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major_square'
    phase = args.phase
    od.download('https://www.kaggle.com/datasets/nguyendqdai/fkp-polyu-hk-ver1'
                , data_dir=data_raw_dir)
    if not(os.path.exists(f'{data_raw_dir}/{square_dir}')):
        os.mkdir(f'{data_raw_dir}/{square_dir}')
    if args.crop:
        extract_roi_all(f'{data_raw_dir}/fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major', f'{data_raw_dir}/{square_dir}')
    else:
        square_dir = 'fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major'
    if args.extension == "jpg":
        dirformatted = 'fkp-polyu-hk-ver1/FKP_PolyU_HK/Segmented/Major_jpg'
        if not(os.path.exists(f'{data_raw_dir}/{dirformatted}')):
            os.makedirs(f'{data_raw_dir}/{dirformatted}')
        for item in os.listdir(f'{data_raw_dir}/{square_dir}'):
            img = Image.open(f'{data_raw_dir}/{square_dir}/{item}')
            filename = item.split('.')[0]
            img = img.resize((64,64))
            img.save(f'{data_raw_dir}/{dirformatted}/{filename}.jpg')            
    else:
        dirformatted = square_dir
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
