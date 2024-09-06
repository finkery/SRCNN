import argparse
import h5py
import glob
import numpy as np
from PIL import Image as pil_image
import rasterio

def train(args):
    h5_file = h5py.File(args.output_path,'w')

    lr_patchs = []
    hr_patchs = []

    for image_path in sorted(glob.glob('{}/*'.format(args.image_dir))):
        with rasterio.open(image_path) as src:
            cellsize = src.res[0]  # 取像元大小
        hr = pil_image.open(image_path)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width,hr_height),resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale,hr.height // args.scale),resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        for i in range(0,hr.shape[0] - args.patch_size + 1,args.stride):
            for j in range(0,hr.shape[1] - args.patch_size + 1,args.stride):
                lr_patchs.append((lr[i // args.scale:i // args.scale + args.patch_size // args.scale,j // args.scale:j // args.scale + args.patch_size // args.scale],cellsize))
                hr_patchs.append((hr[i:i + args.patch_size,j:j + args.patch_size],cellsize))
    
    lr_patchs = np.array(lr_patchs)
    hr_patchs = np.array(hr_patchs)

    h5_file.create_dataset('lr',data = lr_patchs)
    h5_file.create_dataset('hr',data = hr_patchs)

    h5_file.close()

def eval(args):
    h5_file = h5py.File(args.output_path,'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i,image_path in enumerate(sorted(glob.glob('{}/*'.format(args.image_dir)))):
        with rasterio.open(image_path) as src:
            cellsize = src.res[0]  # 取像元大小
        hr = pil_image.open(image_path)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width,hr_height),resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale,hr_height // args.scale),resample=pil_image.BICUBIC)
        hr = (np.array(hr).astype(np.float32),cellsize)
        lr = (np.array(lr).astype(np.float32),cellsize)

        hr_group.create_dataset(str(i),data=hr)
        lr_group.create_dataset(str(i),data=lr)
    
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir',type=str,required=True)
    parser.add_argument('--output-path',type=str,required=True)
    parser.add_argument('--patch-size',type=int,default=36)
    parser.add_argument('--stride',type=int,default=14)
    parser.add_argument('--scale',type=int,default=2)
    parser.add_argument('--eval',action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)