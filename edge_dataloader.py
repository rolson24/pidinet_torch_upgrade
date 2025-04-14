from torch.utils import data
import torchvision.transforms as transforms
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        raise ValueError('No images in the data folder')
        return None
    else:
        return allfiles

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    
    # Ensure that all images and labels are torch Tensors.
    images = [img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images]
    labels = [lab if isinstance(lab, torch.Tensor) else torch.from_numpy(lab) for lab in labels]
    
    # Get the maximum height and width in this batch.
    max_h = max(img.size(1) for img in images)
    max_w = max(img.size(2) for img in images)
    
    padded_images = []
    padded_labels = []
    
    for img, lab in zip(images, labels):
        # Pad the image: pad right and bottom to make them consistent in size.
        _, h, w = img.size()
        pad_img = F.pad(img, (0, max_w - w, 0, max_h - h), mode='constant', value=0)
        padded_images.append(pad_img)
        
        # Assuming the labels are shaped as (1, H, W).
        _, h_lab, w_lab = lab.size()
        pad_lab = F.pad(lab, (0, max_w - w_lab, 0, max_h - h_lab), mode='constant', value=0)
        padded_labels.append(pad_lab)
    
    images_tensor = torch.stack(padded_images, dim=0)
    labels_tensor = torch.stack(padded_labels, dim=0)
    return images_tensor, labels_tensor

class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS', split='train', transform=False, threshold=0.3, ablation=False, fixed_size=None):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        self.fixed_size = fixed_size
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        self.to_tensor = transforms.ToTensor()
        
        transform_list = [self.to_tensor, normalize]
        if fixed_size is not None:
            self.resize = transforms.Resize(fixed_size, transforms.InterpolationMode.BILINEAR)
            transform_list.insert(1, self.resize)

        self.transform = transforms.Compose(transform_list)

        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            # lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            lb = Image.open(os.path.join(self.root, lb_file))
            lb = self.to_tensor(lb)

            # print("label shape: ", lb.size())

            # Resize
            if self.fixed_size is not None:
                lb = self.resize(lb)
            
            # print("label shape after reshape: ", lb.size())

            if lb.dim() == 3:
                lb = lb[0, :, :]
            assert lb.dim() == 2

            # print("label shape after dim change: ", lb.size())

                

            threshold = self.threshold
            # lb = lb[np.newaxis, :, :]
            lb = torch.unsqueeze(lb, 0)
            # lb[lb == 0] = 0
            # lb[np.logical_and(lb>0, lb<threshold)] = 2
            # lb[lb >= threshold] = 1
            lb = torch.where(lb > 0, 2, lb)
            lb = torch.where(lb >= threshold, 1, lb)
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3, ablation=False, fixed_size=None):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        self.fixed_size = fixed_size
        print("fixed size: ", self.fixed_size)
        
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
        transform_list = [self.to_tensor, normalize]
        if fixed_size is not None:
            self.resize = transforms.Resize(fixed_size, transforms.InterpolationMode.BILINEAR)
            transform_list.insert(1, self.resize)

        self.transform = transforms.Compose(transform_list)

        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()

            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)

            lb = torch.from_numpy(lb)

            # print("torch label shape: ", lb.size())
            # print("torch dtype: ", lb.dtype)

            if lb.dim() == 3:
                lb = torch.permute(lb, (2, 0, 1))[0, :, :]
            
            lb = torch.unsqueeze(lb, 0)
            assert lb.size()[0] == 1


            # print("label shape after dim change: ", lb.size())

            # Resize
            if self.fixed_size is not None:
                lb = self.resize(lb)
            
            # print("label shape after reshape: ", lb.size())


            threshold = self.threshold
            # lb = torch.unsqueeze(lb, 0)
            # lb = torch.permute(lb, (2, 0, 1))
            lb = torch.where(lb == 0, 0, lb)
            lb = torch.where((lb > 0) & (lb < threshold), 2, lb)
            lb = torch.where(lb >= threshold, 1, lb)
            
            # lb_torch = lb
            # print("torch final label shape: ", lb_torch.shape)

            # lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            # # print("numpy label shape: ", lb.shape)
            # if lb.ndim == 3:
            #     lb = np.squeeze(lb[:, :, 0])
            # assert lb.ndim == 2
            # threshold = self.threshold
            # lb = lb[np.newaxis, :, :]
            # lb[lb == 0] = 0
            # lb[np.logical_and(lb>0, lb<threshold)] = 2
            # lb[lb >= threshold] = 1
            # print("numpy final label shape: ", lb.shape)


            # # get the difference between lb and lb_torch
            # print("sum lb: ", np.sum(lb))
            # print("sum lb torch: ", np.sum(lb_torch.numpy(force=True)))
            # assert np.sum(lb) == np.sum(lb_torch.numpy(force=True))
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        # print("input image shape: ", img.size())

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class Multicue_Loader(data.Dataset):
    """
    Dataloader for Multicue
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        """
        setting[0] should be 'boundary' or 'edge'
        setting[1] should be '1' or '2' or '3'
        """
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name

class NYUD_Loader(data.Dataset):
    """
    Dataloader for NYUDv2
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        """
        There is no threshold for NYUDv2 since it is singlely annotated
        setting should be 'image' or 'hha'
        """
        self.root = root
        self.split = split
        self.threshold = 128
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, '%s-train_da.lst' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, '%s-test.lst' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            img_file, lb_file, scale = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            if scale < 0.99: # which means it < 1.0
                W = int(scale * pil_image.width)
                H = int(scale * pil_image.height)
                pil_image = pil_image.resize((W, H))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            if scale < 0.9:
                img = img.resize((W, H))
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name

class Custom_Loader(data.Dataset):
    """
    Custom Dataloader
    """
    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.imgList = fold_files(os.path.join(root))

    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):

        with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename