import scipy.io as sio
from PIL import Image
from torch.utils import data as torchdata
import os
import torch
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image
import PIL
import random

from data.segbase import SegmentationDataset

num_classes = 21
ignore_label = 255
root = '/media/b3-542/LIBRARY/Datasets/VOC'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items

class TrainDataset_VOC_old(torchdata.Dataset):
    def __init__(self, image_set, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = '/home/csc302/workspace/liuhy/pytorch-semantic-segmentation/datasets'
        self.imgSize = [512, 512]
        # down sampling rate of segm label
        self.segm_downsampling_rate = 1
        self.random_flip = True
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 8

        # classify images into two classes: 1. h > w and 2. h <= w
        # self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[1., 1., 1.])
            ])

        # voc dataset file name and folder name check
        year = image_set.split('_')[0]
        self.image_set = image_set[len(year) + 1: len(image_set)]
        self.year = year
        self.devkit_path = self.root_dataset
        self.data_path = os.path.join(self.devkit_path, 'VOC' + year)

        self.list_sample = self.load_image_set_index()
        self.num_sample = len(self.list_sample)
        print('num_images', self.num_sample)

        # self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        self.if_shuffled = False
        # if max_sample > 0:
        #     self.list_sample = self.list_sample[0:max_sample]
        # self.num_sample = len(self.list_sample)
        # assert self.num_sample > 0
        # print('# samples: {}'.format(self.num_sample))

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _read_img(self, index, batch_resized_size):
        # load image and label
        image_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        segm_path = os.path.join(self.data_path, 'SegmentationClass', index + '.png')
        img = Image.open(image_path)
        segm = Image.open(segm_path)

        assert(img.mode == 'RGB')
        assert(segm.mode == 'P')
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        # Random uniform crop
        # PIL.Image.size: (width, height)
        w, h = img.size

        target_w = w + max(0, self.imgSize[0] - w)
        target_h = h + max(0, self.imgSize[1] - h)

        processed_img = Image.new('RGB', (target_w, target_h), (127, 127, 127))
        processed_segm = Image.new('P', (target_w, target_h), 255)

        # make the new segm palette == old segm palette, for the visual
        processed_segm.putpalette(segm.getpalette())

        processed_img.paste(img)
        processed_segm.paste(segm)

        x1 = random.randint(0, target_w - self.imgSize[0])
        y1 = random.randint(0, target_h - self.imgSize[1])

        processed_img = processed_img.crop((x1, y1, x1 + self.imgSize[0], y1 + self.imgSize[1]))
        processed_segm = processed_segm.crop((x1, y1, x1 + self.imgSize[0], y1 + self.imgSize[1]))

        # Random horizontal flip
        if self.random_flip:
            if random.random() < 0.5:
                processed_img = processed_img.transpose(Image.FLIP_LEFT_RIGHT)
                processed_segm = processed_segm.transpose(Image.FLIP_LEFT_RIGHT)

        #processed_img = self.img_transform(torch.from_numpy(processed_img.copy()))
        #processed_segm = self.img_transform(torch.from_numpy(processed_segm.copy())

        # Convert to tensors
        w, h = processed_img.size
        processed_img = torch.ByteTensor(torch.ByteStorage.from_buffer(processed_img.tobytes())).view(h, w, 3).permute(2, 0, 1).float()
        processed_segm = torch.ByteTensor(torch.ByteStorage.from_buffer(processed_segm.tobytes())).view(h, w).long()

        processed_img = processed_img.add(-127)

        return processed_img, processed_segm

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        # if not self.if_shuffled:
        #     np.random.shuffle(self.list_sample)
        #     self.if_shuffled = True

        batch_images = torch.zeros(3, self.imgSize[0], self.imgSize[1])
        batch_segms = torch.zeros(self.imgSize[0] // self.segm_downsampling_rate, \
                                  self.imgSize[1] // self.segm_downsampling_rate).long()

        img, segm = self._read_img(self.list_sample[index], self.imgSize)
        batch_images[:, :, :] = img
        batch_segms[:, :] = segm

        # batch_images[:, :img.shape[1], :img.shape[2]] = img
        # batch_segms[:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        # update current sample pointer
        # self.cur_idx += self.batch_per_gpu
        # if self.cur_idx >= self.num_sample:
        #     self.cur_idx = 0
        #     np.random.shuffle(self.list_sample)

        # output = dict()
        # output['img_data'] = batch_images
        # output['seg_label'] = batch_segms
        #print(output)
        # return output
        return batch_images, batch_segms

    def __len__(self):
        return self.num_sample # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class TrainDataset_VOC(SegmentationDataset):
    BASE_DIR = 'VOC2012'
    def __init__(self, dataset_root='', mode='train', transform=None):
        super(TrainDataset_VOC, self).__init__(dataset_root)
        self.root = dataset_root
        # _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(self.root, 'SegmentationClass')
        _image_dir = os.path.join(self.root, 'images')
        self.transform = transform
        self.train = mode

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets/Segmentation')
        if self.train == 'train':
            #_split_f = os.path.join(_splits_dir, 'train.txt')
            _split_f = os.path.join(_splits_dir, 'train.txt')

        elif self.train == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.train == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.train != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.train != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.train == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.train == 'train':
            img, target = self._sync_transform(img, target)
        elif self.train == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.train))

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img, dtype=np.int32)).float()

        target = torch.from_numpy(np.array(target, dtype=np.int32)).long()

        return img, target

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')

class TrainDataset_VOC_simple(torchdata.Dataset):
    def __init__(self, image_set, max_sample=-1, batch_per_gpu=1, is_train=True):
        # self.root_dataset = '/home/csc302/workspace/liuhy/pytorch-semantic-segmentation/datasets'
        self.root_dataset = '/home/liuhy/workspace/pytorch-semantic-segmentation/datasets'
        self.imgSize = [512, 512] # height, width
        # down sampling rate of segm label
        self.segm_downsampling_rate = 1
        self.random_flip = True
        self.is_train = is_train
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 8


        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0


        # voc dataset file name and folder name check
        year = image_set.split('_')[0]
        self.image_set = image_set[len(year) + 1: len(image_set)]
        self.year = year
        self.devkit_path = self.root_dataset
        self.data_path = os.path.join(self.devkit_path, 'VOC' + year)

        self.list_sample = self.load_image_set_index()
        self.num_sample = len(self.list_sample)
        print('num_images', self.num_sample)


        self.if_shuffled = False

        # self.mean_bgr = np.array([0.485, 0.456, 0.406])
        self.mean_bgr = np.array([0.485, 0.456, 0.406])
        self.str = np.array([0.229, 0.224, 0.225])



    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Segmentation', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def __getitem__(self, index):
        # load image and label
        image_path = os.path.join(self.data_path, 'JPEGImages', self.list_sample[index] + '.jpg')
        segm_path = os.path.join(self.data_path, 'SegmentationClass', self.list_sample[index] + '.png')
        img = Image.open(image_path)
        lbl = Image.open(segm_path)

        assert(img.mode == 'RGB')
        assert(lbl.mode == 'P')
        assert(img.size[0] == lbl.size[0])
        assert(img.size[1] == lbl.size[1])

        img = img.resize(self.imgSize, PIL.Image.BILINEAR)
        lbl = lbl.resize(self.imgSize, PIL.Image.NEAREST)

        # im2arr.shape: height x width x channel
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.int32)

        # lbl[lbl == 255] = -1

        return self.transform(img, lbl)

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr

        # if self.is_train:
            # # random crop
            # h, w, _ = img.shape
            # target_h = h + max(0, self.imgSize[0] - h)
            # target_w = w + max(0, self.imgSize[1] - w)
            #
            # crop_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            # crop_lbl = np.ones((target_h, target_w), dtype=np.int32) * 255
            #
            # y1 = random.randint(0, target_h - self.imgSize[0])
            # x1 = random.randint(0, target_w - self.imgSize[1])
            #
            # crop_img = img[y1:y1 + self.imgSize[0], x1:x1+self.imgSize[1], :]
            # crop_lbl[y1:y1 + self.imgSize[0], x1:x1+self.imgSize[1]] = lbl



        #     crop_img = crop_img.transpose(2, 0, 1)
        #
        #     crop_img = torch.from_numpy(crop_img).float()
        #     crop_lbl = torch.from_numpy(crop_lbl).long()
        #
        #     return crop_img, crop_lbl
        #
        # else:

        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        #lbl = lbl.numpy()
        return Image.fromarray(img)

    def __len__(self):
        return self.num_sample


class ValDataset_VOC(torchdata.Dataset):
    def __init__(self, image_set, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = '/home/csc302/workspace/liuhy/pytorch-semantic-segmentation/datasets'
        self.imgSize = [256, 256]
        # down sampling rate of segm label
        self.segm_downsampling_rate = 1
        self.random_flip = True
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 8

        # classify images into two classes: 1. h > w and 2. h <= w
        # self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[127, 127, 127], std=[1., 1., 1.])
            ])

        # voc dataset file name and folder name check
        year = image_set.split('_')[0]
        self.image_set = image_set[len(year) + 1: len(image_set)]
        self.year = year
        self.devkit_path = self.root_dataset
        self.data_path = os.path.join(self.devkit_path, 'VOC' + year)

        self.list_sample = self.load_image_set_index()
        self.num_sample = len(self.list_sample)
        print('num_images', self.num_sample)

        # self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        self.if_shuffled = False
        # if max_sample > 0:
        #     self.list_sample = self.list_sample[0:max_sample]
        # self.num_sample = len(self.list_sample)
        # assert self.num_sample > 0
        # print('# samples: {}'.format(self.num_sample))

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _read_img(self, index, batch_resized_size):
        # load image and label
        image_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        segm_path = os.path.join(self.data_path, 'SegmentationClass', index + '.png')
        img = imread(image_path, mode='RGB')
        segm = imread(segm_path, mode='P')

        assert(img.ndim == 3)
        assert(segm.ndim == 2)
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])

        if self.random_flip == True:
            random_flip = np.random.choice([0, 1])
            if random_flip == 1:
                img = cv2.flip(img, 1)
                segm = cv2.flip(segm, 1)

        ##  note that each sample within a mini batch has different scale param
        img = imresize(img, (batch_resized_size[0], batch_resized_size[1]), interp='bilinear')
        segm = imresize(segm, (batch_resized_size[0], batch_resized_size[1]), interp='nearest')

        # to avoid seg label misalignment
        segm_rounded_height = round2nearest_multiple(batch_resized_size[0], self.segm_downsampling_rate)
        segm_rounded_width = round2nearest_multiple(batch_resized_size[1], self.segm_downsampling_rate)
        segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
        segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

        segm = imresize(segm_rounded, (segm_rounded.shape[0] // self.segm_downsampling_rate, \
                                       segm_rounded.shape[1] // self.segm_downsampling_rate), \
                        interp='nearest')
         # image to float
        img = img.astype(np.float32)[:, :, ::-1] # RGB to BGR!!!
        img = img.transpose((2, 0, 1))
        img = self.img_transform(torch.from_numpy(img.copy()))

        return img, segm

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        batch_images = torch.zeros(3, self.imgSize[0], self.imgSize[1])
        batch_segms = torch.zeros(self.imgSize[0] // self.segm_downsampling_rate, \
                                  self.imgSize[1] // self.segm_downsampling_rate).long()

        img, segm = self._read_img(self.list_sample[index], self.imgSize)
        batch_images[:, :img.shape[1], :img.shape[2]] = img
        batch_segms[:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        # update current sample pointer
        # self.cur_idx += self.batch_per_gpu
        # if self.cur_idx >= self.num_sample:
        #     self.cur_idx = 0
        #     np.random.shuffle(self.list_sample)

        # output = dict()
        # output['img_data'] = batch_images
        # output['seg_label'] = batch_segms
        #print(output)
        # return output
        return batch_images, batch_segms

    def __len__(self):
        return self.num_sample # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class VOC(torchdata.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
