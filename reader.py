import image_util
from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import json

class Settings(object):
    def __init__(self,
                 dataset=None,
                 data_dir=None,
                 label_file=None,
                 resize_h=300,
                 resize_w=300,
                 mean_value=[127.5, 127.5, 127.5],
                 apply_distort=True,
                 apply_expand=True,
                 ap_version='11point'):
        self._dataset = dataset
        self._ap_version = ap_version
        self._data_dir = data_dir
        self._apply_distort = apply_distort
        self._apply_expand = apply_expand
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')
        self._expand_prob = 0.5
        self._expand_max_ratio = 4
        self._hue_prob = 0.5
        self._hue_delta = 18
        self._contrast_prob = 0.5
        self._contrast_delta = 0.5
        self._saturation_prob = 0.5
        self._saturation_delta = 0.5
        self._brightness_prob = 0.5
        self._brightness_delta = 0.125

    @property
    def dataset(self):
        return self._dataset

    @property
    def ap_version(self):
        return self._ap_version

    @property
    def apply_distort(self):
        return self._apply_expand

    @property
    def apply_distort(self):
        return self._apply_distort

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir
    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean

def baidu_star_2018(settings,annotation, mode, shuffle):
    flist = open(annotation)
    annotations=json.load(flist)['annotations'];images=[]
    def reader():
        if mode == 'train' and shuffle:
            random.shuffle(annotations)
        for annotation in annotations:
            image_path = settings.data_dir+'/image/'+annotation['name']
            im = Image.open(image_path);imgs=[]
            images.append(im)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            if im_width==1920 and im_height==1080:
               id_path=settings.data_dir+'/ground_truth/'+str(annotation['id'])+'.npy'
               im=im.resize((im_width/4,im_height/4),Image.ANTIALIAS)
               im=np.array(im)
               if len(im.shape) == 3:
                     im = np.swapaxes(im, 0, 2)
               if os.path.exists(id_path):
                  gt=np.load(id_path)
                  gt=np.transpose(gt)
                  
                  yield im, [gt]
            else:
                continue

    return reader
def train(settings, file_list, shuffle=True):
    file_list = os.path.join(settings.data_dir, file_list)
    return baidu_star_2018(settings, file_list, 'train', shuffle)

def test(settings, file_list):
    file_list = os.path.join(settings.data_dir, file_list)
    return baidu_star_2018(settings, file_list, 'test', False)
def distance(a,b):
    if a.has_key('w') and b.has_key('h'):
        a=[(a['x']+a['w'])*0.5,(a['y']+a['h'])*0.5]
        b=[(b['x']+b['w'])*0.5,(b['y']+b['h'])*0.5]
    elif a.has_key('x') and b.has_key('y'):
        a=[a['x'],a['y']];b=[b['x'],b['y']];
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5;
belta=0.3;
def infer():
    annotations=json.load(open('./annotation/annotation_train_stage2.json'))['annotations']
    img_path=[];
    for a in annotations:
        name=a['name'].replace('stage2/image/','')
        id=a['id']
        points=[]
        img=Image.open('./image/'+a['name']);
        im_width, im_height = img.size
        img = np.array(img)
        # HWC to CHW
        if len(img.shape) == 3:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 1, 0)
        # RBG to BGR
        img = img[[2, 1, 0], :, :]
        img = img.astype('float32')
        img -= 127
        img = img * 0.007843
        ps=a['annotation'];distances=[[] for j in ps]
        for i in range(len(ps)):
               for j in range(len(ps)):
                   if not i==j:
                       distances[i].append(distance(ps[i],ps[j]))
               distances[i].sort()
               di.append(np.mean(distances[i][0:3])*belta)
        yield img,ps,di,id,a['num']
