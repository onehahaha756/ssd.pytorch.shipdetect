#coding:utf-8

"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import *
import torch.utils.data as data

from ssd import build_ssd
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2 ,glob ,shutil
import warnings
from tqdm import tqdm
import os.path as osp
from eval_casia import casia_eval
from nms import nms
import matplotlib.pyplot as plt
import time
warnings.filterwarnings("ignore")

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

multitype=['.jpg','.tif']
multitype=['.jpg']

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def draw_clsdet(img,cls_dets,vis_thresh):
    '''
    cls_dets:[(x1,y1,x2,y2,score),...]
    return :
    show_img: image with rectangle labels
    '''
    # import pdb;pdb.set_trace()
    for i in range(len(cls_dets)):
        
        bbox=[int(x) for x in cls_dets[i][:-2]]
        x1,y1,x2,y2=bbox
        score=cls_dets[i][-2]
        label=cls_dets[i][-1]
        if score>vis_thresh:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2,2)
            cv2.putText(img,str(score)[:5],(x1,y1),2,cv2.FONT_HERSHEY_PLAIN,(0,255,0),3)
    
    #return show_img

def infer_bigpic(det_file,vis_dir, net, imglist,im_size,nms_thre,overlap=256,thresh=0.05,save_results=True,transform=BaseTransform(512,MEANS)):

    det_results={}
    w,h=im_size,im_size

    print('inferenceing...')
    for imgpath in tqdm(imglist):
        big_im=cv2.imread(imgpath)
        try:
            H,W,C=big_im.shape
        except:
            print('{} is not exsits'.format(imgpath))
            continue
        rows,cols=(H-overlap)//(h-overlap),(W-overlap)//(w-overlap)

        step_h,step_w=(h-overlap),(w-overlap)

        basename=os.path.splitext(os.path.basename(imgpath))[0]

        all_boxes=[]
        imginfer_time=Timer()
        imginfer_time.tic()
        for i in range(rows):
            for j in range(cols):
                im=big_im[i*step_h:i*step_h+h,j*step_w:j*step_w+w,:]
                img=im.copy()
                #img=img.transpose(2,0,1)
                img=torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
                # import pdb;pdb.set_trace()
                x = Variable(img.unsqueeze(0))
                x = x.cuda()
                
                detections = net(x).data
                '''
                detections size:batch*cls*top_k*5(score,x,y,x,y) 
                '''
                #import pdb;pdb.set_trace()
                for cl in range(1, detections.size(1)):
                    dets = detections[0, cl, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    # shift in big pic
                    boxes[:,0] += j*step_w 
                    boxes[:,2] += j*step_w 
                    boxes[:,1] += i*step_h
                    boxes[:,3] += i*step_h
                    #save bboxes
                    no_bg_cls=cl-1 #remove backgroud label
                    cls_label=no_bg_cls*np.ones(boxes.size(0))

                    scores = dets[:, 0].cpu().numpy()
                    #import pdb;pdb.set_trace()
                    #concat results
                    cls_dets = np.hstack((boxes.cpu().numpy(),
                                        scores[:, np.newaxis],cls_label[:,np.newaxis])).astype(np.float32,
                                                                       copy=False)
                    
                    #save results
                    #print('detect subpatch {}/{}'.format(i*cols+j,cols*rows))
                    for num in range(cls_dets.shape[0]):
                        all_boxes.append(cls_dets[num].tolist())
                        #print('detect ship {}, confidence {}\n'.format(cls_dets[num][:-2],nms_bboxes[num][-2]))
        print('detect {} object '.format(len(all_boxes)))
        nms_time=Timer()
        nms_time.tic()
        nms_bboxes=nms(all_boxes,nms_thre,thresh)
        print('nms {} bboxes,use {}s'.format(len(all_boxes)-len(nms_bboxes),nms_time.toc()))
        det_results[basename]=nms_bboxes
        if save_results:
            draw_clsdet(big_im,nms_bboxes,thresh)
            cv2.imwrite('{}/{}.jpg'.format(vis_dir,basename),big_im) 
        print('image detect time :{}'.format(imginfer_time.toc()))
    with open(det_file, 'wb') as f:
        pickle.dump(det_results, f, pickle.HIGHEST_PROTOCOL)

#write inference imagelist
def write_infer_imagenames(imglist,imgnames):
    f=open(imgnames,'w',encoding='utf-8')
    for imgpath in imglist:
        basename=osp.splitext(osp.basename(imgpath))[0]
        f.write('{}\n'.format(basename))
    f.close()

def mksavetree(save_dir):
    '''
    generate save tree:
    det_path: ./detections.pkl
    imagenames: ./infer.imgnames
    vis_results: ./vis_results
    '''
    det_path=osp.join(save_dir,'detections.pkl')
    imagenames=osp.join(save_dir,'infer.imgnames')
    vis_dir=osp.join(save_dir,'vis_results')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        os.mkdir(vis_dir)

    return det_path,imagenames,vis_dir

def eval_results(annot_dir,annot_type,det_path,imagesetfile,clss,overthre,conf_thre,nms_thre):

    rec,prec,ap=casia_eval(annot_dir,annot_type,det_path,imagesetfile,clss,overthre,conf_thre,nms_thre)

    det_dir=osp.dirname(det_path)
    results_path=osp.join(det_dir,'results.txt')
    save_ap_fig=osp.join(det_dir,'AP.png')
    plt.plot(rec,prec)
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.xlabel('recall');plt.ylabel('presicion')
    plt.savefig(save_ap_fig)

    with open(results_path,'w',encoding='utf-8') as f:
        f.write('weights path :{}\n'.format(args.trained_model))
        f.write('iou overthre:{}\nConfidence thre:{}\nnms thresh:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}\n'\
                .format(overthre,conf_thre,nms_thre,ap,rec[-1],prec[-1]))
    f.close()



def get_infer_imagelist(imagedir,test_txt=None,multiimgtype=multitype):
    '''
    return whole image list or subimage list from test_txt
    '''
    imglist=[]
    #import pdb;pdb.set_trace()
    if test_txt!=None:
        with open(test_txt,'r') as f:
            for imagename in f.readlines():
                imagename=imagename.strip()
                for imgtype in multiimgtype:
                    imgpath=osp.join(imagedir,imagename+imgtype)
                    if os.path.exists(imgpath):
                        imglist.append(imgpath)
    else:
        for imgtype in multiimgtype:
            imglist+=glob.glob(osp.join(imagedir,'*'+imgtype))
    
    return imglist
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd512_MixShip_35000.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--data_dir', default='/data/03_Datasets/CasiaDatasets/ship/image',
                        help='Location of inference dataset')
    parser.add_argument('--annot_type', default='rect',choices=['rect','polygon']
                        ,help='for cutimages is rect,for origin annot is polygon')
    parser.add_argument('--test_images', default=None,
                        help='txt test imagenames')
    parser.add_argument('--save_folder', default='GFplane_results_512/', type=str,
                        help='File path to save results')
    parser.add_argument('--conf_thre', default=0.3, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--iou_thre', default=0.3, type=float,
                        help='evalution iou thre ')
    parser.add_argument('--nms_thre', default=0.5, type=float,
                        help='nms iou thre ')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--re_evaluate', default=False, type=str2bool,
                        help='re evaluate results ,but do not infer images')

    args = parser.parse_args()

    #import pdb;pdb.set_trace()
    print(args)
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    cfg=casiaship
    # load net
    net = build_ssd('test',cfg, cfg['min_dim'], cfg['num_classes'])   
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    #imagesetdir
    imagedir=os.path.join(args.data_dir,'image')
    # imagedir=args.data_dir
    annotdir=os.path.join(args.data_dir,'label')

    imglist=get_infer_imagelist(imagedir,args.test_images)
    # import pdb;pdb.set_trace()
    #genarate savefolder
    det_path,imagenames,vis_dir=mksavetree(args.save_folder)
    #write test imglist
    transform=BaseTransform(cfg['min_dim'],MEANS)
    write_infer_imagenames(imglist,imagenames)
    if not args.re_evaluate:
        #shutil.rmtree(args.save_folder)
        infer_bigpic(det_path, vis_dir,net, imglist,cfg['min_dim'],args.nms_thre,thresh=args.conf_thre)
    
    eval_results(annotdir,args.annot_type,det_path,imagenames,cfg['classname'],args.iou_thre,args.conf_thre,args.nms_thre)
