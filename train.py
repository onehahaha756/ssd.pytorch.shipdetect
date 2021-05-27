from data import *
from utils.augmentations import GFPlaneAugmentation
from layers.modules import MultiBoxLoss,MultiBoxLoss_noobj
from ssd import build_ssd
import os,glob
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from eval_casia import casia_eval
from inference_remote import infer_bigpic,eval_results
import matplotlib.pyplot as plt
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='CasiaShip', choices=['BigShip','CasiaShip','GFPlane','VOC', 'COCO','SSDD'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=Ship_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=28, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=7600, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
#import pdb;pdb.set_trace()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def test_online(weight_path,cfg,loss,eval='cut'):
    basetransform=BaseTransform(cfg['min_dim'],MEANS)
    test_test= build_ssd('test',cfg, cfg['min_dim'], cfg['num_classes'])
    test_test.load_state_dict(torch.load(weight_path))
    test_test.cuda()
    test_test.eval()
    overthre=0.5
    conf_thre=0.1
    nms_thre=0.5
    #weight_path.split('_')[-1]
    #import pdb;pdb.set_trace()
    detpath='eval/detections.pkl'
    visdir='eval'
    save_ap_fig=os.path.join(visdir,'AP_mix_airbus.png')
    try:
        logfile=open('logs/trainlog_airbus.txt','a',encoding='utf-8')
    except:
        logfile=open('logs/trainlog_airbus.txt','w',encoding='utf-8')
    if eval=='cut':
        imgsetfile='/data/03_Datasets/airbus-ship-detection/airbus_ship_detection512/test.txt'
        testdir='/data/03_Datasets/airbus-ship-detection/airbus_ship_detection512/image'
        annot_dir='/data/03_Datasets/airbus-ship-detection/airbus_ship_detection512/label'
        annot_type='rect'
        #imglist=glob.glob(os.path.join(testdir,'*jpg'))
        imgnames=open(imgsetfile,'r') 
        imglist=[os.path.join(testdir,'{}.jpg'.format(x.strip())) for x in imgnames.readlines()][:100]
    else:
        imgsetfile='/data/03_Datasets/CasiaDatasets/CutShip512_300/origin_test.txt'
        testdir='/data/03_Datasets/CasiaDatasets/ship/image/'
        annot_dir='/data/03_Datasets/CasiaDatasets/ship/labelDota'
        annot_type='polygon'   
        imgnames=open(imgsetfile,'r')     
        #import pdb;pdb.set_trace()
        imglist=[os.path.join(testdir,'{}.jpg'.format(x.strip())) for x in imgnames.readlines()]
    test_test.eval()
    #import pdb;pdb.set_trace()
    infer_bigpic(detpath,visdir,test_test,imglist,cfg['min_dim'],0.5,save_results=False,transform=basetransform)
    #eval_results(annot_dir,annot_type,detpath,imgsetfile,'ship',0.5,0.1,0.5)
    rec,prec,ap=casia_eval(annot_dir,annot_type,detpath,imgsetfile,'ship',overthre,conf_thre,nms_thre)
    plt.plot(rec,prec,label=weight_path.split('_')[-1])
    plt.xlabel('recall');plt.ylabel('presicion')
    plt.legend()
    plt.savefig(save_ap_fig)
    logfile.write('time {}\n'.format(time.asctime(time.localtime())))
    logfile.write('*'*15+'\nweights path :{}\n'.format(weight_path))
    logfile.write('loss :{}\n'.format(loss.data.detach()))
    logfile.write('iou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}\n\n'\
                    .format(overthre,conf_thre,ap,rec[-1],prec[-1]))
    logfile.close()



if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    #
    #torch.cuda.set_device(1)
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        #if args.dataset_root == VOC_ROOT:
        #    parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'SSDD':
        print('root is SSDD')
        cfg = ssdd
        dataset = SSDDDetection(root=args.dataset_root,split='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'GFPlane':
        print('root is GFPlane')
        cfg = gfplane
        dataset = GFPlaneDetection(root=args.dataset_root,split='train',
                               transform=GFPlaneAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'CasiaShip':
        print('root is CasiaShip')
        cfg=casiaship
        dataset = ShipDetection(root=args.dataset_root,split='train',
                               transform=GFPlaneAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'BigShip':
        print('root is Bigship')
        cfg=casiaBigship
        dataset = BigShipDetection(root=args.dataset_root,split='train',
                                    transform=GFPlaneAugmentation(cfg['min_dim'],MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train',cfg, cfg['min_dim'], cfg['num_classes'])
    
    #ssd_net = build_ssd('train',cfg, cfg['min_dim'], cfg['num_classes'])
    #import pdb;pdb.set_trace()
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        #import pdb;pdb.set_trace()
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
    #import pdb;pdb.set_trace()
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss_noobj(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    epoch=0
    epoch_iter=0
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch_iter=0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        #import pdb;pdb.set_trace()
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            print('next batch')
            print('batch local loss: {}\nbatch conf loss:{}\n'.format(loc_loss,conf_loss))
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            loc_loss = 0
            conf_loss = 0
            epoch=epoch+1
            epoch_iter=0
            print('\nepoch: {} ,dataset trained finished!'.format(epoch))
        #import pdb;pdb.set_trace()
        if args.cuda:
            images = Variable(images.to("cuda"))
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        #import pdb;pdb.set_trace()
        # forward
        #import pdb;pdb.set_trace()
        epoch_iter+=1
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        #import pdb;pdb.set_trace()
        loss_l, loss_c = criterion(out, targets)
        #import pdb;pdb.set_trace()
        loss = loss_l + loss_c
        #print('')
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' %(loss.item())+\
        #    ' || Loss_l: %.4f' %(loss_l.item())+' || Loss_c: %.4f'%(loss_c.item()))
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('epoch: %d || '%(epoch),' iter ' + repr(iteration) + '  || '+str(epoch_iter)+'/'+str(len(batch_iterator))+' || Loss: %.4f ||' %(loss.item())+\
            ' || Loss_l: %.4f' %(loss_l.item())+' || Loss_c: %.4f'%(loss_c.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            save_path='weights/ssd512_Airbus512_'+repr(iteration) + '.pth'
            torch.save(ssd_net.state_dict(), save_path)
            
            test_online(save_path,cfg,loss,'cut')

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')
# torch.from_n

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )



if __name__ == '__main__':
    train()
