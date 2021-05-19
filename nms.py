import os
import cv2
import glob
import numpy as np

def iou(box1,box2,utype=0):
    '''xmin,ymin,xmax,ymax
    utype: 0 for union
    utype: 1 for box1
    utype: 2 for box2'''
    xmin1,ymin1,xmax1,ymax1=box1[0],box1[1],box1[2],box1[3]
    xmin2,ymin2,xmax2,ymax2=box2[0],box2[1],box2[2],box2[3]
    size1=(xmax1-xmin1)*(ymax1-ymin1)
    size2=(xmax2-xmin2)*(ymax2-ymin2)

    xmin=max(xmin1,xmin2)
    ymin=max(ymin1,ymin2)
    xmax=min(xmax1,xmax2)
    ymax=min(ymax1,ymax2)
    i_size=(xmax-xmin)*(ymax-ymin)
    if xmin>=xmax or ymin>=ymax:
        return 0
    if utype == 1:
        return i_size/float(size1)
    if utype == 2:
        return i_size/float(size2)
    return i_size/float(size1+size2-i_size)

def nms(predictions,iou_thre,conf_thre):
    '''
    input:
    predictions:(list),[x1,y1,x2,y2,score,clss],shape[nums_bboxes,6]
    iou_thre: nms overlap threshold
    conf_thre: confidence score to filter
    output:
    nms_bboxes:(list),[x1,y1,x2,y2,score,clss],shape[nums_bboxes,6]
    '''
    # import pdb;pdb.set_trace()
    if len(predictions)<2:
        return np.array(predictions).tolist()
    predictions=np.array(predictions)
    predictions=predictions[predictions[:,-1].argsort()] #sort by classes
    classes_num=predictions[:,-1].max()
    #import pdb;pdb.set_trace()
    nms_bboxes=[]
    clss_bboxes=[]
    for clss in range(int(classes_num+1)): 
        # clss 0 : background 
        clss_predictions=predictions[predictions[...,-1]==clss]
        clss_predictions=clss_predictions[np.argsort(-clss_predictions[...,-2])]
        clss_bboxes=[]
        for i in range(len(clss_predictions)):
            predict=clss_predictions[i]
            bb=predict[:-2]
            score=predict[-2]

            if score<conf_thre:
                continue

            if len(clss_bboxes)==0:
                clss_bboxes.append(predict)
                clss_bboxes=np.array(clss_bboxes)
                continue
            #import pdb;pdb.set_trace()
            ixmin = np.maximum(clss_bboxes[:, 0], bb[0])
            iymin = np.maximum(clss_bboxes[:, 1], bb[1])
            ixmax = np.minimum(clss_bboxes[:, 2], bb[2])
            iymax = np.minimum(clss_bboxes[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (clss_bboxes[:, 2] - clss_bboxes[:, 0]) *
                       (clss_bboxes[:, 3] - clss_bboxes[:, 1]) - inters)

            overlaps = inters / uni
            # 求与之前最大的iou
            if (overlaps<iou_thre).all():
                clss_bboxes=np.vstack((clss_bboxes,predict))
            # else:
            #     print('nms bbox {} ,iou {}\n'.format(predict.tolist(),overlaps.max()))
        if len(nms_bboxes)==0:
            nms_bboxes=clss_bboxes.copy()
            #nms_bboxes=np.a
        else:
            nms_bboxes=np.vstack((nms_bboxes,clss_bboxes))
        nms_bboxes=np.array(nms_bboxes).tolist()
    return nms_bboxes


def softnms(bbox,score,iou_thres,score_thres):
    S=score.copy()
    D=[]
    while len(D)<bbox.shape[0]:
        S[D]=0
        m=np.argmax(S)
        if S[m]<score_thres:
            break
        M=bbox[m]
        D.append(m)
        for i in range(bbox.shape[0]):
            if i not in D and iou(M,bbox[i])>=iou_thres:
                S[i]=S[i]*(1-iou(M,bbox[i]))
    return D


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="post processing of output")
    parser.add_argument('in_dir',nargs='+',help="directories of txt files")
    parser.add_argument('output_dir',help="directory of output")
    parser.add_argument('--mask_dir',help='Directory of the mask images')
    args = parser.parse_args()
    main(args)
