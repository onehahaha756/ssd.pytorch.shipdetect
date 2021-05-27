#coding:utf-8
#from typed_ast.ast3 import arg
import numpy as np
import cv2 ,os,glob,random,shutil

def CasiaTxt2polygons(txt_path,annot_name):
    '''
    casia ship v1 annot format:[[x1,y1,x2,y2,x3,y3,x4,y4],[....]]
    '''
    polygon_list = list()
    txt_file=open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        line=line.replace('[','')
        line=line.replace(']','').replace(' ','')
        row_list=line.split(',')
        polygon=[int(float(x)) for x in row_list[:8]]

        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list


def get_img_polygon_objs(img,polygon_list):
    '''
    get an image polygon object
    input: img,polygon_list
    return: list:[annot_name,[rect_points],object_mask]
    '''
    object_list=[]
    for polygon in polygon_list:
        annot_name,points=polygon 
        rect_array=np.array(points).reshape((-1,2))

        xmin=rect_array[:,0].min()
        xmax=rect_array[:,0].max()
        ymin=rect_array[:,1].min()
        ymax=rect_array[:,1].max()
        w,h=xmax-xmin,ymax-ymin
        # make object area,
        object_img=img[ymin:ymax,xmin:xmax,:]
        object_mask=np.zeros((h,w))
        
        rect_points=[]
        for i in range(0,len(points),2):
            rect_points.append((points[i]-xmin,points[i+1]-ymin))
        rect_points=np.array(rect_points)
        #import pdb;pdb.set_trace()
        #get object polygon mask
        object_mask=cv2.fillPoly(object_mask,[rect_points],1)
        try:
            for i in range(3):
                object_img[:,:,i]=object_mask*object_img[:,:,i]
            object_list.append((annot_name,rect_points,object_img))
        except:
            pass
    return object_list

def PasteObj2img(object_info,img):
    #import pdb;pdb.set_trace()
    h,w,c=img.shape
    object_img=object_info[-1]
    object_name=object_info[0]
    object_points=object_info[1]

    # import pdb;pdb.set_trace()
    img_mask=object_img==0
    ysize,xsize,_=object_img.shape

    x_start=random.randint(0,w-xsize)
    y_start=random.randint(0,h-ysize)

    rect_annot=(object_name,[x_start,y_start,x_start+xsize,y_start+ysize])
    #shift rect_points
    object_points[:,0]+=x_start
    object_points[:,1]+=y_start
    #polygon annot
    polygon_annot=(object_name,object_points.tolist())
    #polygon_annot=[]
    img[y_start:y_start+ysize,x_start:x_start+xsize]=img[y_start:y_start+ysize,x_start:x_start+xsize]*img_mask+object_img
    
    return img,rect_annot,polygon_annot
    


def generate_mix_img(ObjectLists,negative_imgdir,outdir,generate_imgnum,max_paste_num=3,vis_label=True):
    #import pdb;pdb.set_trace()
    negative_imglist=glob.glob(os.path.join(negative_imgdir,'*.tif'))
    save_imgdir=os.path.join(outdir,'image')
    save_labeldir=os.path.join(outdir,'label')
    save_polygonlabeldir=os.path.join(outdir,'labelDota')
    vis_labeldir=os.path.join(outdir,'vis_label')

    if not os.path.exists(save_imgdir):
        os.makedirs(save_imgdir)
    if not os.path.exists(save_labeldir):
        os.makedirs(save_labeldir)
    if not os.path.exists(save_polygonlabeldir):
        os.makedirs(save_polygonlabeldir)

    imgnum=0
    for negative_imgpath in negative_imglist:
        if imgnum>generate_imgnum:
            break
        #import pdb;pdb.set_trace()
        negative_img=cv2.imread(negative_imgpath)
        #save img and annot
        basename=os.path.splitext(os.path.basename(negative_imgpath))[0]
        save_imgpath=os.path.join(save_imgdir,'{}_{}.jpg'.format(imgnum,basename))
        save_txtpath=os.path.join(save_labeldir,'{}_{}.txt'.format(imgnum,basename))
        save_polytxtpath=os.path.join(save_polygonlabeldir,'{}_{}.txt'.format(imgnum,basename))
        
        savefile=open(save_txtpath,'w',encoding='utf-8')
        savepolyfile=open(save_polytxtpath,'w',encoding='utf-8')

        paste_num=random.randint(1,max_paste_num)
        rect_annot_list=[]
        rect_poly_list=[]
        
        for i in range(paste_num):
            object_annot=ObjectLists[random.randint(0,len(ObjectLists)-1)]
            img,rect_annot,polygon_annot=PasteObj2img(object_annot,negative_img)
            
            annot_name,rect=rect_annot
            xmin,ymin,xmax,ymax=rect

            _,poly_points=polygon_annot
            points=[x for point in poly_points for x in point]
            x1,y1,x2,y2,x3,y3,x4,y4=points
            #import pdb;pdb.set_trace()
            rect_annot_list.append((annot_name,[xmin,ymin,xmax,ymax]))
            rect_poly_list.append((annot_name,points))
            savefile.write('{} {} {} {} {}\n'.format(xmin,ymin,xmax,ymax,annot_name))
            savepolyfile.write('{} {} {} {} {} {} {} {} {} {}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,annot_name,0))
        if vis_label:
            if not os.path.exists(vis_labeldir):
                 os.makedirs(vis_labeldir)
            show_img=img.copy()
            save_vispath=os.path.join(vis_labeldir,'{}_{}.jpg'.format(imgnum,basename))
            #show rect annot
            for annot in rect_annot_list:
                annot_name,rect=annot 
                xmin,ymin,xmax,ymax=rect 
                cv2.rectangle(show_img,(xmin,ymin),(xmax,ymax),(0,255,0),2,2)
            #show poly annot
            for polyannot in rect_poly_list:
                annot_name,points=polyannot
                pts=np.array(points).reshape(-1,1,2)
                cv2.polylines(show_img,pts,True,(0,0,255),5)

            cv2.imwrite(save_vispath,show_img)


        print('{}/{}  save {} '.format(imgnum,generate_imgnum,save_imgpath))
        cv2.imwrite(save_imgpath,img)
        savefile.close()
        savepolyfile.close()
        imgnum+=1
           

def get_objs(image_dir,annot_dir):
    annot_imglist=glob.glob(os.path.join(image_dir,'*.jpg'))
    ObjectLists=[]
    proc_num=0
    for annot_imgpath in annot_imglist:
        
        annot_img=cv2.imread(annot_imgpath)
        basename=os.path.splitext(os.path.basename(annot_imgpath))[0]
        annot_path=os.path.join(annot_dir,'{}.txt'.format(basename))

        polygon_list=CasiaTxt2polygons(annot_path,'ship')

        object_list=get_img_polygon_objs(annot_img,polygon_list)
        ObjectLists+=object_list
        print('parse annot file {}/{},get {} objects '.format(proc_num,len(annot_imglist),len(ObjectLists)))
        proc_num+=1
        #import pdb;pdb.set_trace()
    return ObjectLists

def main(image_dir,annot_dir,negative_imgdir,output_dir):
    ObjectLists=get_objs(image_dir,annot_dir)
    print('get {} object masks!'.format(len(ObjectLists)))
    generate_mix_img(ObjectLists,negative_imgdir,output_dir,1000)


if __name__=="__main__":

    import argparse

    parser=argparse.ArgumentParser(description='copy polygon object to paste in another pic')
    parser.add_argument('--image_dir', type=str,
                        help='image directory of annot images')
    parser.add_argument('--annot_dir',type=str, 
                        help='annonations of annot images')
    parser.add_argument('--negative_imgdir', type=str,
                        help='for cutimages is rect,for origin annot is polygon')
    parser.add_argument('--output_dir', type=str,
                        help='output samples directory')

    args = parser.parse_args()
    #import pdb;pdb.set_trace()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    main(args.image_dir,args.annot_dir,args.negative_imgdir,args.output_dir)











