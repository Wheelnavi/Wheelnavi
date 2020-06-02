from __future__ import absolute_import
import functions.DBface as DBFace
import functions.segment as segment
import functions.FEGAN as FEGAN
import functions.APDrawingGan as APDrawingGAN
from functions.dependency_imports import *
import re_face_preprocessing.CropFace as cropface
import re_face_preprocessing.FaceSwapByMask as faceswapbymask
import re_face_preprocessing
def onetake(originname,mask,stroke,dbface = False,readdat = False):
    if readdat:
        mask = cv2.imread('extra/mask.png')
        stroke = cv2.imread('extra/stroke.png')
    originimageread = cv2.imread('origin/'+originname)
    #please save landmark before this
    if dbface:
        DBFace.detect_singleimage(originimageread,originname)

    make_segment(originimageread,'masks/'+originname)
    sketch_image(originname)
    sketch = cv2.imread('sketch/'+originname)
    FEGAN.execute_FEGAN(mask,sketch,stroke,image = originimageread,read=False)

    #######################################################################

def sketch_image(imagename,foldername='data/rebuild/',savefoldername='data/sketch/',landmarkdir='/data/landmark/',maskdir='/data/segment/'):
    APDrawingGAN.APDrawingGan(foldername,savefoldername,imagename,os.getcwd()+landmarkdir,os.getcwd()+maskdir)

def remove_presketch(originsketch):
    for filename in os.listdir('sketch/'):
        if originsketch in filename:
            os.remove('sketch/'+filename)

def make_segment(originimageread,imagepath):
    segment.segment(originimageread,imagepath)

def do_fegan(mask,sketch,stroke,originimageread,read):
    #already read files
    FEGAN.execute_FEGAN(mask,sketch,stroke,image = originimageread,read=read)

def save_image_to_gcs(user_code,mode_type,originname,originfile):
    blobfile = STORAGE_CLIENT.bucket(GS_BUCKET_NAME).blob(
        'user_' + user_code + '/' + mode_type + '/' + originname).upload_from_filename(originfile)
    return 1

def load_image_from_gcs(user_code,mode_type,originname):
    blobfile = STORAGE_CLIENT.bucket(GS_BUCKET_NAME).blob(
        str('user_' + user_code + '/' + mode_type + '/' + originname))
    if blobfile.exists():
        blobfile.download_to_filename('data/'+mode_type+'/'+str(user_code)+'_'+originname)
        return 1
    return 404

def remove_image_from_local(mode_type,user_code,originname,all=False):
    if all:
        import glob
        for fol in ['mask','stroke','rebuild','sketch','origin']:
            files = glob.glob('data/'+fol+'/*')
            for f in files:
                os.remove(f)
        pass
    else:
        os.remove('data/'+mode_type+'/'+str(user_code)+'_'+originname)
        pass
    return 1

def preprocess(user_code,rebuildimage_rcv,originimages_rcv,fmask_rcv,stroke_rcv):
    #rebuildimages -> to be fixed
    #originimages -> original images
    userimage = str(user_code)+'.png'
    rebuildimage = rebuildimage_rcv.convert('RGB')
    originimages_cvt = []
    for oneimage in originimages_rcv:
        originimages_cvt.append(oneimage.convert('RGB'))
    fmask = fmask_rcv.convert('RGB')
    croppedimg, averageimg, points = cropface.crop_and_average(rebuildimage,originimages_cvt,save_file=False,_pil = True)
    swappedface = faceswapbymask.pil_preprocessing(averageimg,croppedimg,np.array(fmask))
    cv2.imwrite('swappedface.png',swappedface)
    cv2.imwrite('data/origin/'+userimage,averageimg)
    save_image_to_gcs(str(user_code),'origin',str(user_code)+'.png','data/origin'+str(user_code)+'.png')
    cv2.imwrite('data/rebuild/'+userimage,croppedimg)
    save_image_to_gcs(str(user_code),'origin',str(user_code)+'.png','data/origin'+str(user_code)+'.png')

    with open("data/landmark/{}.txt".format(user_code), "w") as f:
        for point in points:
            text = str(point[0])+' '+str(point[1])
            f.write(text)
    originimageread_pil = averageimg.copy()
    make_segment(originimageread_pil.copy(),'data/segment/'+userimage)
    sketch_image(userimage,foldername = 'data/origin/')
    sketch = cv2.imread('data/sketch/'+userimage)
    save_image_to_gcs(user_code,'sketch',userimage,'data/sketch/'+userimage)
    rebuildimg = FEGAN.execute_FEGAN(fmask_rcv.copy(),sketch,stroke_rcv.copy(),userimage,image = originimages_rcv.copy(),read=False)

def onetake_gcs(user_code,originname,dbface = False,readdat = True,origin=False,preprocess = False):
    userimage = user_code+'_'+originname
    if readdat:
        load_image_from_gcs(user_code,'mask',originname)
        mask = cv2.imread('data/mask/'+userimage)
        strokestat = load_image_from_gcs(user_code,'stroke',originname)
        stroke = cv2.imread('data/stroke/'+userimage)
        load_image_from_gcs(user_code,'landmark',os.path.splitext(originname)[0]+'.txt')
        if strokestat == 404:
            stroke = None

    rebuild_path = 'data/rebuild/'+userimage
    load_image_from_gcs(user_code,'rebuild',originname)

    originimageread = cv2.imread(rebuild_path)
    #please save landmark before this
    if dbface:
        DBFace.detect_singleimage(rebuild_path,userimage)
    if preprocess:
        originimageread_pil = preprocess(user_code,originname)
    else:
        if origin:
            load_image_from_gcs(user_code,'origin',originname)
            originimageread_pil = Image.open('data/origin/'+userimage)
        else:
            originimageread_pil = Image.open(rebuild_path)

    make_segment(originimageread_pil,'data/segment/'+userimage)
    if origin:
        sketch_image(userimage,foldername = 'data/origin/')
    else:
        sketch_image(userimage)
    sketch = cv2.imread('data/sketch/'+userimage)
    save_image_to_gcs(user_code,'sketch',originname,'data/sketch/'+userimage)
    rebuildimg = FEGAN.execute_FEGAN(mask,sketch,stroke,userimage,image = originimageread,read=False)
    save_image_to_gcs(user_code,'result',originname,rebuildimg)
    remove_image_from_local(None,user_code,originname,all=True)

