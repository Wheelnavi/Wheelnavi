from __future__ import absolute_import
import functions.DBface as DBFace
import functions.segment as segment
import functions.FEGAN as FEGAN
import functions.APDrawingGan as APDrawingGAN
from functions.dependency_imports import *

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
        os.remove('data/mask/'+str(user_code)+'_'+originname)
        os.remove('data/stroke/'+str(user_code)+'_'+originname)
        #os.remove('data/landmark/'+str(user_code)+'_'+originname)
        os.remove('data/rebuild/'+str(user_code)+'_'+originname)
    else:
        os.remove('data/'+mode_type+'/'+str(user_code)+'_'+originname)
    return 1

def onetake_gcs(user_code,originname,dbface = False,readdat = True):
    userimage = user_code+'_'+originname
    if readdat:
        load_image_from_gcs(user_code,'mask',originname)
        mask = cv2.imread('data/mask/'+userimage)
        strokestat = load_image_from_gcs(user_code,'stroke',originname)
        stroke = cv2.imread('data/stroke/'+userimage)
        load_image_from_gcs(user_code,'landmark',originname)
        if strokestat == 404:
            stroke = None
    
    rebuild_path = 'data/rebuild/'+userimage
    load_image_from_gcs(user_code,'rebuild',originname)
    originimageread = cv2.imread(rebuild_path)
    originimageread_pil = Image.open(rebuild_path)
    #please save landmark before this
    if dbface:
        DBFace.detect_singleimage(rebuild_path,userimage)
    make_segment(originimageread_pil,'data/segment/'+userimage)
    sketch_image(userimage)
    sketch = cv2.imread('data/sketch/'+userimage)
    save_image_to_gcs(user_code,'sketch',originname,'data/sketch/'+userimage)
    rebuildimg = FEGAN.execute_FEGAN(mask,sketch,stroke,userimage,image = originimageread,read=False)
    save_image_to_gcs(user_code,'result',originname,rebuildimg)
    remove_image_from_local(None,user_code,originname,all=True)

