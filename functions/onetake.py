from __future__ import absolute_import
import DBface
import segment
import FEGAN
import APDrawingGan
from dependency_imports import *
from google.cloud import storage

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

def sketch_image(imagename,foldername='origin/',savefoldername='sketch/',landmarkdir='landmark/',maskdir='mask/'):
    APDrawingGan.APDrawingGan(foldername,savefoldername,imagename,landmarkdir,maskdir)

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
        'user_' + user_code + '/' + mode_type + '/' + originname).download_to_filename('data/'+mode_type+'/'+str(user_code)+'_'+originname)
    return 1

def remove_image_from_local(mode_type,user_code,originname,all=False):
    if all:
        os.remove('data/mask/'+str(user_code)+'_'+originname)
        os.remove('data/stroke/'+str(user_code)+'_'+originname)
        os.remove('data/landmark/'+str(user_code)+'_'+originname)
        os.remove('data/rebuild/'+str(user_code)+'_'+originname)
    else:
        os.remove('data/'+mode_type+'/'+str(user_code)+'_'+originname)
    return 1

def onetake_gcs(user_code,originname,dbface = False,readdat = True):
    if readdat:
        load_image_from_gcs(user_code,'mask',originname)
        mask = cv2.imread('data/mask/'+user_code+'_'+originname)
        load_image_from_gcs(user_code,'stroke',originname)
        stroke = cv2.imread('data/stroke/'+user_code+'_'+originname)
        load_image_from_gcs(user_code,'landmark',originname)
    
    load_image_from_gcs(user_code,'rebuild',originname)
    originimageread = cv2.imread('data/rebuild/'+user_code+'_'+originname)
    #please save landmark before this
    if dbface:
        DBFace.detect_singleimage(originimageread,originname)

    make_segment(originimageread,'data/masks/'+originname)
    sketch_image(originname)
    sketch = cv2.imread('data/sketch/'+originname)
    rebuildimg = FEGAN.execute_FEGAN(mask,sketch,stroke,user_code+'_'+originname,image = originimageread,read=False)
    save_image_to_gcs(user_code,'rebuilt',originname,rebuildimg)
    remove_image_from_local(all=True)

