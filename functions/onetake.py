from __future__ import absolute_import
import functions.DBface as DBFace
import functions.segment as segment
import functions.FEGAN as FEGAN
import functions.APDrawingGan as APDrawingGAN
from functions.dependency_imports import *
import re_face_preprocessing.CropFace as cropface
import re_face_preprocessing.FaceSwapByMask as faceswapbymask
import re_face_preprocessing


def sketch_image(imagename, foldername='data/rebuild/', savefoldername='data/sketch/', landmarkdir='/data/landmark/', maskdir='/data/segment/'):
    APDrawingGAN.APDrawingGan(foldername, savefoldername,
                              imagename, os.getcwd()+landmarkdir, os.getcwd()+maskdir)


def remove_presketch(originsketch):
    for filename in os.listdir('sketch/'):
        if originsketch in filename:
            os.remove('sketch/'+filename)


def make_segment(originimageread, imagepath):
    segment.segment(originimageread, imagepath)


def do_fegan(mask, sketch, stroke, originimageread, read):
    # already read files
    FEGAN.execute_FEGAN(mask, sketch, stroke, image=originimageread, read=read)


def save_image_to_gcs(user_code, mode_type, originname, originfile):
    blobfile = STORAGE_CLIENT.bucket(GS_BUCKET_NAME).blob(
        'user_' + user_code + '/' + mode_type + '/' + originname).upload_from_filename(originfile)
    return 1


def load_image_from_gcs(user_code, mode_type, originname):
    blobfile = STORAGE_CLIENT.bucket(GS_BUCKET_NAME).blob(
        str('user_' + user_code + '/' + mode_type + '/' + originname))
    if blobfile.exists():
        blobfile.download_to_filename(
            'data/'+mode_type+'/'+str(user_code)+'_'+originname)
        return 1
    return 404


def remove_image_from_local(mode_type, user_code, originname, all=False):
    if all:
        import glob
        for fol in ['average','detect_results','input','landmark','segment','mask', 'stroke', 'rebuild', 'sketch', 'origin']:
            files = glob.glob('data/'+fol+'/*')
            for f in files:
                os.remove(f)
        pass
    else:
        os.remove('data/'+mode_type+'/'+str(user_code)+'_'+originname)
        pass
    return 1


def preprocess(user_code, rebuildimage_rcv, originimages_rcv, fmask_rcv, stroke_rcv):
    # rebuildimages -> to be fixed
    # originimages -> original images
    userimage = str(user_code)+'.png'

    rebuildimage = rebuildimage_rcv.convert('RGB')
    rebuildimage.save('data/input/'+userimage)
    save_image_to_gcs(str(user_code), 'input', str(
        user_code)+'.png', 'data/input/'+str(user_code)+'.png')

    originimages_cvt = []
    for oneimage in originimages_rcv:
        originimages_cvt.append(oneimage.convert('RGB'))
    fmask = fmask_rcv.convert('RGB')
    cv2.imwrite('data/mask/'+userimage, np.array(fmask).copy())
    fmaskread = cv2.imread('data/mask/'+userimage)

    croppedimg, averageimg, points, landmarks, fmask = cropface.crop_and_average(
        rebuildimage, originimages_cvt, np.array(fmask).copy(), save_file=False, _pil=True)
    swappedface = faceswapbymask.pil_preprocessing(
        averageimg, croppedimg, np.array(fmask).copy())
    # cv2.imwrite('swappedface.png',swappedface)
    cv2.imwrite('data/origin/'+userimage, swappedface)
    save_image_to_gcs(str(user_code), 'origin', str(
        user_code)+'.png', 'data/origin/'+str(user_code)+'.png')
    cv2.imwrite('data/rebuild/'+userimage, croppedimg)
    save_image_to_gcs(str(user_code), 'rebuild', str(
        user_code)+'.png', 'data/rebuild/'+str(user_code)+'.png')
    cv2.imwrite('data/average/'+userimage, averageimg)
    save_image_to_gcs(str(user_code), 'average', str(
        user_code)+'.png', 'data/average/'+str(user_code)+'.png')

    with open("data/landmark/{}.txt".format(user_code), "w") as f:
        for point in points:
            text = str(point[0])+' '+str(point[1])+'\n'
            f.write(text)
    inputimage = cv2.imread('data/input/'+userimage)
    originimageread_pil = Image.open('data/origin/'+userimage)
    make_segment(originimageread_pil, 'data/segment/'+userimage)
    sketch_image(userimage, foldername='data/origin/')
    sketch = cv2.imread('data/sketch/'+userimage)
    save_image_to_gcs(str(user_code), 'sketch',
                      userimage, 'data/sketch/'+userimage)

    rebuildimg, rebuilt = FEGAN.execute_FEGAN(fmask, sketch, stroke_rcv, userimage, image=np.array(croppedimg).copy(), read=False)
    rebuilt = cv2.imread('data/result/'+userimage)
    fmaskread = cv2.imread('data/mask/'+userimage)
    recov_img,newmask = cropface.rotate_scale_origin(inputimage, rebuilt, fmaskread, landmarks)
    cv2.imwrite('data/recover/'+userimage,recov_img)
    save_image_to_gcs(str(user_code),'recover',userimage,'data/recover/'+userimage)
    save_image_to_gcs(str(user_code), 'result', userimage, rebuildimg)
    remove_image_from_local('','','',all=True)
    with open('data/recover/'+userimage, "rb") as f:
        return HttpResponse(f.read(), content_type="image/png")
