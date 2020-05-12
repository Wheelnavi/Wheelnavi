from functions.dependency_imports import *

from DBFace.model.DBFace import DBFace
import DBFace.common as dbfacecommon
HAS_CUDA = torch.cuda.is_available()
def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = dbfacecommon.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (dbfacecommon.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(dbfacecommon.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)

def facealligner(image, leftEyeCenter,rightEyeCenter,file,desiredLeftEye=(0.35, 0.35),desiredFaceWidth=512, desiredFaceHeight=512):

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)
    cv2.imwrite("detect_result/" +dbfacecommon.file_name_no_suffix(file)+ 'alligned' + ".png", output)

    return output

def detect_image(model, file):

    image = dbfacecommon.imread(file)
    objs = detect(model, image)

    for obj in objs:
        print(obj.landmark)
        f = open('landmarks/'+dbfacecommon.file_name_no_suffix(file)+'.txt','w')
        for land in obj.landmark:
            print(land[0],land[1])
            f.write("{} {}\n".format(int(land[0]),int(land[1])))
        f.close()

        objts = detect(model,facealligner(image,obj.landmark[1],obj.landmark[0],file))
        for objt in objts:
            f = open('landmarks/'+dbfacecommon.file_name_no_suffix(file)+'alligned.txt','w')
            for land in objt.landmark:
                f.write("{} {}\n".format(int(land[0]),int(land[1])))
            f.close()

        dbfacecommon.drawbbox(image, obj)

    dbfacecommon.imwrite("detect_results/" + dbfacecommon.file_name_no_suffix(file) + ".draw.jpg", image)


def image_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("DBFace/model/dbface.pth")
    print('loaded')
    arr = os.listdir("detect_result/")
    for filename in arr:
        print(filename)
        detect_image(dbface, "detect_result/"+filename)

def detect_singleimage(path, name):
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()
    
    dbface.load("DBFace/model/dbface.pth")
    image = dbfacecommon.imread(path)
    objs = detect(dbface, image)

    for obj in objs:
        print(obj.landmark)
        name = os.path.splitext(name)[0]

        f = open('data/landmark/'+name+'.txt','w')
        for land in obj.landmark:
            print(land[0],land[1])
            f.write("{} {}\n".format(int(land[0]),int(land[1])))
        f.close()

        objts = detect(dbface,facealligner(image,obj.landmark[1],obj.landmark[0],name))
        for objt in objts:
            f = open('data/landmark/'+name+'alligned.txt','w')
            for land in objt.landmark:
                f.write("{} {}\n".format(int(land[0]),int(land[1])))
            f.close()

        dbfacecommon.drawbbox(image, obj)
    dbfacecommon.imwrite("data/detect_results/" + name + ".draw.jpg", image)
    return image
