from functions.dependency_imports import *

fegan = __import__('SC-FEGAN.utils.config')

from functions.model import Model



class Ex:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model.load_demo_graph(config)

        self.output_img = None

        self.mat_img = None


    def open(self,image = None,fileName=None):
        if fileName:
            mat_img = cv2.imread(fileName)
            # redbrush = QBrush(Qt.red)
            # blackpen = QPen(Qt.black)
            # blackpen.setWidth(5)


            mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
            mat_img = mat_img/127.5 - 1
            self.mat_img = np.expand_dims(mat_img,axis=0)
            print(mat_img)
            plt.imshow(mat_img, interpolation='nearest')
            plt.show()
        elif image is not None:
            mat_img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
            mat_img = mat_img/127.5 - 1
            self.mat_img = np.expand_dims(mat_img,axis=0)



    def color_change_mode(self):
        self.dlg.exec_()
        self.color = self.dlg.currentColor().name()
        self.pushButton_4.setStyleSheet("background-color: %s;" % self.color)
        self.scene.get_stk_color(self.color)

    def complete(self,mask,sketch,stroke,read):
        mask = self.make_mask(mask,read)
        sketch = self.make_sketch(sketch,read)
        stroke = self.make_stroke(stroke,read)

        noise = self.make_noise()

        sketch = sketch*mask
        stroke = stroke*mask
        noise = noise*mask

        batch = np.concatenate(
                    [self.mat_img,
                     sketch,
                     stroke,
                     mask,
                     noise],axis=3)
        start_t = time.time()
        result = self.model.demo(self.config, batch)
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        result = (result+1)*127.5
        result = np.asarray(result[0,:,:,:],dtype=np.uint8)
        self.output_img = result
        plt.show()

        result = np.concatenate([result[:,:,2:3],result[:,:,1:2],result[:,:,:1]],axis=2)
        plt.imshow(result, interpolation='nearest')

        plt.show()
        return self.output_img


    def make_noise(self):
        noise = np.zeros([512, 512, 1],dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise/255,dtype=np.uint8)
        noise = np.expand_dims(noise,axis=0)
        return noise

    def make_mask(self, masksrc, read=True):
        if masksrc is not None:
            if read:
                src = cv2.imread(masksrc)
            else:
                src = masksrc
            src = cv2.resize(src, (512, 512), interpolation=cv2.INTER_CUBIC)
            dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.bitwise_not(dst)

            dst = np.repeat(dst[:, :, np.newaxis], 3, axis=2)
            plt.imshow(dst, interpolation='nearest')
            plt.show()

            mask = np.asarray(dst[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        else:
            mask = np.zeros((512,512,3))
            mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        return mask

    def make_sketch(self, sketchsrc, read=True): #black line
        if sketchsrc is not None:
            if read:
                src = cv2.imread(sketchsrc)
            else:
                src = sketchsrc
            src = cv2.resize(src, (512, 512), interpolation=cv2.INTER_CUBIC)
            dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.bitwise_not(dst)

#            dst = (dst != 0)
            #dst = np.repeat(dst[:, :, np.newaxis], 3, axis=2)
            print(dst)
            sketch = np.asarray(dst[:,:]/255,dtype=np.uint8)
            plt.imshow(dst, interpolation='nearest')
            plt.show()
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        else:
            sketch = np.zeros((512,512,3))
            # sketch = 255*sketch
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        return sketch

    def make_stroke(self, strokesrc, read=True): #RGB format
        if strokesrc is not None:
            stroke = np.zeros((512,512,3))
            if read:
                src = cv2.imread(strokesrc,flags =cv2.IMREAD_COLOR)
            else:
                src = strokesrc
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            src = cv2.bitwise_not(src)

            src = cv2.resize(src, (512, 512), interpolation=cv2.INTER_CUBIC)

            
            plt.imshow(src, interpolation='nearest')
            plt.show()
            stroke = src/127.5 - 1

            stroke = np.expand_dims(stroke,axis=0)
            print(stroke)

        else:
            stroke = np.zeros((512,512,3))
            stroke = stroke/127.5 - 1
            stroke = np.expand_dims(stroke,axis=0)

        return stroke


def execute_FEGAN(mask,sketch,stroke,imagename,image=None,read=True):
    config = fegan.utils.config.Config('functions/config.yaml')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_NUM)
    model = Model(config)

    ex = Ex(model, config)
    if image is not None:
        ex.open(image = image)
    rebuildimg = ex.complete(mask,sketch,stroke,read)
    cv2.imwrite('data/result/'+imagename,rebuildimg)
    return 'data/result/'+imagename
