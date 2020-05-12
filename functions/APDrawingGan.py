from functions.dependency_imports import *
sys.path.append('APDrawingGAN')
from APDrawingGAN.util import util,html,visualizer
from APDrawingGAN.options import test_options
import APDrawingGAN.data as apdata
import APDrawingGAN.models as apmodel
import traceback

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open('loggg.txt', 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Options(test_options.TestOptions):
    def __init__(self,parser):
        self.parser = parser
        super().initialize(self.parser)

    def gather_options(self):
        # initialize parser with basic optiona
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            parser = self.parser

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = apmodel.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = apdata.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_known_args()[0]

def remove_option(parser, arg):
    for action in parser._actions:
        if (vars(action)['option_strings']
            and vars(action)['option_strings'][0] == arg) \
                or vars(action)['dest'] == arg:
            parser._remove_action(action)

    for action in parser._action_groups:
        vars_action = vars(action)
        var_group_actions = vars_action['_group_actions']
        for x in var_group_actions:
            if x.dest == arg:
                var_group_actions.remove(x)
                return

def APDrawingGan(imagepath,savepath,imagename,landmarkdir,maskdir):
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opt = Options(parser)
    remove_option(opt.parser,'dataroot')
    opt.parser.set_defaults(dataroot=imagepath, name='formal_author',checkpoints_dir='checkpoints/',lm_dir=landmarkdir,bg_dir=maskdir, model='test', dataset_mode='single', norm='batch', which_epoch=300, gpu_ids='-1')

    with HiddenPrints():
        opt = opt.parse()
    opt.use_local = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = apdata.CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = apmodel.create_model(opt)
    model.setup(opt)
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:  # test code only supports batch_size = 1, how_many means how many test images to run
            break
        # in test the loadSize is set to the same as fineSize
        short_path = ntpath.basename(imagepath[0])
        name = os.path.splitext(short_path)[0]
        aspect_ratio=1.0
        width=256
        if name in imagename:
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            for label, im_data in visuals.items():
                im = util.tensor2im(im_data)#tensor to numpy array [-1,1]->[0,1]->[0,255]
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(savepath, image_name)
                h, w, _ = im.shape
                if aspect_ratio > 1.0:
                    im = cv2.imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
                if aspect_ratio < 1.0:
                    im = cv2.imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
                util.save_image(im, save_path)

