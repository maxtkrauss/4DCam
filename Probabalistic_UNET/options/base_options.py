import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # added by al
        parser.add_argument('--netD_mult', type=float, default=1, help='multiply discriminator loss to strengthen or remove discriminator')
        parser.add_argument('--netG_reps', type=int, default=1, help='repeat u-net and pass feature maps to make a double u-net or more')
        parser.add_argument('--use_nll', action='store_true', help='use probabilistic pix2pix with nll loss')
        parser.add_argument('--sigma_min', type=float, default=0.001, help='regularize sigma value')
        parser.add_argument('--lambda_nll', type=float, default=1.0, help='weight for nll loss component')
        # added by Max
        parser.add_argument('--polarization', type=int, default=0, help='linear polarization used from input diffractogram')
        parser.add_argument("--video_mode", type=bool, default=False, help='sets dummy groundtruth for video generation')
        parser.add_argument("--GT_upsample", type=bool, default=False, help='Upsamples output reconstruction in the spatial domain')
        parser.add_argument('--lambda_3d_ssim', type=float, default=100.0, help='weight for 3D SSIM loss component')
        parser.add_argument('--lambda_sc', type=float, default=1.0, help='weight for SC loss component')
        parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for GAN loss component')
        parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for L1 loss component')
        # Add the auto_lambda argument here since it's only used during training
        parser.add_argument('--auto_lambda', action='store_true', help='if specified, use auto lambda tuning')
        parser.add_argument('--norm_bitwise', action='store_true', help='if specified, normalize each image by its bit range instead of the dataset min/max')
        parser.add_argument('--input_corruption_prob', type=float, default=0.0, help='probability of applying synthetic corruption to input A during training')
        parser.add_argument('--input_corruption_types', type=str, default='poisson', help='comma-separated input corruption types: poisson,gaussian,speckle,saltpepper,brightness,contrast,offset,blur,occlusion')
        parser.add_argument('--input_poisson_peak_min', type=float, default=32.0, help='minimum Poisson peak for input corruption (lower is noisier)')
        parser.add_argument('--input_poisson_peak_max', type=float, default=1024.0, help='maximum Poisson peak for input corruption (higher is cleaner)')
        parser.add_argument('--input_gaussian_std_min', type=float, default=0.003, help='minimum Gaussian std for input corruption')
        parser.add_argument('--input_gaussian_std_max', type=float, default=0.03, help='maximum Gaussian std for input corruption')
        parser.add_argument('--input_speckle_std_min', type=float, default=0.003, help='minimum speckle std for input corruption')
        parser.add_argument('--input_speckle_std_max', type=float, default=0.04, help='maximum speckle std for input corruption')
        parser.add_argument('--input_saltpepper_amount_min', type=float, default=0.001, help='minimum salt-pepper corruption amount')
        parser.add_argument('--input_saltpepper_amount_max', type=float, default=0.02, help='maximum salt-pepper corruption amount')
        parser.add_argument('--input_brightness_scale_min', type=float, default=0.6, help='minimum multiplicative brightness scale')
        parser.add_argument('--input_brightness_scale_max', type=float, default=1.4, help='maximum multiplicative brightness scale')
        parser.add_argument('--input_contrast_scale_min', type=float, default=0.6, help='minimum contrast scale')
        parser.add_argument('--input_contrast_scale_max', type=float, default=1.4, help='maximum contrast scale')
        parser.add_argument('--input_offset_min', type=float, default=-0.2, help='minimum additive brightness offset')
        parser.add_argument('--input_offset_max', type=float, default=0.2, help='maximum additive brightness offset')
        parser.add_argument('--input_blur_kernel_min', type=int, default=3, help='minimum Gaussian blur kernel size')
        parser.add_argument('--input_blur_kernel_max', type=int, default=11, help='maximum Gaussian blur kernel size')
        parser.add_argument('--input_occlusion_frac_min', type=float, default=0.1, help='minimum occlusion side fraction')
        parser.add_argument('--input_occlusion_frac_max', type=float, default=0.35, help='maximum occlusion side fraction')
        parser.add_argument('--input_occlusion_fill_value', type=float, default=0.0, help='fill value for occluded regions')
        parser.add_argument('--input_corruption_eval', action='store_true', help='also allow synthetic input corruption outside training mode')
        parser.add_argument('--extra_dataroots', type=str, default='', help='comma-separated extra dataroots to mix with the primary dataroot')
        parser.add_argument('--primary_domain_index', type=int, default=0, help='index of the in-distribution domain within dataroot + extra_dataroots')
        parser.add_argument('--treat_extra_dataroots_as_ood', action='store_true', help='label samples from extra dataroots as OOD for the OOD head')
        parser.add_argument('--use_risk_model', action='store_true', help='enable auxiliary risk and OOD heads for deployment-time reliability prediction')
        parser.add_argument('--lambda_risk', type=float, default=1.0, help='weight for the risk prediction loss')
        parser.add_argument('--lambda_risk_image', type=float, default=1.0, help='weight for direct image-level MAE regression loss')
        parser.add_argument('--lambda_ood', type=float, default=1.0, help='weight for the OOD prediction loss')
        parser.add_argument('--w_recon', type=float, default=1.0, help='overall weight applied to the existing reconstruction objective bundle')
        parser.add_argument('--w_nll', type=float, default=1.0, help='overall weight applied to the NLL term')
        parser.add_argument('--w_risk', type=float, default=1.0, help='overall weight applied to the risk-prediction term')
        parser.add_argument('--w_ood', type=float, default=1.0, help='overall weight applied to the OOD term')
        parser.add_argument('--risk_head_hidden', type=int, default=64, help='hidden channels for the risk head')
        parser.add_argument('--risk_image_hidden', type=int, default=128, help='hidden units for the image-level risk head MLP')
        parser.add_argument('--ood_head_hidden', type=int, default=64, help='hidden units for the OOD head MLP')
        parser.add_argument('--risk_target', type=str, default='abs_error', help='risk supervision target [abs_error | smoothed_abs_error]')
        parser.add_argument('--risk_target_detach', action='store_true', help='detach the MAE target used to supervise the risk head')
        parser.add_argument('--risk_smoothing_kernel', type=int, default=3, help='average-pool kernel for smoothed_abs_error risk targets')
        parser.add_argument('--lambda_risk_consistency', type=float, default=0.25, help='weight for aligning the image-level risk head with the mean predicted risk map')
        parser.add_argument('--final_risk_weight_mae', type=float, default=1.0, help='weight for predicted MAE in the final deployment risk score')
        parser.add_argument('--final_risk_weight_ood', type=float, default=1.0, help='weight for OOD score in the final deployment risk score')
        parser.add_argument('--final_risk_weight_sigma', type=float, default=1.0, help='weight for sigma mean in the final deployment risk score')
        parser.add_argument('--final_risk_mae_scale', type=float, default=0.01, help='scale that maps predicted MAE into a bounded deployment-risk contribution')
        parser.add_argument('--final_risk_sigma_scale', type=float, default=0.01, help='scale that maps sigma mean into a bounded deployment-risk contribution')
        parser.add_argument('--final_risk_ood_floor', type=float, default=0.6, help='OOD probabilities below this contribute little or nothing to the deployment-risk score')
        parser.add_argument('--force_is_ood_label', type=int, default=-1, help='force all samples in a run to share a specific OOD label: -1 keeps dataset-provided labels')
        parser.add_argument('--save_eval_metadata', action='store_true', help='save per-image prediction metadata during test/eval runs')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # wandb parameters
        parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
        parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
