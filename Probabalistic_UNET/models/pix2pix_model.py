import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from pytorch_msssim import ssim
import numpy as np
import time


def spectral_correlation_loss(y_true, y_pred):
    """
    Compute the spectral correlation loss based on the correlation coefficient.
    The loss ensures that generated spectral profiles are well-correlated with ground truth.
    """
    mean_true = torch.mean(y_true, dim=1, keepdim=True)
    mean_pred = torch.mean(y_pred, dim=1, keepdim=True)

    std_true = torch.std(y_true, dim=1, keepdim=True) + 1e-6  # Avoid division by zero
    std_pred = torch.std(y_pred, dim=1, keepdim=True) + 1e-6

    covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=1)
    correlation = covariance / (std_true * std_pred)

    return 1 - torch.mean(correlation)  # Loss is minimized when correlation is high

def ssim_3d_loss(y_true, y_pred):
    """Compute 3D Structural Similarity (SSIM) loss between two 3D tensors."""
    data_range = y_true.max() - y_true.min()  # Compute data range dynamically
    loss = 1 - ssim(y_true, y_pred, data_range=data_range, size_average=True)
    return loss

def spatial_consistency_loss(generated_hsi, input_greyscale):
    """
    Compare the mean of the generated hyperspectral image to the original input greyscale image.
    Enforces spatial alignment and detail preservation.
    """
    generated_greyscale = torch.mean(generated_hsi, dim=1, keepdim=True)  # (B, 1, H, W)
    return F.l1_loss(generated_greyscale, input_greyscale)

def mae_3d_loss(y_true, y_pred):
      return torch.mean(torch.abs(y_true - y_pred))


def laplace_nll(real_B, fake_B, sigma_min=1e-3):
    C = torch.log(torch.tensor(2.0))
    n = real_B.shape[1]
    
    mu = fake_B[:, :n, :, :]
    sigma = fake_B[:, n:, :, :]

    # Ensure sigma is positive and above a minimum threshold
    sigma = torch.clamp(sigma, min=sigma_min)
    
    # Compute the negative log-likelihood
    nll = torch.abs((mu - real_B) / sigma) + torch.log(sigma) + C
    nll_mean = torch.mean(nll)
    
    return nll_mean


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # Load loss gains from opt if available, otherwise use defaults
        self.lambda_3d_ssim = getattr(opt, 'lambda_3d_ssim', 100.0)
        self.lambda_sc = getattr(opt, 'lambda_sc', 1.0)
        self.lambda_gan = getattr(opt, 'lambda_gan', 0.01)
        self.lambda_l1 = getattr(opt, 'lambda_l1', 1.0)
        self.use_nll = opt.use_nll
        self.sigma_min = opt.sigma_min
        self.lambda_nll = opt.lambda_nll


        # after you read self.lambda_* from opt:
        self.auto_lambda = getattr(opt, 'auto_lambda', False)
        if self.auto_lambda:
            # Create learnable log-variances for each base loss (start near log(1))
            self.loss_log_sigma_L1 = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.loss_log_sigma_SSIM3D = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.loss_log_sigma_SC = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.loss_log_sigma_Grad = torch.nn.Parameter(torch.zeros(1, device=self.device))

            # Create a parameter list to hold these parameters
            self.auto_lambda_params = torch.nn.ParameterList([
                self.loss_log_sigma_L1,
                self.loss_log_sigma_SSIM3D,
                self.loss_log_sigma_SC,
                self.loss_log_sigma_Grad
            ])

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # add G_NLL
        self.loss_names = ['G_GAN', 'G_L1', 'G_SC', 'G_3D_SSIM', 'D_real', 'D_fake']
        if self.auto_lambda:
            self.loss_names.extend(['log_sigma_L1', 'log_sigma_SSIM3D', 'log_sigma_SC', 'log_sigma_Grad'])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.GT_upsample = opt.GT_upsample
        # TRANSFORMER Testing:
        
        # change for mean and scale outputs
        #self.netG = networks.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # get netD_weight
        self.netD_mult = opt.netD_mult

        if self.isTrain:  
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.use_nll:
                discriminator_input_nc = opt.input_nc + opt.output_nc // 2
            else:
                discriminator_input_nc = opt.input_nc + opt.output_nc
            
            self.netD = networks.define_D(discriminator_input_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            #self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # add NLL
            #self.criterionNLL = laplace_nll


            # add PDF
            # self.criterionNLL = laplace_pdf 

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        
        # Check if 'B' is in the input; use dummy tensor if not provided
        if 'B' in input:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        else:
            self.real_B = torch.zeros_like(self.real_A).to(self.device)
        
        # Check if paths exist; otherwise, set to None
        self.image_paths = input.get('A_paths' if AtoB else 'B_paths', None)

    #def forward(self):
    #    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #    self.fake_B = self.netG(self.real_A)  # G(A)

    #def forward(self):
    #    output = self.netG(self.real_A)  # G(real)
    #    self.fake_mean = output[:, :self.real_A.shape[1], :, :]  # Mean
    #    self.fake_scale = output[:, self.real_A.shape[1]:, :, :]  # Scale

    def forward(self):
        """Run forward pass."""
        self.fake_B, *_ = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.GT_upsample:
            self.real_A_resized = F.interpolate(self.real_A, size=(660, 660), mode='bilinear', align_corners=False)
        else:
            self.real_A_resized = F.interpolate(self.real_A, size=(128, 128), mode='bilinear', align_corners=False)

        # For NLL, use only mean for D; otherwise, use full image
        if self.use_nll:
            n = self.real_B.shape[1]
            fake_mean = self.fake_B[:, :n, :, :]
            fake_AB = torch.cat((self.real_A_resized, fake_mean), 1)
        else:
            fake_AB = torch.cat((self.real_A_resized, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A_resized, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        #self.loss_D_real = 0
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.use_nll:
            n = self.real_B.shape[1]
            self.mu_B = self.fake_B[:, :n, :, :]
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.mu_B, self.real_B)
            self.loss_G_SC = spectral_correlation_loss(self.real_B, self.mu_B)
            self.loss_G_3D_SSIM = ssim_3d_loss(self.real_B, self.mu_B)
            self.loss_G_MAE = mae_3d_loss(self.real_B, self.mu_B)
            self.loss_G_NLL = laplace_nll(self.real_B, self.fake_B, self.sigma_min)
        else:
            self.mu_B = self.fake_B  # In non-NLL mode, fake_B is just the mean
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.mu_B, self.real_B)
            self.loss_G_SC = spectral_correlation_loss(self.real_B, self.mu_B)
            self.loss_G_3D_SSIM = ssim_3d_loss(self.real_B, self.mu_B)
            self.loss_G_MAE = mae_3d_loss(self.real_B, self.mu_B)
            self.loss_G_NLL = 0.0

        if self.auto_lambda:
            # Uncertainty weighting (Kendall et al.)
            loss = 0
            # L1
            loss += (torch.exp(-self.loss_log_sigma_L1) * self.loss_G_L1 + self.loss_log_sigma_L1)
            # 3D SSIM
            loss += (torch.exp(-self.loss_log_sigma_SSIM3D) * self.loss_G_3D_SSIM + self.loss_log_sigma_SSIM3D)
            # Spectral Correlation
            loss += (torch.exp(-self.loss_log_sigma_SC) * self.loss_G_SC + self.loss_log_sigma_SC)
            # GAN
            loss += (torch.exp(-self.loss_log_sigma_Grad) * self.loss_G_GAN + self.loss_log_sigma_Grad)
            self.loss_G = loss
            # Store sigma values for logging (detach to avoid affecting gradients)
            self.loss_log_sigma_L1_val = self.loss_log_sigma_L1.detach()
            self.loss_log_sigma_SSIM3D_val = self.loss_log_sigma_SSIM3D.detach()
            self.loss_log_sigma_SC_val = self.loss_log_sigma_SC.detach()
            self.loss_log_sigma_Grad_val = self.loss_log_sigma_Grad.detach()
        else:
            self.loss_G = (
                (self.loss_G_3D_SSIM * self.lambda_3d_ssim)
                + (self.loss_G_SC * self.lambda_sc)
                + (self.loss_G_GAN * self.lambda_gan)
                + (self.loss_G_L1 * self.lambda_l1)
                + (self.loss_G_NLL * self.lambda_nll)
            )

        self.loss_G.backward()
    
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
