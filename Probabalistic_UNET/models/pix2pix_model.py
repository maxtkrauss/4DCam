import torch
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from pytorch_msssim import ssim
import numpy as np
import time
from collections import OrderedDict


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


def resize_to_match(tensor, ref_tensor):
    if tensor.shape[-2:] == ref_tensor.shape[-2:]:
        return tensor
    return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='bilinear', align_corners=False)


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
        self.use_risk_model = getattr(opt, 'use_risk_model', False)
        self.sigma_min = opt.sigma_min
        self.lambda_nll = opt.lambda_nll
        self.lambda_risk = getattr(opt, 'lambda_risk', 1.0)
        self.lambda_risk_image = getattr(opt, 'lambda_risk_image', 1.0)
        self.lambda_risk_consistency = getattr(opt, 'lambda_risk_consistency', 0.25)
        self.lambda_ood = getattr(opt, 'lambda_ood', 1.0)
        self.w_recon = getattr(opt, 'w_recon', 1.0)
        self.w_nll = getattr(opt, 'w_nll', 1.0)
        self.w_risk = getattr(opt, 'w_risk', 1.0)
        self.w_ood = getattr(opt, 'w_ood', 1.0)
        self.risk_target_mode = getattr(opt, 'risk_target', 'abs_error')
        self.risk_target_detach = getattr(opt, 'risk_target_detach', False)
        self.risk_smoothing_kernel = max(1, int(getattr(opt, 'risk_smoothing_kernel', 3)))
        self.final_risk_weight_mae = getattr(opt, 'final_risk_weight_mae', 1.0)
        self.final_risk_weight_ood = getattr(opt, 'final_risk_weight_ood', 1.0)
        self.final_risk_weight_sigma = getattr(opt, 'final_risk_weight_sigma', 1.0)
        self.final_risk_mae_scale = max(1e-8, float(getattr(opt, 'final_risk_mae_scale', 0.01)))
        self.final_risk_sigma_scale = max(1e-8, float(getattr(opt, 'final_risk_sigma_scale', 0.01)))
        self.final_risk_ood_floor = float(getattr(opt, 'final_risk_ood_floor', 0.6))
        self.force_is_ood_label = int(getattr(opt, 'force_is_ood_label', -1))
        self.spectral_nc = opt.output_nc // 2 if self.use_nll else opt.output_nc


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
        if self.use_nll:
            self.loss_names.append('G_NLL')
        if self.use_risk_model:
            self.loss_names.extend(['G_RISK', 'G_RISK_IMAGE', 'G_RISK_CONSISTENCY', 'G_OOD', 'risk_mae_pred', 'ood_prob', 'final_risk'])
        if self.auto_lambda:
            self.loss_names.extend(['log_sigma_L1', 'log_sigma_SSIM3D', 'log_sigma_SC', 'log_sigma_Grad'])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.use_risk_model:
            self.visual_names.extend(['risk_B', 'ood_B'])
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
            use_nll=self.use_nll,
        )
        if self.use_risk_model:
            risk_in_channels = self.spectral_nc * 2 + 1
            pooled_feature_dim = risk_in_channels * 3
            risk_hidden = max(8, int(getattr(opt, 'risk_head_hidden', 64)))
            risk_image_hidden = max(8, int(getattr(opt, 'risk_image_hidden', 128)))
            ood_hidden = max(8, int(getattr(opt, 'ood_head_hidden', 64)))
            self.netRisk = torch.nn.Sequential(
                # Spatially aware error head: use local neighborhoods instead of only per-pixel channel mixing.
                torch.nn.Conv2d(risk_in_channels, risk_hidden, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(risk_hidden, risk_hidden, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(risk_hidden, self.spectral_nc, kernel_size=1),
                torch.nn.Softplus(),
            ).to(self.device)
            self.netRiskImage = torch.nn.Sequential(
                torch.nn.Linear(pooled_feature_dim, risk_image_hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(risk_image_hidden, 1),
                torch.nn.Softplus(),
            ).to(self.device)
            self.netOOD = torch.nn.Sequential(
                torch.nn.Linear(pooled_feature_dim, ood_hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(ood_hidden, 1),
            ).to(self.device)
            self.model_names.append('Risk')
            self.model_names.append('RiskImage')
            self.model_names.append('OOD')
        
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
            self.criterionRisk = torch.nn.L1Loss()
            self.criterionOOD = torch.nn.BCEWithLogitsLoss()

            # add NLL
            #self.criterionNLL = laplace_nll


            # add PDF
            # self.criterionNLL = laplace_pdf 

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.use_risk_model:
                g_params = itertools.chain(
                    self.netG.parameters(),
                    self.netRisk.parameters(),
                    self.netRiskImage.parameters(),
                    self.netOOD.parameters(),
                )
            else:
                g_params = self.netG.parameters()
            self.optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.input_is_shifted = input.get('is_shifted', torch.zeros(self.real_A.shape[0])).to(self.device).float().view(-1)
        self.input_shift_strength = input.get('shift_strength', torch.zeros(self.real_A.shape[0])).to(self.device).float().view(-1)
        self.input_source_domain = input.get('source_domain', torch.zeros(self.real_A.shape[0], dtype=torch.long)).to(self.device).long().view(-1)
        self.input_is_ood = input.get('is_ood', self.input_is_shifted).to(self.device).float().view(-1)
        if self.force_is_ood_label in (0, 1):
            self.input_is_ood = torch.full_like(self.input_is_ood, float(self.force_is_ood_label))

    #def forward(self):
    #    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #    self.fake_B = self.netG(self.real_A)  # G(A)

    #def forward(self):
    #    output = self.netG(self.real_A)  # G(real)
    #    self.fake_mean = output[:, :self.real_A.shape[1], :, :]  # Mean
    #    self.fake_scale = output[:, self.real_A.shape[1]:, :, :]  # Scale

    def forward(self):
        """Run forward pass."""
        out = self.netG(self.real_A)  # G(A)
        self.fake_B_raw = out[0] if isinstance(out, (tuple, list)) else out
        if self.use_nll:
            self.mu_B_raw = self.fake_B_raw[:, :self.spectral_nc, :, :]
            self.sigma_B_raw = torch.clamp(self.fake_B_raw[:, self.spectral_nc:, :, :], min=self.sigma_min)
            self.mu_B = resize_to_match(self.mu_B_raw, self.real_B)
            self.sigma_B = resize_to_match(self.sigma_B_raw, self.real_B)
            self.fake_B = torch.cat([self.mu_B, self.sigma_B], dim=1)
        else:
            self.mu_B_raw = self.fake_B_raw
            self.mu_B = resize_to_match(self.mu_B_raw, self.real_B)
            self.sigma_B = torch.zeros_like(self.mu_B)
            self.sigma_B_raw = self.sigma_B
            self.fake_B = self.mu_B
        if self.use_risk_model:
            input_resized = F.interpolate(self.real_A, size=self.mu_B.shape[-2:], mode='bilinear', align_corners=False)
            risk_features = torch.cat([self.mu_B, self.sigma_B, input_resized], dim=1)
            self.risk_B = self.netRisk(risk_features)
            pooled_mean = torch.mean(risk_features, dim=(2, 3))
            pooled_std = torch.std(risk_features, dim=(2, 3), unbiased=False)
            pooled_max = torch.amax(risk_features, dim=(2, 3))
            pooled_features = torch.cat([pooled_mean, pooled_std, pooled_max], dim=1)
            self.predicted_image_mae = self.netRiskImage(pooled_features).view(-1)
            self.ood_logits = self.netOOD(pooled_features).view(-1)
            self.ood_prob = torch.sigmoid(self.ood_logits)
            self.risk_map_mean = torch.mean(self.risk_B, dim=(1, 2, 3))
            self.sigma_mean = torch.mean(self.sigma_B, dim=(1, 2, 3))
            self.risk_mae_component = self.predicted_image_mae / (self.predicted_image_mae + self.final_risk_mae_scale)
            self.risk_sigma_component = self.sigma_mean / (self.sigma_mean + self.final_risk_sigma_scale)
            self.risk_ood_component = torch.clamp(
                (self.ood_prob - self.final_risk_ood_floor) / max(1e-6, (1.0 - self.final_risk_ood_floor)),
                min=0.0,
                max=1.0,
            )
            total_weight = self.final_risk_weight_mae + self.final_risk_weight_ood + self.final_risk_weight_sigma
            total_weight = max(total_weight, 1e-8)
            self.final_risk = (
                self.final_risk_weight_mae * self.risk_mae_component
                + self.final_risk_weight_ood * self.risk_ood_component
                + self.final_risk_weight_sigma * self.risk_sigma_component
            ) / total_weight
            self.ood_B = self.ood_prob.view(-1, 1, 1, 1).expand(-1, 1, self.mu_B.shape[2], self.mu_B.shape[3])
        else:
            self.predicted_image_mae = torch.mean(torch.abs(self.mu_B), dim=(1, 2, 3))
            self.risk_map_mean = self.predicted_image_mae
            self.sigma_mean = torch.mean(self.sigma_B, dim=(1, 2, 3))
            self.risk_mae_component = self.predicted_image_mae / (self.predicted_image_mae + self.final_risk_mae_scale)
            self.risk_sigma_component = self.sigma_mean / (self.sigma_mean + self.final_risk_sigma_scale)
            self.risk_ood_component = torch.zeros_like(self.risk_sigma_component)
            self.final_risk = self.risk_sigma_component
        self.true_image_mae = torch.mean(torch.abs(self.mu_B - self.real_B), dim=(1, 2, 3))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.real_A_resized = F.interpolate(self.real_A, size=self.real_B.shape[-2:], mode='bilinear', align_corners=False)

        # For NLL, use only mean for D; otherwise, use full image
        if self.use_nll:
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
        else:
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
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
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.mu_B, self.real_B)
            self.loss_G_SC = spectral_correlation_loss(self.real_B, self.mu_B)
            self.loss_G_3D_SSIM = ssim_3d_loss(self.real_B, self.mu_B)
            self.loss_G_MAE = mae_3d_loss(self.real_B, self.mu_B)
            self.loss_G_NLL = laplace_nll(self.real_B, self.fake_B, self.sigma_min)
        else:
            fake_AB = torch.cat((self.real_A_resized, self.mu_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.mu_B, self.real_B)
            self.loss_G_SC = spectral_correlation_loss(self.real_B, self.mu_B)
            self.loss_G_3D_SSIM = ssim_3d_loss(self.real_B, self.mu_B)
            self.loss_G_MAE = mae_3d_loss(self.real_B, self.mu_B)
            self.loss_G_NLL = 0.0

        if self.use_risk_model:
            risk_target = torch.abs(self.mu_B - self.real_B)
            if self.risk_target_mode == 'smoothed_abs_error':
                pad = self.risk_smoothing_kernel // 2
                risk_target = F.avg_pool2d(
                    risk_target,
                    kernel_size=self.risk_smoothing_kernel,
                    stride=1,
                    padding=pad,
                )
            if self.risk_target_detach:
                risk_target = risk_target.detach()
            self.loss_G_RISK = self.criterionRisk(self.risk_B, risk_target)
            self.true_image_mae = torch.mean(torch.abs(self.mu_B - self.real_B), dim=(1, 2, 3))
            self.loss_G_RISK_IMAGE = self.criterionRisk(self.predicted_image_mae, self.true_image_mae.detach())
            self.loss_G_RISK_CONSISTENCY = self.criterionRisk(self.predicted_image_mae, self.risk_map_mean.detach())
            self.loss_risk_mae_pred = torch.mean(self.predicted_image_mae)
            self.loss_G_OOD = self.criterionOOD(self.ood_logits, self.input_is_ood)
            self.loss_ood_prob = torch.mean(self.ood_prob)
            self.loss_final_risk = torch.mean(self.final_risk)
        else:
            self.true_image_mae = torch.mean(torch.abs(self.mu_B - self.real_B), dim=(1, 2, 3))
            self.loss_G_RISK = 0.0
            self.loss_G_RISK_IMAGE = 0.0
            self.loss_G_RISK_CONSISTENCY = 0.0
            self.loss_G_OOD = 0.0
            self.loss_risk_mae_pred = 0.0
            self.loss_ood_prob = 0.0
            self.loss_final_risk = torch.mean(self.final_risk)

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
            loss = self.w_recon * loss
            if self.use_nll:
                loss = loss + (self.w_nll * self.loss_G_NLL * self.lambda_nll)
            if self.use_risk_model:
                loss = loss + (
                    self.w_risk * (
                        (self.loss_G_RISK * self.lambda_risk)
                        + (self.loss_G_RISK_IMAGE * self.lambda_risk_image)
                        + (self.loss_G_RISK_CONSISTENCY * self.lambda_risk_consistency)
                    )
                ) + (self.w_ood * self.loss_G_OOD * self.lambda_ood)
            self.loss_G = loss
            # Store sigma values for logging (detach to avoid affecting gradients)
            self.loss_log_sigma_L1_val = self.loss_log_sigma_L1.detach()
            self.loss_log_sigma_SSIM3D_val = self.loss_log_sigma_SSIM3D.detach()
            self.loss_log_sigma_SC_val = self.loss_log_sigma_SC.detach()
            self.loss_log_sigma_Grad_val = self.loss_log_sigma_Grad.detach()
        else:
            recon_loss = (
                (self.loss_G_3D_SSIM * self.lambda_3d_ssim)
                + (self.loss_G_SC * self.lambda_sc)
                + (self.loss_G_GAN * self.lambda_gan)
                + (self.loss_G_L1 * self.lambda_l1)
            )
            self.loss_G = (self.w_recon * recon_loss) + (self.w_nll * self.loss_G_NLL * self.lambda_nll)
            if self.use_risk_model:
                self.loss_G = self.loss_G + (
                    self.w_risk * (
                        (self.loss_G_RISK * self.lambda_risk)
                        + (self.loss_G_RISK_IMAGE * self.lambda_risk_image)
                        + (self.loss_G_RISK_CONSISTENCY * self.lambda_risk_consistency)
                    )
                ) + (self.w_ood * self.loss_G_OOD * self.lambda_ood)

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

    def get_current_metrics(self):
        metrics = OrderedDict()
        metrics['predicted_image_mae'] = float(torch.mean(self.predicted_image_mae).detach().cpu())
        metrics['sigma_mean'] = float(torch.mean(self.sigma_mean).detach().cpu())
        metrics['final_risk'] = float(torch.mean(self.final_risk).detach().cpu())
        metrics['true_image_mae'] = float(torch.mean(self.true_image_mae).detach().cpu())
        metrics['is_ood'] = float(torch.mean(self.input_is_ood).detach().cpu())
        metrics['is_shifted'] = float(torch.mean(self.input_is_shifted).detach().cpu())
        metrics['source_domain'] = float(torch.mean(self.input_source_domain.float()).detach().cpu())
        metrics['risk_mae_component'] = float(torch.mean(self.risk_mae_component).detach().cpu())
        metrics['risk_sigma_component'] = float(torch.mean(self.risk_sigma_component).detach().cpu())
        metrics['risk_ood_component'] = float(torch.mean(self.risk_ood_component).detach().cpu())
        metrics['final_risk_weight_mae'] = float(self.final_risk_weight_mae)
        metrics['final_risk_weight_ood'] = float(self.final_risk_weight_ood)
        metrics['final_risk_weight_sigma'] = float(self.final_risk_weight_sigma)
        if self.use_risk_model:
            metrics['ood_prob'] = float(torch.mean(self.ood_prob).detach().cpu())
        return metrics
