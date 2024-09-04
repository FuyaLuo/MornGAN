import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torchvision import models
import math
from skimage import measure
from kmeans_pytorch import kmeans



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Spectral normalization base class 
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

################################SN#######################

#######Positional Encoding module, borrowed from https://github.com/open-mmlab/mmgeneration
class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]

        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class CatersianGrid(nn.Module):
    """Catersian Grid for 2d tensor.
    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)

#######################################Positional Encoding############

##########Central Difference Convolution, borrowed from https://github.com/ZitongYu/CDCN/
class CDC2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC2d, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x[:, :, 1:-1, 1:-1], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            # print(out_normal.size())
            # print(out_diff.size())
            # print(x.size())

            return out_normal - self.theta * out_diff
###############

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, ngf, net_Gen_type, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if net_Gen_type == 'gen_v1':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoderv1, dec_args)
    elif net_Gen_type == 'gen_CSG2':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder_CSG2, enc_args, ResnetGenDecoderv1, dec_args) 
    else:
        raise NotImplementedError('Generation Net [%s] is not found' % net_Gen_type)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminatorSN, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD

def define_DF(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netDF = DF_Plexer(n_domains, NLayerDiscriminatorSN, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netDF.cuda(gpu_ids[0])

    plex_netDF.apply(weights_init)
    return plex_netDF

def define_S(input_nc, ngf, n_blocks, n_domains, num_classes=19, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, n_blocks, ngf, num_classes, norm_layer, use_dropout, gpu_ids)
    plex_netS = S_Plexer(n_domains, SegmentorHeadv2, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netS.cuda(gpu_ids[0])

    plex_netS.apply(weights_init)
    return plex_netS


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))
######Optional added by lfy

def UpdateSegGT(seg_tensor, ori_seg_GT, prob_th):
    "Use the high confidence predicted class to update original segmentation GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm = sm(seg_tensor.detach())
    pred_max_tensor = torch.max(pred_sm, dim=1)
    pred_max_value = pred_max_tensor[0]
    pred_max_category = pred_max_tensor[1]
    seg_HP_mask = torch.zeros_like(pred_max_value)
    seg_HP_mask = torch.where(pred_max_value > prob_th, torch.ones_like(pred_max_value), seg_HP_mask)
    seg_GT_float = ori_seg_GT.float()
    segGT_UC_mask = torch.zeros_like(seg_GT_float)
    segGT_UC_mask = torch.where(seg_GT_float == 255.0, torch.ones_like(seg_GT_float), segGT_UC_mask)
    seg_HP_mask_UC = seg_HP_mask.mul(segGT_UC_mask)
    mask_new_GT = seg_HP_mask_UC.mul(pred_max_category.float()) + (torch.ones_like(seg_HP_mask_UC) - seg_HP_mask_UC).mul(seg_GT_float)

    return mask_new_GT.detach()

def OnlSemDisModule(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th):
    "Online semantic distillation module: Use the common high confidence predicted class to update original segmentation GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(seg_tensor1.detach())
    pred_sm2 = sm(seg_tensor2.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_tensor2 = torch.max(pred_sm2, dim=1)
    pred_max_value1 = pred_max_tensor1[0]
    pred_max_value2 = pred_max_tensor2[0]
    pred_max_category1 = pred_max_tensor1[1]
    pred_max_category2 = pred_max_tensor2[1]

    mask_category1 = pred_max_category1.float()
    mask_category2 = pred_max_category2.float()
    mask_sub = mask_category1 - mask_category2
    mask_inter = torch.zeros_like(mask_category1)
    mask_inter = torch.where(mask_sub == 0.0, torch.ones_like(mask_category1), mask_inter)

    seg_GT_float = ori_seg_GT.float()
    seg_HP_mask1 = torch.zeros_like(pred_max_value1)
    seg_HP_mask1 = torch.where(pred_max_value1 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask1)
    seg_HP_mask2 = torch.zeros_like(pred_max_value1)
    seg_HP_mask2 = torch.where(pred_max_value2 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask2)
    mask_inter_HP = seg_HP_mask1.mul(seg_HP_mask1)

    segGT_UC_mask = torch.zeros_like(seg_GT_float)
    segGT_UC_mask = torch.where(seg_GT_float == 255.0, torch.ones_like(seg_GT_float), segGT_UC_mask)
    seg_inter_mask_UC = mask_inter.mul(segGT_UC_mask)

    seg_inter_mask_UC_HP = mask_inter_HP.mul(seg_inter_mask_UC)

    mask_new_GT = seg_inter_mask_UC_HP.mul(mask_category1) + (torch.ones_like(seg_inter_mask_UC_HP) - seg_inter_mask_UC_HP).mul(seg_GT_float)
    mask_final = SemDenoiseProc4IR(torch.squeeze(mask_new_GT), input_IR)
    ###Removal of vegetated areas from supervision
    mask_Bkg_all = torch.zeros_like(mask_final)
    mask_Bkg_all = torch.where(mask_final < 11.0, torch.ones_like(mask_Bkg_all), torch.zeros_like(mask_Bkg_all))
    mask_Build_new = torch.zeros_like(mask_final)
    mask_Build_new = torch.where(mask_final == 2.0, torch.ones_like(mask_Build_new), torch.zeros_like(mask_Build_new))

    mask_Sign_new = torch.zeros_like(mask_final)
    mask_Sign_new = torch.where(mask_final == 6.0, torch.ones_like(mask_Sign_new), torch.zeros_like(mask_Sign_new))
    mask_Light_new = torch.zeros_like(mask_final)
    mask_Light_new = torch.where(mask_final == 7.0, torch.ones_like(mask_Light_new), torch.zeros_like(mask_Light_new))
    mask_Car_new = torch.zeros_like(mask_final)
    mask_Car_new = torch.where(mask_final == 13.0, torch.ones_like(mask_Car_new), torch.zeros_like(mask_Car_new))
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    "Before the parameters of the segmentation network are fixed, the threshold for the background category is set to 0.99; "
    "conversely, the threshold for all categories is set to 0.95."
    if torch.mean(mask_sub) == 0.0:
        High_th = prob_th
    else:
        High_th = prob_th + 0.04
    
    # High_th = 0.99
    LHP_mask = torch.zeros_like(pred_max_value1)
    LHP_mask = torch.where(pred_max_value1 < High_th, torch.ones_like(LHP_mask), torch.zeros_like(LHP_mask))
    # VegRoad_LP_mask = LHP_mask.mul(mask_Veg_new) + LHP_mask.mul(mask_Road_new)
    VegRoad_LP_mask = LHP_mask.mul(mask_Bkg_stuff)
    ####Confusing categories Mask
    
    mask_CurtVeg = (torch.ones_like(mask_Bkg_stuff) - VegRoad_LP_mask).mul(mask_final) + VegRoad_LP_mask * 255.0

    return mask_CurtVeg.expand_as(ori_seg_GT).detach()

def SemDenoiseProc4IR(ori_mask, input_IR):
    "Semantic denoising process for NTIR images: Use original IR image to refine segmentaton mask for specific categories, "
    "i.e., Sky, Vegetation, Pole, and Person."
    "ori_mask: h * w,  input_IR: 1 * 3 * h * w"

    x_norm = (input_IR - torch.min(input_IR)) / (torch.max(input_IR) - torch.min(input_IR))
    IR_gray = torch.squeeze(.299 * x_norm[:,0:1,:,:] + .587 * x_norm[:,1:2,:,:] + .114 * x_norm[:,2:3,:,:])

    Pole_mask = torch.zeros_like(ori_mask)
    Veg_mask = torch.zeros_like(ori_mask)
    Sky_mask = torch.zeros_like(ori_mask)
    Person_mask = torch.zeros_like(ori_mask)
    Pole_mask = torch.where(ori_mask == 5.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Veg_mask = torch.where(ori_mask == 8.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Sky_mask = torch.where(ori_mask == 10.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Person_mask = torch.where(ori_mask == 11.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    cnt_Pole = torch.sum(Pole_mask)
    cnt_Veg = torch.sum(Veg_mask)
    cnt_Sky = torch.sum(Sky_mask)
    cnt_Person = torch.sum(Person_mask)
    region_Pole = Pole_mask.mul(IR_gray)
    region_Veg = Veg_mask.mul(IR_gray)
    region_Sky = Sky_mask.mul(IR_gray)
    region_Person = Person_mask.mul(IR_gray)

    if cnt_Pole > 0:
        Pole_region_mean = torch.sum(region_Pole) / cnt_Pole
        if cnt_Sky > 0:
            Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
            #####Corrected Pole region mean.
            Pole_region_Corr_mean = (Pole_region_mean + Sky_region_mean) * 0.5
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_Corr_mean), 2))
        else:
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_mean), 2))

    # if cnt_Pole > 0:
    #     Pole_region_mean = torch.sum(region_Pole) / cnt_Pole
    #     Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_mean), 2))

    if cnt_Veg > 0:
        Veg_region_mean = torch.sum(region_Veg) / cnt_Veg
        Veg_intradis = Veg_mask.mul(torch.pow((region_Veg - Veg_region_mean), 2))

    if cnt_Sky > 0:
        Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
        Sky_intradis = Sky_mask.mul(torch.pow((region_Sky - Sky_region_mean), 2))

    if cnt_Person > 0:
        Person_region_mean = torch.sum(region_Person) / cnt_Person
        Person_intradis = Person_mask.mul(torch.pow((region_Person - Person_region_mean), 2))

    ######Denoised for Sky
    if (cnt_Sky * cnt_Veg) > 0:
        Sky_Veg_dis = Sky_mask.mul(torch.pow((region_Sky - Veg_region_mean), 2))
        Sky_Veg_dis_err = Sky_intradis - Sky_Veg_dis
        Sky2Veg_mask = torch.zeros_like(ori_mask)
        Sky2Veg_mask = torch.where(Sky_Veg_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Sky_refine = Sky2Veg_mask * 255.0 + (Sky_mask - Sky2Veg_mask) * 10.0

        new_Sky_mask = Sky_mask - Sky2Veg_mask
        cnt_Sky_new = torch.sum(new_Sky_mask)
        region_Sky_new = new_Sky_mask.mul(IR_gray)
        if cnt_Sky_new > 0:
            Sky_region_mean_new = torch.sum(region_Sky_new) / cnt_Sky_new
        else:
            Sky_region_mean_new = Sky_region_mean
    elif cnt_Sky > 0:
        Sky_region_mean_new = Sky_region_mean
        mask_Sky_refine = Sky_mask * 10.0
    else:
        mask_Sky_refine = Sky_mask * 10.0
        # Sky_region_mean_new = Sky_region_mean

    ######Denoised for Pole
    if (cnt_Pole * cnt_Sky) > 0:
        Pole_Sky_dis = Pole_mask.mul(torch.pow((region_Pole - Sky_region_mean_new), 2))
        Pole_Sky_dis_err = Pole_intradis - Pole_Sky_dis
        Pole2Sky_mask = torch.zeros_like(ori_mask)
        Pole2Sky_mask = torch.where(Pole_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Pole_refine = Pole2Sky_mask * 255.0 + (Pole_mask - Pole2Sky_mask) * 5.0
    else:
        mask_Pole_refine = Pole_mask * 5.0

    ######Denoised for Person
    if (cnt_Person * cnt_Veg) > 0:
        Person_Veg_dis = Person_mask.mul(torch.pow((region_Person - Veg_region_mean), 2))
        Person_Veg_dis_err = Person_intradis - Person_Veg_dis
        Person2Veg_mask = torch.zeros_like(ori_mask)
        Person2Veg_mask = torch.where(Person_Veg_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Person_refine = Person2Veg_mask * 255.0 + (Person_mask - Person2Veg_mask) * 11.0

        new_Person_mask = Person_mask - Person2Veg_mask
        cnt_Person_new = torch.sum(new_Person_mask)
        region_Person_new = new_Person_mask.mul(IR_gray)
        if cnt_Person_new > 0:
            Person_region_mean_new = torch.sum(region_Person_new) / cnt_Person_new
        else:
            Person_region_mean_new = Person_region_mean
    elif cnt_Person > 0:
        Person_region_mean_new = Person_region_mean
        mask_Person_refine = Person_mask * 11.0
    else:
        mask_Person_refine = Person_mask * 11.0
        # Person_region_mean_new = Person_region_mean

    ######Denoised for Vegetation
    if (cnt_Veg * cnt_Sky * cnt_Person) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        uncertain_mask_veg = torch.zeros_like(ori_mask)
        fuse_uncer = Veg2Sky_mask + Veg2Person_mask
        uncertain_mask_veg = torch.where(fuse_uncer > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        mask_Veg_refine = uncertain_mask_veg * 255.0 + (Veg_mask - uncertain_mask_veg) * 8.0
    elif (cnt_Veg * cnt_Sky) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Sky_mask * 255.0 + (Veg_mask - Veg2Sky_mask) * 8.0
    elif (cnt_Veg * cnt_Person) > 0:
        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Person_mask * 255.0 + (Veg_mask - Veg2Person_mask) * 8.0
    else:
        mask_Veg_refine = Veg_mask * 8.0

    mask_refine = mask_Sky_refine + mask_Pole_refine + mask_Person_refine + mask_Veg_refine + \
                    (torch.ones_like(ori_mask) - Pole_mask - Veg_mask - Sky_mask - Person_mask).mul(ori_mask)

    return mask_refine.detach()

def ClsMeanPixelValue(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_ratio_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation 
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
                # temp_tensor = att_maps[0, i, :, :]
                if (torch.sum(temp_tensor)).item() > 0:
                    # print((torch.sum(temp_tensor)).item())
                    out_cls_tensor[i, 0] = 1.0
                    out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)

                    out_tensor[i, :] = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(temp_tensor)    # b * c * 1 * 1
                    # ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w))
                    # out_tensor[i, :] = ave_fea

    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    # out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor, out_cls_tensor, out_cls_ratio_tensor

def CondGradRepaLoss(fake_img, fake_mask, real_IR, gpu_ids=[]):
    "Conditional Gradient Repair loss for background categories. fake_img: fake vis image. fake_mask: IR seg mask."
    
    ###Conditional Gradient Repair loss for background categories
    _, _, h, w = fake_img.size()
    _, seg_h, seg_w = fake_mask.size()
    fake_mask_resize = F.interpolate(fake_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')

    seg_mask_fake = fake_mask_resize[0]
    IR_bkg_mask = torch.zeros_like(seg_mask_fake)
    IR_bkg_mask = torch.where(seg_mask_fake < 11.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_UC_mask = torch.zeros_like(seg_mask_fake)
    IR_UC_mask = torch.where(seg_mask_fake == 255.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_bkg_fuse_mask = IR_bkg_mask + IR_UC_mask
    getgradmap = Get_gradmag_gray()
    IR_grad = getgradmap(real_IR)
    fake_vis_grad = getgradmap(fake_img)
    IR_grad_bkg = IR_grad.mul(IR_bkg_fuse_mask.expand_as(IR_grad))
    vis_grad_bkg = fake_vis_grad.mul(IR_bkg_fuse_mask.expand_as(fake_vis_grad))
    IR_grad_bkg_sum = torch.sum(IR_grad_bkg)
    
    if IR_grad_bkg_sum > 0:
        # bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg.detach() - vis_grad_bkg)) / IR_grad_bkg_sum.detach()

        IR_grad_bkg_mean = IR_grad_bkg_sum / torch.sum(IR_bkg_fuse_mask)
        IR_grad_bkg_high_mask = torch.zeros_like(IR_grad_bkg)
        IR_grad_bkg_high_mask = torch.where(IR_grad_bkg > IR_grad_bkg_mean, torch.ones_like(IR_grad_bkg), torch.zeros_like(IR_grad_bkg))
        IR_grad_bkg_high = IR_grad_bkg_high_mask.mul(IR_grad_bkg)
        vis_grad_bkg_high = IR_grad_bkg_high_mask.mul(vis_grad_bkg)
        IR_grad_bkg_high_sum = torch.sum(IR_grad_bkg_high)
        if IR_grad_bkg_high_sum > 0:
            bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high)) / IR_grad_bkg_high_sum.detach()
        else:
            bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)
    else:
        bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)

    return bkg_EC_loss



def FakeIRPersonLossv2(Seg_mask, fake_IR, real_vis, gpu_ids=[]):
    "Encouraging the min value of the pedestrian region in the fake IR image "
    "to be larger than the mean value of the vegetation region. "

    # b, c, h, w = fake_IR.size()

    b, c, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    # real_mask_resize = F.interpolate(real_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = GT_mask_resize[0]
    person_mask = torch.zeros_like(GT_mask_resize)
    # light_mask = torch.zeros_like(GT_mask_resize)
    person_mask = torch.where(GT_mask_resize == 11, torch.ones_like(GT_mask_resize), torch.zeros_like(GT_mask_resize))
    light_mask = torch.where(GT_mask_resize == 6, torch.ones_like(GT_mask_resize), torch.zeros_like(GT_mask_resize))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    real_img_norm = (real_vis + 1.0) * 0.5
    fake_IR_gray = .299 * fake_img_norm[:,0:1,:,:] + .587 * fake_img_norm[:,1:2,:,:] + .114 * fake_img_norm[:,2:3,:,:]
    # real_vis_gray = .299 * real_img_norm[:,0:1,:,:] + .587 * real_img_norm[:,1:2,:,:] + .114 * real_img_norm[:,2:3,:,:]
    fake_mean_fea, fake_cls_tensor, _ = ClsMeanPixelValue(fake_IR_gray, Seg_mask.detach(), 19, gpu_ids)
    if (fake_cls_tensor[11, :] * fake_cls_tensor[0, :]) > 0 :
        person_region = (person_mask.expand_as(fake_IR_gray)).mul(fake_IR_gray)
        non_person_mask = torch.ones_like(person_mask) - person_mask
        person_region_padding1 = person_region + non_person_mask ####To get person region min value
        road_mean_value = (fake_mean_fea[0, :]).detach()
        person_min_value = torch.min(person_region_padding1)
        # person_mean_value = fake_mean_fea[11, :]
        # person_dis_loss = F.relu(road_mean_value - person_min_value)
        person_dis_loss = F.relu(road_mean_value - person_min_value) / (road_mean_value + 1e-4)
        # person_dis_loss = F.relu(veg_mean_value - person_mean_value) + (veg_mean_value - person_min_value) ** 2
    else:
        person_dis_loss = 0.0

    return person_dis_loss

def PatchNormFea(input_array, sqrt_patch_num, gpu_ids=[]):
    "Calculate the L2 normalized features for each patch."
    h, w = input_array.size()

    crop_size = h // sqrt_patch_num
    pos_list = list(range(0, h, crop_size))
    patch_num = sqrt_patch_num * sqrt_patch_num
    patch_pixel = crop_size * crop_size
    out_fea_array = torch.zeros(patch_num, patch_pixel).cuda(gpu_ids)
    for p in range(sqrt_patch_num):
        for q in range(sqrt_patch_num):
            idx = p * sqrt_patch_num + q
            pos_h = pos_list[p]
            pos_w = pos_list[q]
            temp_patch = input_array[pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
            temp_patch_view = temp_patch.reshape(1, patch_pixel)
            out_fea_array[idx, :] = torch.div(temp_patch_view, (torch.norm(temp_patch_view) + 1e-4))

    return out_fea_array

def bhw_to_onehot(bhw_tensor1, num_classes, gpu_ids):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes: 20 (19 + uncertain_clsidx)
    Returns: b,num_classes,h,w
    """
    assert bhw_tensor1.ndim == 3, bhw_tensor1.shape
    # assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    # bhw_tensor = bhw_tensor1
    # bhw_tensor[(bhw_tensor == 255)] = 5

    bhw_tensor = torch.zeros_like(bhw_tensor1).cuda(gpu_ids)
    uncertain_clsidx = num_classes - 1
    bhw_tensor = torch.where(bhw_tensor1 == 255, uncertain_clsidx, bhw_tensor1)
    # one_hot = torch.eye(num_classes).index_select(dim=0, index=bhw_tensor.reshape(-1)).cuda(gpu_ids)
    one_hot = torch.eye(num_classes).cuda(gpu_ids).index_select(dim=0, index=bhw_tensor.reshape(-1))
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    out_tensor = one_hot.permute(0, 3, 1, 2)

    return out_tensor[:, :-1, :, :]

def SemEdgeLoss(seg_tensor, GT_mask, num_classes, gpu_ids):
    "Encourage semantic edge prediction consistent with GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm = sm(seg_tensor)
    GT2onehot = bhw_to_onehot(GT_mask, num_classes+1, gpu_ids)
    AvgPool_k3 = nn.AvgPool2d(3, stride=1, padding=1)
    pred_semedge = torch.abs(pred_sm - AvgPool_k3(pred_sm))
    GT_semedge = torch.abs(GT2onehot - AvgPool_k3(GT2onehot))
    if torch.sum(GT_semedge) > 0:
        losses = torch.sum(torch.abs(GT_semedge.detach() - pred_semedge)) / torch.sum(GT_semedge.detach())
    else:
        losses = 0.0

    return losses

def ObjSemEdgeLoss(seg_tensor, GT_mask, num_classes, gpu_ids):
    "Encourage semantic edges of thing categories to be consistent with GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm = sm(seg_tensor)
    GT2onehot = bhw_to_onehot(GT_mask, num_classes+1, gpu_ids)
    AvgPool_k3 = nn.AvgPool2d(3, stride=1, padding=1)
    pred_semedge = torch.abs(pred_sm - AvgPool_k3(pred_sm))
    GT_semedge = torch.abs(GT2onehot - AvgPool_k3(GT2onehot))
    obj_tensor = torch.zeros(1, num_classes, 1, 1).cuda(gpu_ids)
    obj_tensor[:, 11:, :, :] = 1.0
    obj_tensor[:, 2, :, :] = 1.0
    obj_tensor[:, 6, :, :] = 1.0
    obj_tensor[:, 7, :, :] = 1.0
    obj_GT_semedge = GT_semedge.mul(obj_tensor.expand_as(GT_semedge))
    obj_pred_semedge = pred_semedge.mul(obj_tensor.expand_as(pred_semedge))
    if torch.sum(obj_GT_semedge) > 0:
        losses = torch.sum(torch.abs(obj_GT_semedge.detach() - obj_pred_semedge)) / torch.sum(obj_GT_semedge.detach())
    else:
        losses = 0.0

    return losses

def GetFeaMatrixCenter(fea_array, cluster_num, max_iter, gpu_ids):
    "Obtain the central features of each cluster of the feature matrix."
    if gpu_ids == 0:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 1:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:1'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 2:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:2'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 3:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:3'), tqdm_flag=False, iter_limit=max_iter
        )
    else:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:4'), tqdm_flag=False, iter_limit=max_iter
        )

    return cluster_centers.cuda(gpu_ids)

def ClsACALoss(real_vis_fea, cls_mask_real, fake_vis_fea, cls_mask_fake, fea_dim, cluster_num, max_iter, gpu_ids):
    "Calculating adaptive collaborative attention loss for a single class."

    real_fea_cls_masked = real_vis_fea.mul(cls_mask_real.expand_as(real_vis_fea))
    real_fea_cls_matrix = (real_fea_cls_masked.view(fea_dim, -1)).t() ### N * c
    nonZeroRows = torch.abs(real_fea_cls_matrix).sum(dim=1) > 0
    real_fea_cls_matrix = real_fea_cls_matrix[nonZeroRows]
    cls_cluster_center_array = GetFeaMatrixCenter(real_fea_cls_matrix, cluster_num, max_iter, gpu_ids) ###Cn * c
    cls_center_fea_norm = F.normalize(cls_cluster_center_array, p=2, dim=1)
    real_fea_cls_matrix_norm = F.normalize(real_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_real = torch.mm(real_fea_cls_matrix_norm, cls_center_fea_norm.t()) ### Np * Cn
    cls_sim_map_max_real = torch.max(cls_sim_map_real, dim=1)
    cls_fea_sim_mean_real = torch.mean(cls_sim_map_max_real[0])
    sim_map_clustermax_real = torch.max(cls_sim_map_real, dim=0)
    fea_sim_clustermean_real = torch.mean(sim_map_clustermax_real[0])

    fake_fea_cls_masked = fake_vis_fea.mul(cls_mask_fake.expand_as(fake_vis_fea))
    fake_fea_cls_matrix = (fake_fea_cls_masked.view(fea_dim, -1)).t() ### N * c
    fake_fea_cls_matrix_norm = F.normalize(fake_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_fake = torch.mm(fake_fea_cls_matrix_norm, cls_center_fea_norm.t()) ### N * Cn
    cls_sim_map_max_fake = torch.max(cls_sim_map_fake, dim=1)
    cls_fea_sim_mean_fake = torch.sum(cls_sim_map_max_fake[0]) / torch.sum(cls_mask_fake.detach())
    sim_map_clustermax_fake = torch.max(cls_sim_map_fake, dim=0)
    fea_sim_clustermean_fake = torch.mean(sim_map_clustermax_fake[0])
    loss_cls_sim = F.relu(0.9 * cls_fea_sim_mean_real.detach() - cls_fea_sim_mean_fake)
    loss_cls_div = F.relu(0.9 * fea_sim_clustermean_real.detach() - fea_sim_clustermean_fake)
    losses = loss_cls_sim + loss_cls_div

    return losses

def AdaCooAttLoss(real_vis_mask, real_vis_fea, fake_vis_mask, fake_vis_fea, cluster_num, max_iter, gpu_ids):
    "Adaptive Cooperative Attention Loss."
    _, c, h, w = real_vis_fea.size()
    real_vis_mask_resize = F.interpolate(real_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')
    fake_vis_mask_resize = F.interpolate(fake_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')
    Light_mask_real = torch.zeros_like(real_vis_mask_resize)
    Sign_mask_real = torch.zeros_like(real_vis_mask_resize)
    Person_mask_real = torch.zeros_like(real_vis_mask_resize)
    Vehicle_mask_real = torch.zeros_like(real_vis_mask_resize)
    Motor_mask_real = torch.zeros_like(real_vis_mask_resize)

    Light_mask_real = torch.where(real_vis_mask_resize == 6.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Sign_mask_real = torch.where(real_vis_mask_resize == 7.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Person_mask_real = torch.where(real_vis_mask_resize == 11.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Vehicle_mask_real = torch.where((real_vis_mask_resize > 12.0) & (real_vis_mask_resize < 17.0), torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Motor_mask_real = torch.where(real_vis_mask_resize == 17.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))

    Light_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Sign_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Person_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Vehicle_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Motor_mask_fake = torch.zeros_like(fake_vis_mask_resize)

    Light_mask_fake = torch.where(fake_vis_mask_resize == 6.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Sign_mask_fake = torch.where(fake_vis_mask_resize == 7.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Person_mask_fake = torch.where(fake_vis_mask_resize == 11.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Vehicle_mask_fake = torch.where((fake_vis_mask_resize > 12.0) & (fake_vis_mask_resize < 17.0), torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Motor_mask_fake = torch.where(fake_vis_mask_resize == 17.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))

    if (torch.sum(Light_mask_real) > cluster_num) & (torch.sum(Light_mask_fake) > cluster_num):
        loss_light = ClsACALoss(real_vis_fea, Light_mask_real, fake_vis_fea, Light_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_light = 1.0
    else:
        loss_light = 0.0
        idx_light = 0.0

    if (torch.sum(Sign_mask_real) > cluster_num) & (torch.sum(Sign_mask_fake) > cluster_num):
        loss_sign = ClsACALoss(real_vis_fea, Sign_mask_real, fake_vis_fea, Sign_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_sign = 1.0
    else:
        loss_sign = 0.0
        idx_sign = 0.0

    if (torch.sum(Person_mask_real) > cluster_num) & (torch.sum(Person_mask_fake) > cluster_num):
        loss_person = ClsACALoss(real_vis_fea, Person_mask_real, fake_vis_fea, Person_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_person = 1.0
    else:
        loss_person = 0.0
        idx_person = 0.0

    if (torch.sum(Vehicle_mask_real) > cluster_num) & (torch.sum(Vehicle_mask_fake) > cluster_num):
        loss_vehicle = ClsACALoss(real_vis_fea, Vehicle_mask_real, fake_vis_fea, Vehicle_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_vehicle = 1.0
    else:
        loss_vehicle = 0.0
        idx_vehicle = 0.0

    if (torch.sum(Motor_mask_real) > cluster_num) & (torch.sum(Motor_mask_fake) > cluster_num):
        loss_motor = ClsACALoss(real_vis_fea, Motor_mask_real, fake_vis_fea, Motor_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_motor = 1.0
    else:
        loss_motor = 0.0
        idx_motor = 0.0

    obj_cls_num = idx_light + idx_sign + idx_person + idx_vehicle + idx_motor
    if obj_cls_num > 0:
        losses = (loss_light + loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
    else:
        losses = 0.0

    return losses

def FakeIRFGMergeMask(vis_segmask, IR_seg_tensor, gpu_ids):
    "Selecting a suitable foreground mask from the fake IR image and fuse it with the real IR image."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    vis_FG_idx_list = [6, 7, 11, 13, 14, 15, 16, 17]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    IR_road_mask = np.zeros_like(real_IR_segmask)
    IR_road_mask = np.where(real_IR_segmask < 2.0, 1.0, 0.0)
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(vis_GT_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(temp_connect_mask) > 50:
                if vis_FG_idx_list[i] in traffic_sign_list:
                    output_FG_Mask += temp_connect_mask
                elif vis_FG_idx_list[i] in large_FG_list:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
                else:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
    # print(np.sum(output_FG_Mask))

    return torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)


class Vgg16(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features.cuda(gpu_ids)
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        # for x in range(23, 30):
        #     self.to_relu_5_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # h = self.to_relu_5_3(h)
        # h_relu_5_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

def compute_vgg_loss(img, target, gpu_ids=[]):
    # img_vgg = vgg_preprocess(img)
    # target_vgg = vgg_preprocess(target)
    vgg = Vgg16(gpu_ids)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    loss_mse = torch.nn.MSELoss()
    img_fea = vgg(img)
    target_fea = vgg(target)
    content_loss = 0.0
    for j in range(4):
        content_loss += loss_mse(img_fea[j], target_fea[j])

    return content_loss * 0.25

class Get_gradmag_gray(nn.Module):
    "To obtain the magnitude values of the gradients at each position."
    def __init__(self):
        super(Get_gradmag_gray, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x_norm = (x + 1) / 2
        x_norm = (.299*x_norm[:,0:1,:,:] + .587*x_norm[:,1:2,:,:] + .114*x_norm[:,2:3,:,:])
        x0_v = F.conv2d(x_norm, self.weight_v, padding = 1)
        x0_h = F.conv2d(x_norm, self.weight_h, padding = 1)

        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x_gradmagn

class Get_gradmag_mask(nn.Module):
    "To obtain the magnitude values of the gradients at each position."
    def __init__(self):
        super(Get_gradmag_mask, self).__init__()
        kernel_v = [[-1, -1, -1], 
                    [0, 0, 0], 
                    [1, 1, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-1, 0, 1], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        # x_norm = (x + 1) / 2
        # x_norm = (.299*x_norm[:,0:1,:,:] + .587*x_norm[:,1:2,:,:] + .114*x_norm[:,2:3,:,:])
        x0_v = F.conv2d(x, self.weight_v, padding = 1)
        x0_h = F.conv2d(x, self.weight_h, padding = 1)

        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2))
        out = torch.zeros_like(x_gradmagn)
        out[x_gradmagn > 0] = 1.0

        return x_gradmagn

def FuseCOIMask(input_mask, gpu_ids):

    b, c, h, w = input_mask.size()

    mask_road_edge = torch.zeros_like(input_mask)
    mask_building_edge = torch.zeros_like(input_mask)
    mask_vehicle_edge = torch.zeros_like(input_mask)

    mask_road_edge[input_mask != 0] = 255.0
    mask_building_edge[input_mask != 2] = 255.0
    # mask_person_edge[image_array != 11] = 255.0
    mask_vehicle_edge[input_mask < 13] = 255.0
    mask_vehicle_edge[input_mask > 16] = 255.0

    mask_road_edge_broad = 255.0 * torch.ones((b, c, h+2, w+2)).cuda(gpu_ids)
    mask_building_edge_broad = 255.0 * torch.ones((b, c, h+2, w+2)).cuda(gpu_ids)
    mask_vehicle_edge_broad = 255.0 * torch.ones((b, c, h+2, w+2)).cuda(gpu_ids)

    mask_road_edge_broad[:, :, 1:(h+1), 1:(w+1)] = mask_road_edge
    mask_building_edge_broad[:, :, 1:(h+1), 1:(w+1)] = mask_building_edge
    mask_vehicle_edge_broad[:, :, 1:(h+1), 1:(w+1)] = mask_vehicle_edge

    getMaskGrad = Get_gradmag_mask()

    road_edge = getMaskGrad(mask_road_edge_broad)
    building_edge = getMaskGrad(mask_building_edge_broad)
    vehicle_edge = getMaskGrad(mask_vehicle_edge_broad)

    fuse_edge = road_edge[:, :, 1:(h+1), 1:(w+1)] + building_edge[:, :, 1:(h+1), 1:(w+1)] + vehicle_edge[:, :, 1:(h+1), 1:(w+1)]
    Contour_OI = torch.zeros_like(input_mask)
    Contour_OI[fuse_edge > 0] = 1.0

    non_edge_mask = torch.ones_like(input_mask) - Contour_OI
    fuse_COI_segmask = non_edge_mask * (input_mask) + Contour_OI * 12.0

    return fuse_COI_segmask.detach()




def SGALoss(real_IR_edgemap, fake_vis_gradmap, sqrt_patch_num, gradient_th):
    "SGA Loss. The ratio of the gradient at the edge location to the maximum gradient in "
    "its neighborhood is encouraged to be greater than a given threshold."

    b, c, h, w = fake_vis_gradmap.size()
    # patch_num = sqrt_patch_num * sqrt_patch_num
    AAP_module = nn.AdaptiveAvgPool2d(sqrt_patch_num)
    real_IR_edgemap_pooling = AAP_module(real_IR_edgemap.expand_as(fake_vis_gradmap))
    if torch.sum(real_IR_edgemap) > 0:
        pooling_array = real_IR_edgemap_pooling[0].detach().cpu().numpy()
        h_nonzero, w_nonzero = np.nonzero(pooling_array[0])
        patch_idx_rand = np.random.randint(0, len(h_nonzero))
        patch_idx_x = h_nonzero[patch_idx_rand]
        patch_idx_y = w_nonzero[patch_idx_rand]
        crop_size = h // sqrt_patch_num
        pos_list = list(range(0, h, crop_size))

        pos_h = pos_list[patch_idx_x]
        pos_w = pos_list[patch_idx_y]
        # rand_patch = self.Tensor(b, c, crop_size, crop_size)
        rand_edgemap_patch = real_IR_edgemap[:, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
        rand_gradmap_patch = fake_vis_gradmap[:, :, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]

        sum_edge_pixels = torch.sum(rand_edgemap_patch) + 1
        # print('Sum_edge_pixels of IR edge map is: ', sum_edge_pixels.detach().cpu().numpy())
        fake_grad_norm = rand_gradmap_patch / torch.max(rand_gradmap_patch)
        losses = (torch.sum(F.relu(gradient_th * rand_edgemap_patch - fake_grad_norm))) / sum_edge_pixels
    else:
        losses = 0

    return losses

def get_center_patch(real_img, fake_img, gpu_ids):
    b, c, h, w = fake_img.size()
    pos_h = h // 4
    patch_h = h // 2
    real_img_patch = (real_img[:, :, pos_h:(pos_h + patch_h), pos_h:(pos_h + patch_h)]).cuda(gpu_ids[0])
    fake_img_patch = fake_img[:, :, pos_h:(pos_h + patch_h), pos_h:(pos_h + patch_h)]

    return real_img_patch, fake_img_patch

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )

# Defines the total variation (TV) loss, which encourages spatial smoothness in the generated image.
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


###Add an CSG module
class ResnetGenEncoder_CSG2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder_CSG2, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(5, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]


        self.model = nn.Sequential(*model)
        self.csg = CatersianGrid()


    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, torch.cat((input, self.csg(input)), 1), self.gpu_ids)
        return self.model(torch.cat((input, self.csg(input)), 1))

####Add 1 Pyramid Guided Attention Block v4 before ResBlock groups
class ResnetGenDecoderv1(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoderv1, self).__init__()
        self.gpu_ids = gpu_ids
        

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(32, int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.PReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, in_dim, num_classes, norm_layer):
        super(ASPP, self).__init__()
        
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(in_dim, 64, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(in_dim, 64, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(in_dim, 64, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(in_dim, 64, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(in_dim, 128, 1, stride=1, bias=False),
        #                                      norm_layer(128),
        #                                      nn.PReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_dim, 64, 1, stride=1, bias=False),
                                             nn.PReLU())
        self.conv1 = nn.Conv2d(320, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv_1x1_4(self.dropout(x))

        return out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminatorSN, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                SNConv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            SNConv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.PReLU(),
            \
            SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3

# Defines the SegmentorHead.
class SegmentorHead(nn.Module):
    def __init__(self, input_nc, num_classes=19, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(SegmentorHead, self).__init__()
        self.gpu_ids = gpu_ids

        self.model_seg = ASPP(input_nc, num_classes, norm_layer)

    def forward(self, input):

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs = nn.parallel.data_parallel(self.model_seg, input, self.gpu_ids)
        else:
            outs = self.model_seg(input)
        return outs

# Defines the SegmentorHeadv2. Zero Padding and CSG for positional encoding.
class SegmentorHeadv2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, num_classes=19, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='zero'):
        super(SegmentorHeadv2, self).__init__()
        assert(n_blocks >= 0)
        self.gpu_ids = gpu_ids

        # model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
        #                    bias=use_bias),
        #          norm_layer(ngf),
        #          nn.PReLU()]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(5, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        
        mult = 2**(n_downsampling)
        # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=4, stride=2,
        #                                  padding=1, output_padding=0,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.PReLU()]
        # model += [ASPP(int(ngf * mult / 2), num_classes, norm_layer)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [ASPP(int(ngf), num_classes, norm_layer)]

        self.model = nn.Sequential(*model)
        self.csg = CatersianGrid()

    def forward(self, input):

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs, seg_fea = nn.parallel.data_parallel(self.model, torch.cat((input, self.csg(input)), 1), self.gpu_ids)
        else:
            outs, seg_fea = self.model(torch.cat((input, self.csg(input)), 1))
        return outs, seg_fea

class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group in self.optimizers[dom_a].param_groups:
            param_group['lr'] = new_lr

        for param_group in self.optimizers[dom_b].param_groups:
            param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params

class S_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(S_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def init_optimizers(self, opt, lr, betas):
         self.optimizers = [opt(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=betas) \
                           for net in self.networks]

    def forward(self, input, domain):
        segmentor = self.networks[domain]
        return segmentor.forward(input)

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group_a in self.optimizers[dom_a].param_groups:
            param_group_a['lr'] = new_lr
            print('Learning rate of SegA is: %.4f.' % param_group_a['lr'])

        for param_group_b in self.optimizers[dom_b].param_groups:
            # print(param_group_b['lr'])
            param_group_b['lr'] = new_lr
            print('Learning rate of SegB is: %.4f.' % param_group_b['lr'])

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Segmentors' % len(self.networks) +'\n'+ \
            'Number of parameters per Segmentor: %d' % t_params

class DF_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(DF_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminatorF = self.networks[domain]
        return discriminatorF.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Object Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Object Discriminator: %d' % t_params

class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
