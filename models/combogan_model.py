import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torch.nn as nn
# import json

class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'

    def __init__(self, opt):
        super(ComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_Gen_type,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)
            self.netS = networks.define_S(opt.input_nc, opt.ngf, opt.netG_n_blocks // 2, self.n_domains, opt.num_class, opt.norm, opt.use_dropout, self.gpu_ids)
            
        #############Edit by lfy###############
            self.EdgeMap_A = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.EdgeMap_B = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

            
        #######################################
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)
                self.load_network(self.netS, 'S', which_epoch)
                

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            self.fake_pools_patch = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # self.fake_pools_mask = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1 = torch.nn.SmoothL1Loss()
            self.L1_sum = torch.nn.SmoothL1Loss(reduction='sum')
            self.downsample = torch.nn.AvgPool2d(3, stride=2)
            self.criterionCycle = self.L1
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t)) 
            self.criterionLatent = lambda y,t : self.L1(y, t.detach())
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0],f[0],v) + \
                                                networks.GANLoss(r[1],f[1],v) + \
                                                networks.GANLoss(r[2],f[2],v)) / 3 
            self.criterionGANv2 = lambda r,f,v : networks.GANLoss(r,f,v)
            self.criterionSGAIR = lambda y,t,r,v : networks.SGALoss(y, t, r, v)
            self.criterionSGAVis = lambda y,t,r,v : networks.SGALoss(y, t, r, v)
            self.criterionCGR = lambda y,t,r,v : networks.CondGradRepaLoss(y, t, r, v)
            self.criterionACA = lambda y,t,r,v,f,m,l : networks.AdaCooAttLoss(y, t, r, v, f, m, l)
            self.criterionIRClsDis = lambda y,t,r,v : networks.FakeIRPersonLossv2(y, t, r, v)
            self.criterionSemEdge = lambda y,t,r,v : networks.SemEdgeLoss(y, t, r, v)
            self.criterionSSIM = networks.SSIM_Loss(win_size=self.opt.ssim_winsize, data_range=1.0, size_average=True, channel=3)
            self.criterionTV = networks.TVLoss(TVLoss_weight=1)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netS.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle = [0]*self.n_domains
            self.loss_sga, self.loss_tv = [0]*self.n_domains, [0]*self.n_domains
            self.loss_S_enc, self.loss_S_rec = [0]*self.n_domains, [0]*self.n_domains
            self.loss_DS, self.loss_SR = [0]*self.n_domains, [0]*self.n_domains

            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward
            self.get_gradmag = networks.Get_gradmag_gray()
            self.UpdateVisGT = lambda y,t,r : networks.UpdateSegGT(y, t, r)
            self.UpdateIRGTv1 = lambda y,t,v,r,l : networks.OnlSemDisModule(y, t, v, r, l)
            self.get_FG_MergeMask = lambda y,t,v : networks.FakeIRFGMergeMask(y, t, v)
            

            self.seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
            self.kl_loss = torch.nn.KLDivLoss(size_average=False)
            self.sm = torch.nn.Softmax(dim = 1)
            self.log_sm = torch.nn.LogSoftmax(dim = 1)
            self.num_classes = opt.num_class
            self.max_value = opt.max_value
            self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda(self.gpu_ids[0]) + 1
            self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda(self.gpu_ids[0]) + 1
            self.only_hard_label = opt.only_hard_label
            # self.class_balance = opt.class_balance
            self.often_balance = opt.often_balance

            ####Added options
            self.patch_num_sqrt = opt.sqrt_patch_num
            self.grad_th_vis = opt.grad_th_vis
            self.grad_th_IR = opt.grad_th_IR
            self.IR_crop_txt = opt.IR_FG_txt
            self.IR_memory_txt = opt.IR_patch_classratio_txt
            self.num_ACA_cluster = opt.num_ACA_cluster

            self.SGA_start_ep = opt.SGA_start_epoch
            self.SGA_fullload_ep = opt.SGA_fullload_epoch
            if self.SGA_fullload_ep == 0:
                self.lambda_sga = opt.lambda_sga
                self.lambda_tv = opt.lambda_tv
            else:
                self.lambda_sga = 0.0
                self.lambda_tv = 0.0

            self.SSIM_start_ep = opt.SSIM_start_epoch
            self.SSIM_fullload_ep = opt.SSIM_fullload_epoch
            if self.SSIM_fullload_ep == 0:
                self.lambda_ssim = opt.lambda_ssim
            else:
                self.lambda_ssim = 0.0

            self.netS_start_ep = opt.netS_start_epoch
            self.netS_end_ep = opt.netS_end_epoch
            self.updateGT_ep = opt.updateGT_start_epoch
            self.vis_prob_th = opt.vis_prob_th
            self.IR_prob_th = opt.IR_prob_th

            ###This index is used to control when real NTIR images are utilized to train the segmentation network for domain B.
            self.DB_GT_update_idx = 0.0

            self.IR_img_memory_idx = 0.0 ###This index is used to control when to memorize segmentation mask information for domain B.
            self.netS_freezing_idx = 0.0 ###This index is used to control when to freeze the weights of the segmented network.
            self.updateGT_DB_ep = self.updateGT_ep + 10
            if self.netS_start_ep == 0:
                self.lambda_sc = opt.lambda_sc               
            else:
                self.lambda_sc = 0.0

            if self.updateGT_ep == 0:
                ###This index is used to control when to update the segmentation pseudo-labels of domain A using 
                # high-confidence predictions.
                self.DA_GT_update_idx = 1.0  
                self.lambda_cgr = opt.lambda_CGR             
            else:
                self.DA_GT_update_idx = 0.0
                self.lambda_cgr = 0.0

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')
#############Edit by lfy###############
    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
            input_EM_A = input['EMA']
            self.EdgeMap_A.resize_(input_EM_A.size()).copy_(input_EM_A)
            input_EM_B = input['EMB']
            self.EdgeMap_B.resize_(input_EM_B.size()).copy_(input_EM_B)
            input_SM_A = input['SMA']
            self.SegMask_A = input_SM_A.long().cuda(self.gpu_ids[0])
            self.B_idx = input['IdxB']
            self.B_pos_h = input['PosH']
            self.B_pos_w = input['PosW']

        self.image_paths = input['path']
#######################################

    def test(self, output_only=False):
        with torch.no_grad():
            if output_only:
                # cache encoding to not repeat it everytime
                encoded = self.netG.encode(self.real_A, self.DA)
                for d in range(self.n_domains):
                    if d == self.DA and not self.opt.autoencode:
                        continue
                    fake = self.netG.decode(encoded, d)
                    self.visuals = [fake]
                    self.labels = ['fake_%d' % d]
            else:
                self.visuals = [self.real_A]
                self.labels = ['real_%d' % self.DA]

                # cache encoding to not repeat it everytime
                encoded = self.netG.encode(self.real_A, self.DA)
                for d in range(self.n_domains):
                    if d == self.DA and not self.opt.autoencode:
                        continue
                    fake = self.netG.decode(encoded, d)
                    self.visuals.append( fake )
                    self.labels.append( 'fake_%d' % d )
                    if self.opt.reconstruct:
                        rec = self.netG.forward(fake, d, self.DA)
                        self.visuals.append( rec )
                        self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, pred_real, fake, domain):
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)

    def update_class_criterion(self, labels):
        weight = torch.FloatTensor(self.num_classes).zero_().cuda(self.gpu_ids[0])
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().cuda(self.gpu_ids[0])
        often = torch.FloatTensor(self.num_classes).zero_().cuda(self.gpu_ids[0])
        often += 1
        # print(labels.shape)
        n, h, w = labels.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(labels==i)
            if count[i] < 32*32*n: #small objective
                weight[i] = self.max_value
        if self.often_balance:
            often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        # print(self.class_weight)
        return nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255)

    def count_IR_class_ratio(self, labels):
        count = np.zeros((1, self.num_classes))
        n, h, w = labels.shape
        for i in range(self.num_classes):
            count[0, i] = ((torch.sum(labels == i)).item()) / (h * w)
        ###Small sample categories includes six categories: traffic light, traffic sign, person, truck, bus, and motorcycle. 
        ###Their indexes are 6, 7, 11, 14, 15 and 17, respectively.
        SSC_idx = np.zeros((self.num_classes, 1))
        SSC_idx[6:8, 0] = 1.0
        SSC_idx[11, 0] = 1.0
        SSC_idx[14:16, 0] = 1.0
        SSC_idx[17, 0] = 1.0
        SSC_ratio_sum = np.dot(count, SSC_idx)
        if SSC_ratio_sum == 0.0:
            cls_ratio_norm = np.zeros((1, 6))
        else:
            SSC_ratio = np.zeros((1, 6))
            SSC_ratio[0, 0:2] = count[0, 6:8]
            SSC_ratio[0, 2] = count[0, 11]
            SSC_ratio[0, 3:5] = count[0, 14:16]
            SSC_ratio[0, 5] = count[0, 17]
            count_submean = SSC_ratio - (SSC_ratio_sum / 6.0)
            cls_ratio_norm = count_submean / np.linalg.norm(count_submean, axis=1, keepdims=True)
        # print(cls_ratio_norm)

        return cls_ratio_norm

    def update_label(self, labels, prediction):
        criterion = nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255, reduction = 'none')
        #criterion = self.seg_loss
        loss = criterion(prediction, labels)
        # print('original loss: %f'% self.seg_loss(prediction, labels) )
        #mm = torch.median(loss)
        loss_data = loss.data.cpu().numpy()
        mm = np.percentile(loss_data[:], self.only_hard_label)
        #print(m.data.cpu(), mm)
        labels[loss < mm] = 255
        return labels

    def backward_G(self):
        encoded_A = self.netG.encode(self.real_A, self.DA)
        encoded_B = self.netG.encode(self.real_B, self.DB)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:
            # Same encoder and decoder should recreate image
            idt_A = self.netG.decode(encoded_A, self.DA)
            loss_idt_A = self.criterionIdt(idt_A, self.real_A)
            idt_B = self.netG.decode(encoded_B, self.DB)
            loss_idt_B = self.criterionIdt(idt_B, self.real_B)
        else:
            loss_idt_A, loss_idt_B = 0, 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(encoded_A, self.DB)
        pred_fake_B = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake_B, False)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(encoded_B, self.DA)
        pred_fake_A = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake_A, False)
        # Forward cycle loss
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_cyc + \
                                    (self.criterionSSIM((self.rec_A + 1) / 2, (self.real_A + 1) / 2)) * self.lambda_ssim

        # Backward cycle loss
        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_cyc + \
                                    (self.criterionSSIM((self.rec_B + 1) / 2, (self.real_B + 1) / 2)) * self.lambda_ssim



        # Optional total variation loss on generate fake images, added by lfy
        self.loss_tv[self.DA] = self.lambda_tv * self.criterionTV(self.fake_A)
        self.loss_tv[self.DB] = self.lambda_tv * self.criterionTV(self.fake_B)

        "Random size for segmentation network training. Then, retain original image size."
        if self.netS_freezing_idx == 0.0:
            rand_scale = torch.randint(32, 80, (1, 1)) #32, 80
            rand_size = int(rand_scale.item() * 4)
            rand_size_B = int(rand_scale.item() * 4)
        else:
            rand_size_B = 256
            rand_size = 256

        SegMask_A_s = F.interpolate(self.SegMask_A.expand(1, 1, 256, 256).float(), size=[rand_size, rand_size], mode='nearest')
        real_A_s = F.interpolate(self.real_A, size=[rand_size, rand_size], mode='bilinear', align_corners=False) ###torch.flip(input_A, [3])
        fake_B_s = F.interpolate(self.fake_B, size=[rand_size, rand_size], mode='bilinear', align_corners=False)
        fake_A_s = F.interpolate(self.fake_A, size=[rand_size, rand_size], mode='bilinear', align_corners=False)
        real_B_s = F.interpolate(self.real_B, size=[rand_size_B, rand_size_B], mode='bilinear', align_corners=False) ###torch.flip(input_A, [3])

        real_A_s_ds = F.interpolate(real_A_s, size=[rand_size // 2, rand_size // 2], mode='bilinear', align_corners=False) ###torch.flip(input_A, [3])
        fake_B_s_ds = F.interpolate(fake_B_s, size=[rand_size // 2, rand_size // 2], mode='bilinear', align_corners=False)
        
        # print(self.SegMask_A.shape)
        # print(real_A_s.shape)

        self.real_A_pred, _ = self.netS.forward(real_A_s, self.DA)
        fake_B_pred, _ = self.netS.forward(fake_B_s, self.DB)
        self.real_B_pred, _ = self.netS.forward(real_B_s, self.DB)
        fake_A_pred, _ = self.netS.forward(fake_A_s, self.DA)

        self.fake_B_pred_d, _ = self.netS.forward(fake_B_s.detach(), self.DB)
        self.fake_A_pred_d, _ = self.netS.forward(fake_A_s.detach(), self.DA)

        if self.DB_GT_update_idx == 0.0:
            #0-40 epoch
            self.loss_S_rec[self.DB] = 0.0
            self.loss_S_enc[self.DB] = 0.0

            if self.DA_GT_update_idx == 1.0:
                ####30-40 epoch, training semantic segmentation networks for domain A with updating segmentation GT,  
                ###and training semantic segmentation networks for domain B by pseudo-labels of domain A and pseudo-NTIR images
                # self.loss_S_enc[self.DB] = 0.0
                self.SegMask_A_update = self.UpdateVisGT(self.real_A_pred.detach(), SegMask_A_s[0].long(), self.vis_prob_th)
                self.seg_loss = self.update_class_criterion(self.SegMask_A_update.long())

                ####
                self.loss_S_enc[self.DA] = self.lambda_sc * (self.seg_loss(self.real_A_pred, self.SegMask_A_update.long()) + \
                                            0.5 * self.criterionSemEdge(self.real_A_pred, self.SegMask_A_update.long(), 19, self.gpu_ids[0]))
                self.loss_S_rec[self.DA] = self.lambda_sc * self.seg_loss(self.fake_B_pred_d, self.SegMask_A_update.long()) 
                self.SegMask_B_update = self.UpdateIRGTv1(self.real_B_pred.detach(), self.fake_A_pred_d, 255 * torch.ones_like(SegMask_A_s[0].long()), real_B_s, self.IR_prob_th)

            else:
                ####0-20 epoch, self.lambda_sc is set to 0.
                ####20-30 epoch, training semantic segmentation networks for domain A without updating segmentation GT
                self.loss_S_rec[self.DA] = 0.0
                seg_loss_A = self.update_class_criterion(SegMask_A_s[0].long())
                ####
                self.loss_S_enc[self.DA] = self.lambda_sc * seg_loss_A(self.real_A_pred, SegMask_A_s[0].long())
                self.SegMask_A_update = SegMask_A_s[0].long().detach()
                self.SegMask_B_update = 255 * torch.ones_like(SegMask_A_s[0].long())
                
        else:
            # #40-80
            if self.netS_freezing_idx < 1:
                ####40-75 epoch, training semantic segmentation networks for domain A with updating segmentation GT,
                ####and training semantic segmentation networks for domain B by both real-NTIR and pseudo-NTIR images.
                self.loss_S_rec[self.DB] = 0.0
                self.SegMask_A_update = self.UpdateVisGT(self.real_A_pred.detach(), SegMask_A_s[0].long(), self.vis_prob_th)
                seg_loss_A = self.update_class_criterion(self.SegMask_A_update.long())
                self.loss_S_enc[self.DA] = self.lambda_sc * (seg_loss_A(self.real_A_pred, self.SegMask_A_update.long()) + \
                                            0.5 * self.criterionSemEdge(self.real_A_pred, self.SegMask_A_update.long(), 19, self.gpu_ids[0]))
                self.loss_S_rec[self.DA] = self.lambda_sc * seg_loss_A(fake_B_pred, self.SegMask_A_update.long())
                self.SegMask_B_update = self.UpdateIRGTv1(self.real_B_pred.detach(), self.fake_A_pred_d, 255 * torch.ones_like(SegMask_A_s[0].long()), real_B_s, self.IR_prob_th)
                
                seg_loss_B = self.update_class_criterion(self.SegMask_B_update.long())
                self.loss_S_enc[self.DB] = self.lambda_sc * seg_loss_B(self.real_B_pred, self.SegMask_B_update.long())

            else:
                ####75-80 epoch, constraining semantic consistency after fixing segmentation networks of the two domains.
                self.loss_S_enc[self.DA] = 0.0
                self.loss_S_enc[self.DB] = 0.0

                self.SegMask_A_update = self.UpdateVisGT(self.real_A_pred.detach(), SegMask_A_s[0].long(), self.vis_prob_th)
                seg_loss_A = self.update_class_criterion(self.SegMask_A_update.long())
                
                self.loss_S_rec[self.DA] = self.lambda_sc * seg_loss_A(fake_B_pred, self.SegMask_A_update.long())
                # print(self.SegMask_B_update.size())
                self.SegMask_B_update = self.UpdateIRGTv1(self.real_B_pred.detach(), self.real_B_pred.detach(), 255 * torch.ones(1, 256, 256).cuda((self.gpu_ids[0])), real_B_s, 0.95)
                # print(self.SegMask_B_update.size())
                SegMask_B_update2 = F.interpolate(self.SegMask_B_update.expand(1, 1, 256, 256).float(), size=[rand_size, rand_size], mode='nearest')
                seg_loss_B = self.update_class_criterion(SegMask_B_update2[0].long())
                self.loss_S_rec[self.DB] = self.lambda_sc * seg_loss_B(fake_A_pred, SegMask_B_update2[0].long())

        ######################################

        if self.IR_img_memory_idx == 1.0:
            with open(self.IR_crop_txt, 'r') as IRFGList:
                self.txt_lines = IRFGList.readlines()
            
            mem_IR_idx_list = []
            for idx_img in range(len(self.txt_lines) - 1):
                str_list_idx = self.txt_lines[idx_img].strip('\n')
                line_list_idx = str_list_idx.split(' ')
                mem_IR_idx_list.append(int(line_list_idx[0])) 
            # if (str(self.B_idx[0].item()) + '\n') not in self.txt_lines:
            if self.B_idx[0].item() not in mem_IR_idx_list:
                with open(self.IR_crop_txt, "a+") as FGtxt:
                    line_content = str(self.B_idx[0].item()) + ' ' + str(self.B_pos_h[0].item()) + \
                                    ' ' + str(self.B_pos_w[0].item()) + '\n'
                    FGtxt.write(line_content)
                with open(self.IR_memory_txt, "a+") as ratio_txt:
                    cls_ratio = self.count_IR_class_ratio(self.SegMask_B_update)
                    list_content = cls_ratio.tolist()
                    line_content = str(list_content[0]).strip('[').strip(']')
                    ratio_txt.writelines(str(line_content) + '\n')

        # Optional Scale Robustness Loss on generated fake images, added by lfy
        if self.DB_GT_update_idx > 0.0:
            ####40-80 epoch
            inv_idx = torch.rand(1)
            if inv_idx > 0.5:
                real_A_ds = F.interpolate(self.real_A, size=[128, 128], mode='bilinear', align_corners=False)
                real_B_ds = F.interpolate(self.real_B, size=[128, 128], mode='bilinear', align_corners=False)
                encoded_real_A_ds = self.netG.encode(real_A_ds, self.DA)
                fake_B_real_A_ds = self.netG.decode(encoded_real_A_ds, self.DB)
                encoded_real_B_ds = self.netG.encode(real_B_ds, self.DB)
                fake_A_real_B_ds = self.netG.decode(encoded_real_B_ds, self.DA)

                fake_A_ds = F.interpolate(self.fake_A, size=[128, 128], mode='bilinear', align_corners=False)
                fake_B_ds = F.interpolate(self.fake_B, size=[128, 128], mode='bilinear', align_corners=False)
            else:
                real_A_ds = F.interpolate(self.real_A, size=[384, 384], mode='bilinear', align_corners=False)
                real_B_ds = F.interpolate(self.real_B, size=[384, 384], mode='bilinear', align_corners=False)
                encoded_real_A_ds = self.netG.encode(real_A_ds, self.DA)
                fake_B_real_A_ds = F.interpolate(self.netG.decode(encoded_real_A_ds, self.DB), size=[256, 256], mode='bilinear', align_corners=False)
                encoded_real_B_ds = self.netG.encode(real_B_ds, self.DB)
                fake_A_real_B_ds = F.interpolate(self.netG.decode(encoded_real_B_ds, self.DA), size=[256, 256], mode='bilinear', align_corners=False)

                fake_A_ds = self.fake_A
                fake_B_ds = self.fake_B


            self.loss_SR[self.DA] = self.lambda_cyc * self.L1(fake_B_real_A_ds, fake_B_ds.detach()) + \
                                    (self.criterionSSIM((fake_B_real_A_ds + 1) / 2, (fake_B_ds.detach() + 1) / 2))
            self.loss_SR[self.DB] = self.lambda_cyc * self.L1(fake_A_real_B_ds, fake_A_ds.detach()) + \
                                    (self.criterionSSIM((fake_A_real_B_ds + 1) / 2, (fake_A_ds.detach() + 1) / 2))

        else:
            self.loss_SR[self.DA] = 0.0
            self.loss_SR[self.DB] = 0.0

        ######################

        if self.netS_freezing_idx == 1.0:
            ####Conditional Gradient Repair Losses
            loss_CGR = self.criterionCGR(self.fake_A, self.SegMask_B_update.detach(), self.real_B.detach(), self.gpu_ids[0])
            ####Domain-specific losses (i.e., loss_DS) include the CGR loss and the ACA loss.
            self.loss_DS[self.DB] = self.lambda_cgr * loss_CGR
            self.loss_DS[self.DA] = self.criterionACA(self.SegMask_A_update.detach(), encoded_A.detach(), \
                                        self.SegMask_B_update.detach(), rec_encoded_B, self.num_ACA_cluster, 100000, self.gpu_ids[0])

        else:
            self.loss_DS[self.DA] = 0.0
            self.loss_DS[self.DB] = 0.0

        # Optional structured gradient alignment loss on generate fake images, added by lfy
        ####The last two terms of loss_sga[self.DA] denote the monochromatic regularization term and the temperature 
        # regularization term, respectively.
        self.loss_sga[self.DA] = self.lambda_sga * self.criterionSGAVis(self.EdgeMap_A, self.get_gradmag(self.fake_B), self.patch_num_sqrt, self.grad_th_vis) + \
                                    torch.max(torch.max(self.fake_B, 1)[0] - torch.min(self.fake_B, 1)[0]) + \
                                    self.lambda_ssim * self.criterionIRClsDis(self.SegMask_A.detach(), self.fake_B, self.real_A.detach(), self.gpu_ids[0])
        self.loss_sga[self.DB] = self.lambda_sga * self.criterionSGAIR(self.EdgeMap_B, self.get_gradmag(self.fake_A), self.patch_num_sqrt, self.grad_th_IR)

        ######################################

        # Optional cycle loss on encoding space
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # Optional loss on downsampled image before and after
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0


        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) + \
                 (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                 (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                 (loss_fwd_A + loss_fwd_B) * self.lambda_fwd + (self.loss_S_enc[self.DA] + self.loss_S_enc[self.DB]) + \
                 (self.loss_tv[self.DA] + self.loss_tv[self.DB]) + self.loss_S_rec[self.DA] + self.loss_S_rec[self.DB] + \
                 (self.loss_sga[self.DA] + self.loss_sga[self.DB]) + (self.loss_DS[self.DA] + self.loss_DS[self.DB]) + \
                 (self.loss_SR[self.DA] + self.loss_SR[self.DB])  ######Edit by lfy
                
        loss_G.backward()



    def optimize_parameters(self):
        # print(self.real_A.shape)
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)

        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.netS.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        if self.netS_freezing_idx == 0.0:
            self.netS.step_grads(self.DA, self.DB)
        # else:
        #     print('Segmentation Net is frezzing.')
        # self.netS.step_grads(self.DA, self.DB)
        
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)
        

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses, sce_losses, scr_losses, ds_losses, sga_losses, sr_losses = \
            extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle), extract(self.loss_S_enc), \
                extract(self.loss_S_rec), extract(self.loss_DS), extract(self.loss_sga), extract(self.loss_SR)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses), ('SCE', sce_losses), \
            ('SCR', scr_losses), ('DS', ds_losses), ('SGA', sga_losses), ('SR', sr_losses)])   #########Edit by lfy

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def get_current_visuals2(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
            self.masks = [self.real_A_pred, self.fake_B_pred_d, self.real_B_pred, self.fake_A_pred_d, self.SegMask_A_update, self.SegMask_B_update]
            self.mask_labels = ['eseg_A', 'rseg_A', 'eseg_B', 'rseg_B', 'GT_A', 'GT_B']
            images = [util.tensor2im(v.data) for v in self.visuals]
            # print(type(images))
            seg_masks = [util.colorize_mask(v.data) for v in self.masks]
            out = OrderedDict(zip(self.labels + self.mask_labels, images + seg_masks))
        else:
            images = [util.tensor2im(v.data) for v in self.visuals]
            out = OrderedDict(zip(self.labels, images))
        return out

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netS, 'S', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac

        #######Edit by lfy, weight gain########
        if self.SGA_fullload_ep != 0:
            if curr_iter > (self.SGA_start_ep - 1):
                if curr_iter > (self.SGA_fullload_ep - 1):
                    self.lambda_sga = self.opt.lambda_sga
                    self.lambda_tv = self.opt.lambda_tv
                else:
                    gain_frac = (curr_iter - self.SGA_start_ep) / (self.SGA_fullload_ep - self.SGA_start_ep)
                    self.lambda_sga = gain_frac * self.opt.lambda_sga
                    self.lambda_tv = gain_frac * self.opt.lambda_tv

        if self.SSIM_fullload_ep != 0:
            if curr_iter > (self.SSIM_start_ep - 1):
                if curr_iter > (self.SSIM_fullload_ep - 1):
                    self.lambda_ssim = self.opt.lambda_ssim
                else:
                    gain_frac = (curr_iter - self.SSIM_start_ep) / (self.SSIM_fullload_ep - self.SSIM_start_ep)
                    self.lambda_ssim = gain_frac * self.opt.lambda_ssim

        if self.updateGT_ep != 0:
            if curr_iter > (self.netS_end_ep - 1):
                self.lambda_sc = self.opt.lambda_sc
                self.DB_GT_update_idx = 1.0
            else:
                if curr_iter > (self.netS_start_ep - 1):
                    self.lambda_sc = self.opt.lambda_sc
                    if curr_iter > (self.updateGT_DB_ep - 1):
                        self.DB_GT_update_idx = 1.0
                    else:
                        if curr_iter > (self.updateGT_ep - 1):
                            self.DA_GT_update_idx = 1.0
                        else:
                            self.DA_GT_update_idx = 0.0

        if curr_iter > (self.netS_end_ep - 1):
            self.netS_freezing_idx = 1.0
            self.IR_img_memory_idx = 1.0
            self.lambda_cgr = self.opt.lambda_CGR
            for param in self.netS.parameters():
                param.requires_grad = False
