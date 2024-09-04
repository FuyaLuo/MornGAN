from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')
        self.parser.add_argument('--IR_edge_path', type=str, default='./FLIR_IR_edge_map/', help='the folder to load IR image edge map')
        self.parser.add_argument('--Vis_edge_path', type=str, default='./FLIR_Vis_edge_map/', help='the folder to load Visible image edge map')
        self.parser.add_argument('--Vis_mask_path', type=str, default='./FLIR_Vis_seg_mask/', help='the folder to load Visible image segmentation mask')
        self.parser.add_argument('--IR_FG_txt', type=str, default='./FLIR_txt_file/IR_FG_list.txt', help='the txt file that records the filename of the NTIR image and the coordinates of the upper left corner of the crop.')
        self.parser.add_argument('--IR_patch_classratio_txt', type=str, default='./FLIR_txt_file/IR_patch_classratio.txt', help='the txt file recording the area proportions of each small sample category in the NTIR image.')
       
        self.parser.add_argument('--ssim_winsize', type=int, default=3, help='window size of SSIM loss')
        self.parser.add_argument('--num_class', type=int, default=19, help='number of segmentation class')
        self.parser.add_argument('--num_ACA_cluster', type=int, default=4, help='number of cluster for ACA Loss')
        self.parser.add_argument('--encoded_nc', type=int, default=128, help='channel number of encoded tensor')

        self.parser.add_argument('--niter', required=True, type=int, help='# of epochs at starting learning rate (try 50*n_domains)')
        self.parser.add_argument('--niter_decay', required=True, type=int, help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')
       
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='weight for identity "autoencode" mapping (A -> A)')
        self.parser.add_argument('--lambda_latent', type=float, default=0.0, help='weight for latent-space loss (A -> z -> B -> z)')
        self.parser.add_argument('--lambda_forward', type=float, default=0.0, help='weight for forward loss (A -> B; try 0.2)')
        self.parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight for SSIM loss')
        self.parser.add_argument('--lambda_tv', type=float, default=5.0, help='weight for TV loss')
        self.parser.add_argument('--lambda_sc', type=float, default=1.0, help='weight for semantic consistency loss')
        self.parser.add_argument('--vis_prob_th', type=float, default=0.95, help='probability threshold for updating visible image segmentation GT')
        self.parser.add_argument('--IR_prob_th', type=float, default=0.95, help='probability threshold for updating IR image segmentation GT')
        self.parser.add_argument('--grad_th_vis', type=float, default=0.8, help='threshold for SGA Loss about gradient of fake IR image')
        self.parser.add_argument('--grad_th_IR', type=float, default=0.8, help='threshold for SGA Loss about gradient of fake Visible image')
        
        self.parser.add_argument('--lambda_sga', type=float, default=0.5, help='weight for structured gradient alignment loss')
        self.parser.add_argument('--SGA_start_epoch', type=int, default=0, help='# of epochs at starting gradient orientation consistence loss')
        self.parser.add_argument('--SGA_fullload_epoch', type=int, default=0, help='# of epochs at starting gradient orientation consistence loss with full loaded weights')
        self.parser.add_argument('--SSIM_start_epoch', type=int, default=10, help='# of epochs at starting SSIM loss')
        self.parser.add_argument('--SSIM_fullload_epoch', type=int, default=10, help='# of epochs at starting SSIM loss with full loaded weights')
        self.parser.add_argument('--netS_start_epoch', type=int, default=0, help='# of epochs to start training the segmentation network')
        self.parser.add_argument('--netS_end_epoch', type=int, default=50, help='# of epochs to freeze the parameters of the segmentation network')
        self.parser.add_argument('--updateGT_start_epoch', type=int, default=30, help='# of epochs at starting updating uncertain region of sgementation GT')
        self.parser.add_argument("--often_balance", action='store_true', help="balance the apperance times.")
        self.parser.add_argument("--max_value", type=float, default=7, help="Max Value of Class Weight.")
        self.parser.add_argument("--only_hard_label", type=float, default=0, help="class balance.")
        self.parser.add_argument("--lambda_CGR",type=float, default=1.0, help="weight for the conditional gradient repair loss.")
        self.parser.add_argument("--sqrt_patch_num", type=int, default=8, help="sqrt patch number for structral gradient alignment loss.")

        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
