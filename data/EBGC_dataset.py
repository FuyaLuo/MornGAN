import os.path, glob
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, night_train_transformv3
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import json

class EBGCDataset(BaseDataset):
    def __init__(self, opt):
        super(EBGCDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        self.night_edge_transform = night_train_transformv3(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))
        if self.opt.isTrain:
            self.IR_edge_paths = opt.IR_edge_path  #######Edit by lfy########
            self.Vis_edge_paths = opt.Vis_edge_path
            self.Vis_mask_paths = opt.Vis_mask_path
            self.IR_FG_txt = opt.IR_FG_txt
            self.num_class = opt.num_class
            self.IR_memory_txt = opt.IR_patch_classratio_txt

        self.paths = [sorted(make_dataset(d)) for d in self.dirs] 
        self.sizes = [len(p) for p in self.paths] 

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def load_image_train(self, dom, idx):
        path = self.paths[dom][idx]
        img = np.array(Image.open(path).convert('RGB'))
        path_split = path.split('/')
        img_name = path_split[-1]
        if dom == 0:
            edge_map_file = self.Vis_edge_paths + img_name
            seg_mask_file = self.Vis_mask_paths + img_name
            seg_mask = np.array(Image.open(seg_mask_file))
        else:
            edge_map_file = self.IR_edge_paths + img_name
            seg_mask = np.zeros_like(np.array(Image.open(edge_map_file)))
        edge_map = np.array(Image.open(edge_map_file).convert('L'))

        for func in self.night_edge_transform:
            img, edge_map, seg_mask = func(img, edge_map, seg_mask)
        # img = img.transpose(2, 0, 1) / 255.0
        transform_list_torch = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_new = transforms.Compose(transform_list_torch)
        # img = torch.from_numpy(img_file.transpose((2, 0, 1)))
        img_new = Image.fromarray(np.uint8(img))
        img_res = transform_new(img_new)
        edge_map = edge_map / 255.0
        seg_mask = np.asarray(Image.fromarray(seg_mask), dtype=np.int64)
        # print(seg_mask)

        return img_res, path, torch.tensor(edge_map), seg_mask.copy()

    def load_image_train_crop(self, dom, idx, crop_pos_h, crop_pos_w):
        path = self.paths[dom][idx]
        img = np.array(Image.open(path).convert('RGB'))
        path_split = path.split('/')
        img_name = path_split[-1]
        if dom == 0:
            edge_map_file = self.Vis_edge_paths + img_name
            seg_mask_file = self.Vis_mask_paths + img_name
            seg_mask = np.array(Image.open(seg_mask_file))
        else:
            edge_map_file = self.IR_edge_paths + img_name
            seg_mask = np.zeros_like(np.array(Image.open(edge_map_file)))
        edge_map = np.array(Image.open(edge_map_file).convert('L'))

        w1 = crop_pos_h
        h1 = crop_pos_w

        img_crop = img[w1 : (w1 + 256), h1 : (h1 + 256)]
        edge_map_crop = edge_map[w1 : (w1 + 256), h1 : (h1 + 256)]
        seg_mask_crop = seg_mask[w1 : (w1 + 256), h1 : (h1 + 256)]
        # print(img_crop.shape)

        if np.random.rand() < 0.5:
            img_crop = img_crop[:,::-1]
            edge_map_crop = edge_map_crop[:,::-1]
            seg_mask_crop = seg_mask_crop[:,::-1]

        transform_list_torch = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_new = transforms.Compose(transform_list_torch)
        # img = torch.from_numpy(img_file.transpose((2, 0, 1)))
        img_new = Image.fromarray(np.uint8(img_crop))
        img_res = transform_new(img_new)
        # print(img_res.size)
        edge_map_crop = edge_map_crop / 255.0
        seg_mask_crop = np.asarray(Image.fromarray(seg_mask_crop), dtype=np.int64)

        return img_res, path, torch.tensor(edge_map_crop), seg_mask_crop.copy()

    def count_class_ratio(self, input_mask):
        count = np.zeros((1, self.num_class))
        h, w = input_mask.shape
        for i in range(self.num_class):
            count[0, i] = (np.sum(input_mask == i)) / (h * w)
        ###Small sample categories includes six categories: traffic light, traffic sign, person, truck, bus, and motorcycle. 
        ###Their indexes are 6, 7, 11, 14, 15 and 17, respectively.
        SSC_idx = np.zeros((self.num_class, 1))
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

        return cls_ratio_norm, SSC_ratio_sum

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)

            A_img, A_path = self.load_image(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path}
        else:
            # Choose two of our domains to perform a pass on
            # DA, DB = random.sample(range(len(self.dirs)), 2) #########DA是0,1中的任意一个
            DA, DB = 0, 1
            index_A = random.randint(0, self.sizes[DA] - 1)
            A_img, A_path, edge_map_A, seg_mask_A = self.load_image_train(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path, 'EMA':edge_map_A, 'SMA':seg_mask_A}
                
            if os.path.getsize(self.IR_FG_txt) != 0:
                A_patch_cls_ratio, A_SSC_ratio_sum = self.count_class_ratio(seg_mask_A)
                # IRFGList = open(self.IR_FG_txt, 'r')
                with open(self.IR_FG_txt, 'r') as IRFGList:
                    lines = IRFGList.readlines()
                ####In order to memorize the class proportions of all NTIR images, different sampling schemes 
                # are selected based on the number of samples memorized. When the number of samples is less than half, 
                # random indexing is used; when the number of samples is greater than half, sampling is performed from 
                # the unsampled samples; when the memorization of all samples is completed, the memory-guided sample 
                # selection strategy will be enabled.
                if len(lines) < (0.5 * self.sizes[DB]):
                    index_B = random.randint(0, self.sizes[DB] - 1)
                    IR_pos_h = np.random.randint(0, 32)
                    IR_pos_w = np.random.randint(0, 104)
                elif len(lines) < self.sizes[DB]:
                    all_idx_list = [num for num in range(self.sizes[DB])]
                    mem_IR_idx_list = []
                    for idx_img in range(len(lines) - 1):
                        str_list_idx = lines[idx_img].strip('\n')
                        line_list_idx = str_list_idx.split(' ')
                        mem_IR_idx_list.append(int(line_list_idx[0])) 

                    diff_list = list(set(all_idx_list).difference(set(mem_IR_idx_list)))
                    rand_idx = random.randint(0, len(diff_list) - 1)
                    index_B = diff_list[rand_idx]
                    IR_pos_h = np.random.randint(0, 32)
                    IR_pos_w = np.random.randint(0, 104)

                else:
                    if A_SSC_ratio_sum > 0:

                        with open(self.IR_memory_txt,'r') as txt_f:
                            mask_new_list = txt_f.readlines()
                        # mask_ratio_array = np.zeros((len(mask_new_list), self.num_class))
                        mask_ratio_array = np.zeros((len(mask_new_list), 6))
                        # list111 = []
                        for idx_mask in range(len(mask_new_list)):
                            float_list = []
                            str_list = mask_new_list[idx_mask].strip('\n')
                            line_list = str_list.split(', ')
                            float_list = [float(x) for x in line_list]
                            mask_ratio_array[idx_mask, :] = np.array(float_list, dtype=float)

                        cos_similarity_mul = np.dot(mask_ratio_array, A_patch_cls_ratio.reshape(-1, 1))
                        
                        #####top k neiborghhood
                        # print(cos_similarity_mul.shape)
                        sim_arr_row = cos_similarity_mul.reshape(1, -1)
                        KNN_idx_list = np.argpartition(sim_arr_row, -10)
                        top_k_list = KNN_idx_list[-10:]
                        k_idx = np.random.randint(0, 10, 1)
                        index_txt = top_k_list[0, k_idx][0]
                        # #print(index_txt[0])

                        # index_txt = np.argmax(cos_similarity_mul)
                        lines_IR_FG = lines[index_txt].strip('\n')
                        lines_IR_FG_split = lines_IR_FG.split(' ')
                        # print(np.argmax(cos_similarity_mul), np.max(cos_similarity_mul))
                        index_B = int(lines_IR_FG_split[0])
                        IR_pos_h = int(lines_IR_FG_split[1])
                        IR_pos_w = int(lines_IR_FG_split[2])
                    else:
                        index_B = random.randint(0, self.sizes[DB] - 1)
                        IR_pos_h = np.random.randint(0, 32)
                        IR_pos_w = np.random.randint(0, 104)
            else:
                index_B = random.randint(0, self.sizes[DB] - 1)
                IR_pos_h = np.random.randint(0, 32)
                IR_pos_w = np.random.randint(0, 104)

            B_img, _, edge_map_B, seg_mask_B = self.load_image_train_crop(DB, index_B, IR_pos_h, IR_pos_w)
            bundle.update( {'B': B_img, 'DB': DB, 'EMB':edge_map_B, 'SMB':seg_mask_B, 'IdxB':index_B, 'PosH':IR_pos_h, 'PosW':IR_pos_w} )
        

        return bundle


    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'EBGCDataset'
