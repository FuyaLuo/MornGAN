
###############MornGAN_KAIST
python train.py --name MornGAN_KAIST --dataroot ./KAIST_datasets/ --n_domains 2 --niter 160 --niter_decay 0 --loadSize 288 \
--fineSize 256 --resize_or_crop crop --IR_edge_path ./KAIST_IR_edge_map/ --Vis_edge_path ./KAIST_Vis_edge_map/ --Vis_mask_path \
./KAIST_Vis_seg_mask/ --IR_FG_txt ./KAIST_txt_file/IR_FG_list.txt --IR_patch_classratio_txt ./KAIST_txt_file/IR_patch_classratio.txt \
--net_Gen_type gen_v1 --netS_start_epoch 50 --updateGT_start_epoch 90 --netS_end_epoch 145 --grad_th_vis 0.44 --grad_th_IR 0.44 \
--IR_prob_th 0.95 --lambda_sc 1.0 --lambda_sga 0.5 --gpu_ids 1



