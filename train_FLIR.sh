
###############MornGAN_FLIR
python train.py --name MornGAN_FLIR --dataroot ./FLIR_datasets/ --n_domains 2 --niter 80 --niter_decay 0 --loadSize 288 \
--fineSize 256 --resize_or_crop crop --IR_edge_path ./FLIR_IR_edge_map/ --Vis_edge_path ./FLIR_Vis_edge_map/ --Vis_mask_path \
./FLIR_Vis_seg_mask/ --IR_FG_txt ./FLIR_txt_file/IR_FG_list.txt --IR_patch_classratio_txt ./FLIR_txt_file/IR_patch_classratio.txt \
--net_Gen_type gen_v1 --netS_start_epoch 20 --updateGT_start_epoch 30 --netS_end_epoch 75 --IR_prob_th 0.95 --lambda_sc 1.0 \
--lambda_sga 0.5 --gpu_ids 1



