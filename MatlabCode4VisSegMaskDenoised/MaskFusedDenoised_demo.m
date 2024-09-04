clear all;
clc;

HMSANet_Mask_folder = 'Folder of segmentation masks predicted by HMSANet';
Detectron2_Mask_folder = 'Folder of segmentation masks predicted by Detectron2';
vis_folder = 'Folder for DC images';
des_path_original = 'Folder for saving the fused and denoised segmentation masks';
dir_file = dir(fullfile(Detectron2_Mask_folder, '*.png'));
file_names = {dir_file.name};

if ~exist(des_path_original, 'dir'), mkdir(des_path_original);end

for i = 1:length(file_names)
    original_filename = file_names{1, i};
    new_imgname = strrep(original_filename, '.png', '_prediction.png');
    HMSANet_Mask_file = [HMSANet_Mask_folder, new_imgname];%%%%%%%%%FLIR
    Detectron2_Mask_file = [Detectron2_Mask_folder, original_filename];

    HMSANet_Mask = double(imread(HMSANet_Mask_file));
    Detectron2_Mask = double(imread(Detectron2_Mask_file));
    mask_person_HMSANet = zeros(size(HMSANet_Mask));
    mask_person_Det2 = zeros(size(HMSANet_Mask));
    mask_car_HMSANet = zeros(size(HMSANet_Mask));
    mask_car_Det2 = zeros(size(HMSANet_Mask));
    mask_truck_Det2 = zeros(size(HMSANet_Mask));
    mask_build_HMSANet = zeros(size(HMSANet_Mask));
    mask_build_Det2 = zeros(size(HMSANet_Mask));
    
    mask_person_HMSANet(HMSANet_Mask == 11) = 1;
    mask_person_HMSANet(HMSANet_Mask == 12) = 1;
    mask_person_Det2(Detectron2_Mask == 11) = 1;
    person_region_int = mask_person_HMSANet .* mask_person_Det2;
    person_region_fuse = person_region_int * 11 + ((mask_person_HMSANet - person_region_int) * 255);
    
    mask_car_HMSANet(HMSANet_Mask == 13) = 1;
    mask_car_Det2(Detectron2_Mask == 13) = 1;
    mask_truck_Det2(Detectron2_Mask == 14) = 1;
    car_region_int = mask_car_HMSANet .* mask_car_Det2;
    %%%Due to differences in category definitions, HMSANet will define a truck as a car.
    car2truck = mask_car_HMSANet .* mask_truck_Det2;
    car_region_fuse = car_region_int * 13 + car2truck * 14 + ((mask_car_HMSANet - car_region_int - car2truck) * 255);
    
    mask_build_HMSANet(HMSANet_Mask == 2) = 1;
    mask_build_Det2(Detectron2_Mask == 2) = 1;
    build_region_int = mask_build_HMSANet .* mask_build_Det2;
    build_region_fuse = build_region_int * 2 + ((mask_build_HMSANet - build_region_int) * 255);

    new_mask = person_region_fuse + car_region_fuse + build_region_fuse + ((ones(size(HMSANet_Mask)) - mask_person_HMSANet - mask_car_HMSANet - mask_build_HMSANet) .* HMSANet_Mask);
    
    vis_file = [vis_folder, original_filename];
    vis_img = double(imread(vis_file));
    %%%%%Fuse only.
%     img_output = fullfile(des_path_original, original_filename);
%     imwrite(uint8(new_mask), img_output, 'png');
    
    %%%%%%%%%%%Denoised
    mask_refine = SegMaskDenoised(vis_img, new_mask);

    img_output = fullfile(des_path_original, original_filename);
    imwrite(uint8(mask_refine), img_output, 'png');
end
