function mask_out = SegMaskDenoised(input_img, ori_mask)

mask_vegetation = zeros(size(ori_mask));
mask_pole = zeros(size(ori_mask));
mask_sky = zeros(size(ori_mask));
    
mask_vegetation(ori_mask == 8) = 1;
mask_pole(ori_mask == 5) = 1;
mask_sky(ori_mask == 10) = 1;

cnt_vegetation = sum(sum(mask_vegetation));
cnt_pole = sum(sum(mask_pole));
cnt_sky = sum(sum(mask_sky));

region_vegetation = zeros(size(input_img));
region_pole = zeros(size(input_img));
region_sky = zeros(size(input_img));

[h, w] = size(ori_mask);

for i = 1:3
    region_vegetation(:, :, i) = mask_vegetation .* input_img(:, :, i);
    region_pole(:, :, i) = mask_pole .* input_img(:, :, i);
    region_sky(:, :, i) = mask_sky .* input_img(:, :, i);
end

if cnt_vegetation > 0
    vegetation_pixel_mean = (mean(mean(region_vegetation)) * h * w) / cnt_vegetation;
    vegetation_intraerr = region_vegetation - repmat(vegetation_pixel_mean, [h, w, 1]);
    vegetation_intradis = mask_vegetation .* sum(vegetation_intraerr .* vegetation_intraerr, 3);
end

if cnt_pole > 0
    pole_pixel_mean = (mean(mean(region_pole)) * h * w) / cnt_pole;
    pole_intraerr = region_pole - repmat(pole_pixel_mean, [h, w, 1]);
    pole_intradis = mask_pole .* sum(pole_intraerr .* pole_intraerr, 3);
end

if cnt_sky > 0
    sky_pixel_mean = (mean(mean(region_sky)) * h * w) / cnt_sky;
    sky_intraerr = region_sky - repmat(sky_pixel_mean, [h, w, 1]);
    sky_intradis = mask_sky .* sum(sky_intraerr .* sky_intraerr, 3);
end
%%Denoised for vegetation
if (cnt_vegetation * cnt_sky) > 0
    vegetation_sky_err = region_vegetation - repmat(sky_pixel_mean, [h, w, 1]);
    vegetation_sky_dis = mask_vegetation .* sum(vegetation_sky_err .* vegetation_sky_err, 3);
    veg2sky_dis_err = vegetation_intradis - vegetation_sky_dis;
    veg2sky_mask = zeros(size(ori_mask));
    veg2sky_mask(veg2sky_dis_err > 0) = 1;
    mask_vegetation_refine = veg2sky_mask * 255 + (mask_vegetation - veg2sky_mask) * 8;
    
    new_veg_mask = mask_vegetation - veg2sky_mask;
    new_veg_region = repmat(new_veg_mask, [1,1,3]) .* input_img;
    cnt_veg_new = sum(sum(new_veg_mask));
    if cnt_veg_new > 0 
        veg_pixel_mean_new = (mean(mean(new_veg_region)) * h * w) / cnt_veg_new;
    else
        veg_pixel_mean_new = vegetation_pixel_mean;
    end
        
else
    mask_vegetation_refine = mask_vegetation * 8;
end
%%Denoised for pole    
if (cnt_pole * cnt_sky) > 0
    pole_sky_err = region_pole - repmat(sky_pixel_mean, [h, w, 1]);
    pole_sky_dis = mask_pole .* sum(pole_sky_err .* pole_sky_err, 3);
    pole2sky_dis_err = pole_intradis - pole_sky_dis;
    pole2sky_mask = zeros(size(ori_mask));
    pole2sky_mask(pole2sky_dis_err > 0) = 1;
    mask_pole_refine = pole2sky_mask * 255 + (mask_pole - pole2sky_mask) * 5;
    
    new_pole_mask = mask_pole - pole2sky_mask;
    new_pole_region = repmat(new_pole_mask, [1,1,3]) .* input_img;
    cnt_pole_new = sum(sum(new_pole_mask));
    if cnt_pole_new > 0 
        pole_pixel_mean_new = (mean(mean(new_pole_region)) * h * w) / cnt_pole_new;
    else
        pole_pixel_mean_new = pole_pixel_mean;
    end
else
    mask_pole_refine = mask_pole * 5;
end
%%Denoised for sky
if (cnt_vegetation * cnt_pole * cnt_sky) > 0
    sky_pole_err = region_sky - repmat(pole_pixel_mean_new, [h, w, 1]);
    sky_pole_dis = mask_sky .* sum(sky_pole_err .* sky_pole_err, 3);
    sky2pole_dis_err = sky_intradis - sky_pole_dis;
    sky2pole_mask = zeros(size(ori_mask));
    sky2pole_mask(sky2pole_dis_err > 0) = 1;
    
    sky_vegetation_err = region_sky - repmat(veg_pixel_mean_new, [h, w, 1]);
    sky_vegetation_dis = mask_sky .* sum(sky_vegetation_err .* sky_vegetation_err, 3);
    sky2veg_dis_err = sky_intradis - sky_vegetation_dis;
    sky2veg_mask = zeros(size(ori_mask));
    sky2veg_mask(sky2veg_dis_err > 0) = 1;
    
    sky_uncertain_mask = zeros(size(ori_mask));
    fuse_unc = sky2pole_mask + sky2veg_mask;
    sky_uncertain_mask(fuse_unc > 0) = 1;
    
    mask_sky_refine = sky_uncertain_mask * 255 + (mask_sky - sky_uncertain_mask) * 10;
else
    if (cnt_pole * cnt_sky) > 0
        sky_pole_err = region_sky - repmat(pole_pixel_mean_new, [h, w, 1]);
        sky_pole_dis = mask_sky .* sum(sky_pole_err .* sky_pole_err, 3);
        sky2pole_dis_err = sky_intradis - sky_pole_dis;
        sky2pole_mask = zeros(size(ori_mask));
        sky2pole_mask(sky2pole_dis_err > 0) = 1;
        mask_sky_refine = sky2pole_mask * 255 + (mask_sky - sky2pole_mask) * 10;
    elseif (cnt_vegetation * cnt_sky) > 0
        sky_vegetation_err = region_sky - repmat(veg_pixel_mean_new, [h, w, 1]);
        sky_vegetation_dis = mask_sky .* sum(sky_vegetation_err .* sky_vegetation_err, 3);
        sky2veg_dis_err = sky_intradis - sky_vegetation_dis;
        sky2veg_mask = zeros(size(ori_mask));
        sky2veg_mask(sky2veg_dis_err > 0) = 1;
        mask_sky_refine = sky2veg_mask * 255 + (mask_sky - sky2veg_mask) * 10;
    else
        
        mask_sky_refine = mask_sky * 10;
    end
end

mask_out = mask_vegetation_refine + mask_pole_refine + mask_sky_refine + (ones(size(ori_mask)) - mask_vegetation - mask_pole - mask_sky) .* ori_mask;

