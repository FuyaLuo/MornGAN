# Memory-Guided Collaborative Attention for Nighttime Thermal Infrared Image Colorization of Traffic Scenes
Pytorch implementation of the paper "Memory-Guided Collaborative Attention for Nighttime Thermal Infrared Image Colorization of Traffic Scenes".

![tease](https://github.com/FuyaLuo/MornGAN/blob/main/docs/Model.PNG)

### [Paper](https://ieeexplore.ieee.org/document/10648659)

## Qualitative Comparison of Video Colorization
### Example 1
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CycleGAN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PearlGAN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MornGAN
<p float="left">
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/CycleGAN_video1.gif" width="256" />
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/PearlGAN_video1.gif" width="256" /> 
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/MornGAN_video1.gif" width="256" />
</p>

### Example 2
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CycleGAN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PearlGAN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MornGAN
<p float="left">
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/CycleGAN_video2.gif" width="256" />
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/PearlGAN_video2.gif" width="256" /> 
  <img src="https://github.com/FuyaLuo/MornGAN/blob/main/Qualitative%20comparison%20of%20video%20translation/MornGAN_video2.gif" width="256" />
</p>

## Abstract
>Robust imaging under challenging conditions, such as starlit nights, has broadened the adoption of thermal infrared (TIR) cameras for nighttime driving scenes. Given that TIR images are monochromatic, which makes them difficult to interpret by humans and limits the applicability of RGB-based algorithms, it is reasonable to perform colorization of nighttime TIR (NTIR) images by converting them into corresponding daytime color images (NTIR2DC). Despite the impressive results achieved by previous NTIR2DC methods, how to improve the colorization performance of small-sample categories without semantic annotation is under-explored. To address this issue, we propose a novel learning framework called Memory-guided cOllaboRative atteNtion Generative Adversarial Network (MornGAN), which is inspired by the analogical reasoning mechanisms of humans. Specifically, we first propose an online semantic distillation module to mine and refine the semantic cues of NTIR images. Then, a memory-guided sample selection strategy and adaptive collaborative attention loss are devised to enhance the semantic preservation of small-sample categories. Further, a new conditional gradient repair loss is introduced for reducing edge distortion during translation. Extensive experiments on the NTIR2DC task show that the proposed MornGAN significantly outperforms other image-to-image translation methods in terms of semantic preservation and edge consistency, which helps improve the object detection accuracy remarkably. 

## Prerequisites
* Python 3.6 
* Pytorch 1.1.0 and torchvision 0.3.0 
* TensorboardX
* visdom
* dominate
* pytorch-msssim
* kmeans_pytorch
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.

## Data Preparation 
Download [FLIR](https://www.flir.co.uk/oem/adas/adas-dataset-form/) and [KAIST](https://soonminhwang.github.io/rgbt-ped-detection/data/). First, the corresponding training set and test set images are sampled according to the txt files in the `./img_list/` folder. Then, all images are first resized to 500x400, and then crop centrally to obtain images with a resolution of 360x288. Note that due to negligent checking by the authors, the test set images for the KAIST dataset only need to be center cropped to 360x288 without a resize step. Finally, place all images into the corresponding dataset folders. Domain A and domain B correspond to the daytime visible image and the nighttime TIR image, respectively. As an example, the corresponding folder structure for the FLIR dataset is:
 ```
mkdir FLIR_datasets
# The directory structure should be this:
FLIR_datasets
   ├── trainA (daytime RGB images)
       ├── FLIR_00002.png 
       └── ...
   ├── trainB (nighttime IR images)
       ├── FLIR_00135.png
       └── ...
   ├── testA (testing daytime RGB images)
       ├── FLIR_09112.png (The test image that you want)
       └── ... 
   ├── testB (testing nighttime IR images)
       ├── FLIR_08872.png (The test image that you want)
       └── ... 

mkdir FLIR_testsets
# The directory structure should be this:
FLIR_testsets
   ├── test0 (empty folder)
   ├── test1 (testing nighttime IR images)
       ├── FLIR_08863.png
       └── ...
```

We predict the edge maps of Nighttime TIR images and daytime color images using [MCI](https://drive.google.com/file/d/1Qf2wIyzr0J8nWSuc8d6bHyO2Mxzeuamv/view?usp=sharing) method and Canny edge detection method, respectively. Next, place all edge maps into the corresponding folders(e.g., `/FLIR_IR_edge_map/` and `/FLIR_Vis_edge_map/` for FLIR dataset).

For segmentation mask prediction of DC images, we first utilize [HMSANet](https://github.com/segcv/hierarchical-multi-scale-attention) and [Detectron2](https://github.com/facebookresearch/detectron2) models to obtain the initial mask. Then, the masks obtained from the predictions of the two models are fused and semantic denoising is performed, which can be realized by running the code `MaskFusedDenoised_demo.m`. You will need to modify the four paths in the code to suit your situation. The final masks for the DC images used for training of the FLIR and KAIST datasets can be downloaded via [google drive](https://drive.google.com/file/d/1yQulkm3IHMXCOtkBRS-T05D2WoOVjx0G/view?usp=sharing). Next, place all segmentation masks into the corresponding folders(i.e., `/FLIR_Vis_seg_mask/` and `/KAIST_Vis_seg_mask/` for the FLIR and KAIST dataset, respectively).

## Inference Using Pretrained Model

<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/1hAVuCjn1wtVv73FF4ASRtxQBLHBnMIE5/view?usp=sharing) and save it in `./checkpoints/MornGAN_FLIR/`. Place the test images of the FLIR dataset in `./FLIR_testsets/test1/`. Then run the command 
```bash
python test_output_only.py --phase test --serial_test --name MornGAN_FLIR --dataroot ./FLIR_testsets/ --n_domains 2 --which_epoch 80 --results_dir ./res_FLIR/ --loadSize 288 --net_Gen_type gen_v1 --no_flip --gpu_ids 0
```
</details>

<details>
  <summary>
    <b>2) KAIST</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/1uqU0lBRH_O9TrXfT6EQlxwUlCx380am_/view?usp=sharing) and save it in `./checkpoints/MornGAN_KAIST/`. Place the test images of the FLIR dataset in `./KAIST_testsets/test1/`. Then run the command 
```bash
python test_output_only.py --phase test --serial_test --name MornGAN_KAIST --dataroot ./KAIST_testsets/ --n_domains 2 --which_epoch 160 --results_dir ./res_KAIST/ --loadSize 288 --net_Gen_type gen_v1 --no_flip --gpu_ids 0
```
</details>

## Training

To reproduce the performance, we recommend that users try multiple training sessions.
<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./FLIR_datasets/`. Then run the command
  ```bash
  bash ./train_FLIR.sh
  ```
</details>


<details>
  <summary>
    <b>2) KAIST</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./KAIST_datasets/`. Then run the command
   ```bash
   bash ./train_KAIST.sh
   ```

</details>

## Evaluation
<details>
  <summary>
    <b>1) Semantic segmenation</b>
  </summary>
  
   Download the code for the semantic segmentation model [HMSANet](https://github.com/segcv/hierarchical-multi-scale-attention) and then follow the instructions to install it. Next, download the pre-trained [model](https://drive.google.com/open?id=1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) on the Cityscape dataset, and then change line 52 in the `config.py` to the path of the folder where these pre-training weights are located. After that, download the segmentation mask and code for both datasets via [google drive](https://drive.google.com/file/d/1e0HZR5TyAMK9GXzG1O9ETkuYsoOzT3md/view?usp=sharing). Put `misc.py` in folder `./utils/` and replace the original file, all other files are placed inside the directory `/semantic-segmentation-main/`. For the evaluation on FLIR dataset, run the command
   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 eval_FLIR.py --dataset cityscapes --syncbn --apex --fp16 --eval_folder /Your_FLIR_Results_Path --snapshot /Your_Pretrained_Models_Path/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth --dump_assets --dump_all_images --result_dir ./Your_FLIR_Mask_SavePath
   ```
   And for the evaluation on KAIST dataset, run the command
   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 eval_KAIST.py --dataset cityscapes --syncbn --apex --fp16 --eval_folder /Your_KAIST_Results_Path --snapshot /Your_Pretrained_Models_Path/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth --dump_assets --dump_all_images --result_dir ./Your_KAIST_Mask_SavePath
   ```
   
</details>

<details>
  <summary>
    <b>2) Object detection</b>
  </summary>
  
  Download the code for [YOLOv7](https://github.com/WongKinYiu/PyTorch_YOLOv4), then follow the instructions to install it. Next, download the YOLOv7 detection txt file we transformed from the FLIR and KAIST datasets via [google drive](https://drive.google.com/file/d/1-GSjW-p1CPsF43SDzHSuRRDFeGkIn4ZF/view?usp=sharing). Once the unzip is complete, place all files in the `/yolov7-main/` folder. Note that the files `FLIR.yaml`, `FLIR_imglist.txt`, `KAIST.yaml` and `KAIST_imglist.txt` should be placed in the directory `/yolov7-main/data/`. Then, the translation results of FLIR and KAIST should be placed inside the `/yolov7-main/FLIR_datasets/images/` and `/yolov7-main/KAIST_datasets/images/` directories respectively. For the evaluation on FLIR dataset, run the command
  ```bash
  python test.py --data data/FLIR.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights pretrain_weights/yolov7.pt --name FLIR_640_val --verbose
  ```
  And for the evaluation on KAIST dataset, run the command
   ```bash
   python test.py --data data/KAIST.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 2 --weights pretrain_weights/yolov7.pt --name KAIST_640_val --verbose
   ```
       
</details>


<details>
  <summary>
    <b>3) Edge consistency</b>
  </summary>
  
   Please refer to the [PearlGAN](https://github.com/FuyaLuo/PearlGAN) repository.

    
</details>

## Downloading files using Baidu Cloud Drive
If the above Google Drive link is not available, you can try to download the relevant code and files through the [Baidu cloud link](https://pan.baidu.com/s/1ojaqDf6dV_XYrsOqi1NNAg), extraction code: ir2d.

## Citation
If you like our work and use the code or models for your research, please cite our work as follows.
```
@article{luo2024memory,
  title={Memory-guided collaborative attention for nighttime thermal infrared image colorization of traffic scenes},
  author={Luo, Fu-Ya and Cao, Yi-Jun and Yang, Kai-Fu and Wang, Gang and Li, Yong-Jie},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```

## License

The codes and the pretrained model in this repository are under the BSD 2-Clause "Simplified" license as specified by the LICENSE file. 

## Acknowledgments
This code is heavily borrowed from [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN).  
Spectral Normalization code is borrowed from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py).  
