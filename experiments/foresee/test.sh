python val_kitti.py \
--dataroot ../../datasets/KITTI_object \
--dataset kitti_object_roi \
--load_ckpt epoch19_step18000.pth \
--encoder ResNeXt101_32x4d_body_stride16 \
--decoder_out_c 100 
