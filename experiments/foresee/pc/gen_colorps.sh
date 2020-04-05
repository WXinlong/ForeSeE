python kitti_prediction_colorps.py \
--dataroot ../../../datasets/KITTI_object \
--dataset kitti_prediction \
--load_ckpt ../epoch19_step18000.pth \
--pcd_dir pseudo-lidar/ForSeE.colorps/training \
--encoder ResNeXt101_32x4d_body_stride16 \
--decoder_out_c 100 \
--phase  val

