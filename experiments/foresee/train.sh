CUDA_VISIBLE_DEVICES=0  python  train_kitti.py \
--dataroot ../../datasets/KITTI_object \
--dataset kitti_object_roi \
--encoder ResNeXt101_32x4d_body_stride16 \
--decoder_out_c 100 \
--lr 0.001 \
--batchsize 4 \
--epoch 0 20 \
--use_tfboard 

