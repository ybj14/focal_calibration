########## RESNET50 ##################
MODEL_DIRECTORY=/home/binjie/projects/focal_calibration/train_scripts/results

##L2
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss mse_loss \
--save-path $MODEL_DIRECTORY/ &

##SE
CUDA_VISIBLE_DEVICES=1 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss se_loss \
--save-path $MODEL_DIRECTORY/ &

##CE
CUDA_VISIBLE_DEVICES=2 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss cross_entropy \
--save-path $MODEL_DIRECTORY/ &

##Brier Loss
CUDA_VISIBLE_DEVICES=3 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss brier_score \
--save-path $MODEL_DIRECTORY/ &

##MMCE
CUDA_VISIBLE_DEVICES=4 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss mmce_weighted --lamda 2.0 \
--save-path $MODEL_DIRECTORY/ &

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=5 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 1.0 \
--save-path $MODEL_DIRECTORY/ &

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=6 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 2.0 \
--save-path $MODEL_DIRECTORY/ &

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=7 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 3.0 \
--save-path $MODEL_DIRECTORY/ &

