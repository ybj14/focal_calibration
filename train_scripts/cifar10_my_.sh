########## RESNET50 ##################
MODEL_DIRECTORY=/home/binjie/projects/focal_calibration/train_scripts/results

export CUDNN_HOME=/opt/cudnn-v7.6
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=$CPATH:${CUDNN_HOME}/include
export CPLUS_INCLUDE_PATH=${CUDNN_HOME}/include:$CPLUS_INCLUDE_PATH

##L2
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss mse_loss_s \
--save-path $MODEL_DIRECTORY/ &

##SE
CUDA_VISIBLE_DEVICES=1 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss se_loss_s \
--save-path $MODEL_DIRECTORY/ &

##CE
CUDA_VISIBLE_DEVICES=2 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss mse_loss_n \
--save-path $MODEL_DIRECTORY/ &

##Brier Loss
CUDA_VISIBLE_DEVICES=3 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss se_loss_n \
--save-path $MODEL_DIRECTORY/ &