MODEL_DIRECTORY=/home/binjie/projects/focal_calibration/train_scripts/results
########## RESNET50 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_brier_score_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_1.0_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_2.0_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_3.0_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_mmce_weighted_lamda_2.0_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_mse_loss_350.model \
>> ./my_results.txt
CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path $MODEL_DIRECTORY/ \
--saved_model_name resnet50_se_loss_350.model \
>> ./my_results.txt
