export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=3

for step in 1 3
do
    for seed in 0 1 2 3 4
    do
        for weight in 0
        do
            python train.py \
                --domain_name ball_in_cup \
                --save_video \
                --task_name catch \
                --encoder_type pixel --work_dir . \
                --action_repeat 4 --num_eval_episodes 10 \
                --pre_transform_image_size 84 --image_size 92 \
                --agent RAD_BYOL --frame_stack 3 --data_augs translate  \
                --seed $seed --critic_lr 1e-3 --actor_lr 1e-3 --encoder_lr 1e-3 --eval_freq 5000 --batch_size 512 --num_train_steps 26000 \
                --num_layers 4 --extra node15-3 --pred_step $step --weight $weight
        done 
    done
done 