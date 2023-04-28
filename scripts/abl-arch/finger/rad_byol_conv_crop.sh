export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=3

for step in 1
do
    for seed in 0 1 2 3 4
    do
        for weight in 1
        do
            python train.py \
                --domain_name finger \
                --save_video \
                --task_name spin \
                --encoder_type pixel --work_dir . \
                --action_repeat 2 --num_eval_episodes 10 \
                --pre_transform_image_size 100 --image_size 84 \
                --agent RAD_BYOL_CONV --frame_stack 3 --data_augs crop  \
                --seed $seed --critic_lr 1e-3 --actor_lr 1e-3 --encoder_lr 1e-3 --eval_freq 10000 --batch_size 512 --num_train_steps 51000 \
                --num_layers 4 --extra node17-3 --pred_step $step --weight $weight
        done 
    done
done 