export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=3

for view in 2
do
    for step in 1
    do
        for seed in 0 1 2 3 4
        do
            for weight in 1
            do
                python train.py \
                    --domain_name cheetah \
                    --save_video \
                    --task_name run \
                    --encoder_type pixel --work_dir . \
                    --action_repeat 4 --num_eval_episodes 10 \
                    --pre_transform_image_size 100 --image_size 84 \
                    --agent RAD_BYOL_MV --frame_stack 3 --data_augs crop  \
                    --seed $seed --critic_lr 2e-4 --actor_lr 2e-4 --encoder_lr 2e-4 --eval_freq 5000 --batch_size 512 --num_train_steps 26000 \
                    --num_layers 4 --extra node11-3 --pred_step $step --weight $weight --view $view
            done 
        done
    done 
done