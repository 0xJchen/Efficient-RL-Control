export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=1

for view in 1
do
    for step in 1
    do
        for seed in 0 1 2 3 4
        do
            for weight in 1
            do
                python train.py \
                    --domain_name cartpole \
                    --save_video \
                    --task_name swingup \
                    --encoder_type pixel --work_dir . \
                    --action_repeat 8 --num_eval_episodes 10 \
                    --pre_transform_image_size 84 --image_size 92 \
                    --agent RAD_BYOL_MV_TEST --frame_stack 3 --data_augs translate  \
                    --seed $seed --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 2500 --batch_size 512 --num_train_steps 13000 \
                    --num_layers 4 --extra node14-1 --pred_step $step --weight $weight --view $view
            done 
        done
    done 
done

