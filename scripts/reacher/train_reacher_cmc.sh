export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --domain_name reacher \
    --save_video \
    --task_name easy \
    --encoder_type pixel --work_dir . \
    --action_repeat 4 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 108 \
    --agent RAD_BYOL_SharedProj_CMC --frame_stack 3 --data_augs translate  \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --encoder_lr 1e-3 --eval_freq 5000 --batch_size 512 --num_train_steps 130000 \
    --num_layers 4 --extra node02-3 #--soda_tau 0 --extra2 _tau0