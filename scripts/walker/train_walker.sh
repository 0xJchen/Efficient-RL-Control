export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=2
python train.py \
    --domain_name walker \
    --save_model \
    --task_name walk \
    --encoder_type pixel --work_dir . \
    --action_repeat 2 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent RAD_BYOL_SharedProj --frame_stack 3 --data_augs crop \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --encoder_lr 1e-3 --eval_freq 10000 --batch_size 512 --num_train_steps 260000 \
    --num_layers 4 --extra node22-2