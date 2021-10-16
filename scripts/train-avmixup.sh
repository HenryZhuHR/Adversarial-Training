python3 train.py \
    --arch resnet34 \
    --device cuda:0 \
    --batch_size 128 \
    --max_epoch 30 \
    --lr 1e-2 \
    --num_worker 8 \
    --seed 10486 \
    --model_save_dir server/checkpoints \
    --model_save_name resnet34-lr1e_2 \
    --data ~/datasets/gc10_none_mask_divided \
    --logdir server/runs

python3 train.py \
    --arch resnet34 \
    --device cuda:0 \
    --batch_size 256 \
    --max_epoch 30 \
    --lr 1e-4 \
    --num_worker 8 \
    --seed 10000 \
    --model_save_dir server/checkpoints \
    --model_save_name resnet34-avMixup \
    --data ~/datasets/custom \
    --logdir server/runs \
    --adversarial_training \
    --attack_method pgd \
    --attack_args epsilon=2/255 alpha=0.01 num_steps=20