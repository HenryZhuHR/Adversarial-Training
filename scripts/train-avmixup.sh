python3 train.py \
    --arch resnet34 \
    --device cuda:0 \
    --batch_size 256 \
    --max_epoch 500 \
    --lr 0.001 \
    --num_worker 8 \
    --model_save_dir server/checkpoints \
    --model_save_name resnet34-avMixup \
    --data ~/datasets/custom \
    --logdir server/runs \
    --adversarial_training \
    --attack_method pgd \
    --attack_args epsilon=4 alpha=8 num_steps=10


python3 train.py \
    --arch resnet34 \
    --device cuda:1 \
    --batch_size 128 \
    --max_epoch 500 \
    --lr 0.001 \
    --num_worker 8 \
    --model_save_dir server/checkpoints \
    --model_save_name resnet34-avMixup-8_8_40 \
    --data ~/datasets/custom \
    --logdir server/runs \
    --adversarial_training \
    --attack_method pgd \
    --attack_args epsilon=8 alpha=8 num_steps=40
