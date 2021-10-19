rm -rf server

for NUM_STEPS in 5 10
do 
    python3 train-pure.py \
        --arch resnet34 \
        --device cuda:0 \
        --batch_size 256 \
        --max_epoch 50 \
        --lr 1e-4 \
        --num_worker 8 \
        --seed 100 \
        --model_save_dir server/checkpoints \
        --model_save_name resnet34 \
        --data ~/datasets/gc10_none_mask_divided \
        --logdir server/runs
done

for NUM_STEPS in 5 10 20 40
do 
    for EPSILON in 1 2 4 8 16 32
    do
        for ALPHA in 0.01 0.02 0.03 0.04 0.05 0.06
        do
        python3 train-adv.py \
            --arch resnet34 \
            --device cuda:0 \
            --batch_size 256 \
            --max_epoch 50 \
            --lr 1e-4 \
            --num_worker 8 \
            --seed 100 \
            --model_save_dir server/checkpoints \
            --model_save_name resnet34-adv-${EPSILON}_${ALPHA}_${NUM_STEPS} \
            --data ~/datasets/gc10_none_mask_divided \
            --logdir server/runs \
            --epsilon ${EPSILON} \
            --alpha ${ALPHA} \
            --iters ${NUM_STEPS}
        done
    done
done



# python3 train-adv.py \
#     --arch resnet34 \
#     --device cuda:0 \
#     --batch_size 256 \
#     --max_epoch 50 \
#     --lr 1e-4 \
#     --num_worker 8 \
#     --seed 100 \
#     --model_save_dir server/checkpoints \
#     --model_save_name resnet34-adv-lr1e-4_003_10 \
#     --data ~/datasets/gc10_none_mask_divided \
#     --logdir server/runs \
#     --epsilon 4 \
#     --alpha 0.03 \
#     --iters 10