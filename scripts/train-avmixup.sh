rm -rf server

BATCH_SIZE=128
EPOCHS=300

for NUM_STEPS in 1
do 
    python3 train-pure.py \
        --arch resnet34 \
        --device cuda:1 \
        --batch_size ${BATCH_SIZE} \
        --max_epoch ${EPOCHS} \
        --lr 1e-4 \
        --num_worker 8 \
        --seed 100 \
        --model_save_dir server/checkpoints \
        --model_save_name resnet34 \
        --data ~/datasets/gc10_none_mask_divided \
        --logdir server/runs
done

# for ALPHA in 0.03 0.05 0.01
# do 
# 	for NUM_STEPS in 5 10 20 40
# 	do 
# 		for EPSILON in 2 4 8 16
# 		do 
# 		python3 train-adv.py \
#             --arch resnet34 \
#             --device cuda:1 \
#             --batch_size ${BATCH_SIZE} \
#             --max_epoch ${EPOCHS} \
#             --lr 1e-4 \
#             --num_worker 8 \
#             --seed 100 \
#             --model_save_dir server/checkpoints \
#             --model_save_name resnet34-adv-${EPSILON}_${ALPHA}_${NUM_STEPS} \
#             --data ~/datasets/gc10_none_mask_divided \
#             --logdir server/runs \
#             --epsilon ${EPSILON} \
#             --alpha ${ALPHA} \
#             --iters ${NUM_STEPS}
# 		done
# 	done
# done

for LR in 1e-3 1e-4 1e-5 2e-3 2e-4 2e-5 3e-3 3e-4 3e-5
do
    python3 train-adv.py \
        --arch resnet34 \
        --device cuda:1 \
        --batch_size ${BATCH_SIZE} \
        --max_epoch ${EPOCHS} \
        --lr ${LR} \
        --num_worker 8 \
        --seed 100 \
        --model_save_dir server/checkpoints \
        --model_save_name resnet34_adv~lr=${LR} \
        --data ~/datasets/gc10_none_mask_divided \
        --logdir server/runs \
        --epsilon 2 \
        --alpha 0.03 \
        --iters 10
done

python3 train-adv.py \
    --arch resnet34 \
    --device cuda:1 \
    --batch_size 128 \
    --max_epoch 200 \
    --lr 1e-4 \
    --num_worker 8 \
    --seed 100 \
    --model_save_dir server-re/checkpoints \
    --model_save_name adv~lr=1e-4 \
    --data ~/datasets/new_dataset \
    --logdir server-re/runs \
    --epsilon 2 \
    --alpha 0.03 \
    --iters 10

python3 train-adv.py \
    --arch resnet34 \
    --device cuda:1 \
    --batch_size 128 \
    --max_epoch 100 \
    --lr 1e-4 \
    --num_worker 8 \
    --seed 100 \
    --model_save_dir server-re/checkpoints \
    --model_save_name adv_re~lr=1e-4 \
    --data ~/datasets/gc10_none_mask_divided \
    --logdir server-re/runs \
    --epsilon 2 \
    --alpha 0.03 \
    --iters 10