SEED=1
T5_PATH=/home/ruolin/Github/CPT4DST/t5-small
GPU=1
for DATASET_ORDER in 1 2 3 4 5
do
  CUDA_VISIBLE_DEVICES=${GPU} python mytrain.py \
    --do_train \
    --CL ADAPTER \
    --task_type DST \
    --model_checkpoint ${T5_PATH} \
    --saving_dir output/t5_adapter_seed${SEED}_order${DATASET_ORDER} \
    --max_history 200 \
    --dataset_list SGD \
    --n_epochs 20 \
    --test_every_step \
    --train_batch_size 8 \
    --valid_batch_size 32 \
    --test_batch_size 32 \
    --lr 3e-3 \
    --seed ${SEED} \
    --dataset_order ${DATASET_ORDER}
done
