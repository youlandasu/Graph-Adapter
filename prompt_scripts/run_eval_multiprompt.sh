set -e
GPU=0

CL_TYPE="MULTI_PROMPT"
NUM_DOMAIN_PROMPT=100
INIT_STYLE="vocab_sample"
NUM_GRAPH_PROMPT=250

LR=0.5
EPOCH=20
FIRST_LR=0.5
FIRST_EPOCH=20
META_LR=0.5
META_EPOCH=10
MEM_SIZE=50
NUM_ACC_STEP=2
BATCH_SIZE=8
DEV_BATCH_SIZE=32
SEED=1
DATASET_ORDER=1
GRAPH_LR=1E-2
GRAPH_WD=5E-4
GRADIENT_CLIP_VAL=0

MODEL_TYPE=t5
PRETRAINED_MODEL=/home/ruolin/Github/CPT4DST/t5-small

added_params="--CL ${CL_TYPE} --num_domain_prompt ${NUM_DOMAIN_PROMPT}  --embedding_initialization ${INIT_STYLE} --num_graph_prompt ${NUM_GRAPH_PROMPT} " #--small_sample_run
train_prompt_params="--episodic_mem_size ${MEM_SIZE} --meta_lr ${META_LR} --meta_n_epochs ${META_EPOCH} --first_lr ${FIRST_LR} --first_epochs ${FIRST_EPOCH} --learning_rate=${LR} --num_train_epochs=${EPOCH} --accumulate_grad_batches=${NUM_ACC_STEP} --train_batch_size=${BATCH_SIZE} --eval_batch_size=${DEV_BATCH_SIZE} \
--graph_lr ${GRAPH_LR} --graph_wd ${GRAPH_WD}"
testing_params=""

for SEED in 2
do
  OUTPUT_DIR_NAME="output_temp/${CL_TYPE}/embed_log_softmax_order${DATASET_ORDER}_${INIT_STYLE}_mem${MEM_SIZE}_lr${LR}_epoch${EPOCH}_firstlr${FIRST_LR}_firstEpoch${FIRST_EPOCH}_metalr${META_LR}_metaEpoch${META_EPOCH}_BZ${BATCH_SIZE}_ACC${NUM_ACC_STEP}_seed${SEED}"

  mkdir -p $OUTPUT_DIR_NAME
  cp $0 $OUTPUT_DIR_NAME

  CUDA_VISIBLE_DEVICES=${GPU} python prompt_train.py \
    --dataset_list SGD \
    --task_type DST \
    --do_eval \
    --test_every_step \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path=${PRETRAINED_MODEL} \
    --output_dir=$OUTPUT_DIR_NAME \
    --gpus=1  \
    --seed ${SEED} \
    --dataset_order ${DATASET_ORDER} \
    ${added_params} \
    ${train_prompt_params} \
    ${testing_params}
done