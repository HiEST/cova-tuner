MODEL_DIR=${SM_HP_MODEL_DIR}
PIPELINE_CONFIG_PATH=${SM_HP_PIPELINE_CONFIG_PATH}
NUM_TRAIN_STEPS=${SM_HP_NUM_TRAIN_STEPS}
SAMPLE_1_OF_N_EVAL_EXAMPLES=${SM_HP_SAMPLE_1_OF_N_EVAL_EXAMPLES}

if [ ${SM_NUM_GPUS} > 0 ]
then
   NUM_WORKERS=${SM_NUM_GPUS}
else
   NUM_WORKERS=1
fi

ls
ls ${PIPELINE_CONFIG_PATH}
ls ${MODEL_DIR}

env

echo "pdw: `pwd`"
echo "python train.py \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --model_dir ${MODEL_DIR} \
    --num_train_steps ${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples ${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --checkpoint-dir ${SM_HP_CHECKPOINT_DIR}"

python train.py \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --model_dir ${MODEL_DIR} \
    --num_train_steps ${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples ${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --checkpoint-dir ${SM_HP_CHECKPOINT_DIR}

echo "Finished"