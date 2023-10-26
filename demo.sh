export PYTHONPATH=$(pwd)

TMP=0.2
TMP_REFINE=0.6
GEN_LEN=256
BS=16
NUM_SEQ=1

if [ "$CUDA_VISIBLE_DEVICES" = "0,1" ]; then
    PORT=12345
else
    PORT=65432
fi


MODEL_ID=CYCLE-350M 
MODEL_NAME=./zenodo/models/$MODEL_ID
NUM_TEST_CASES=1
AGGR_METHOD=self_refine
PROMPT_FILE="./zenodo/data/HumanEval_for_code_generation.jsonl"
TASK=HumanEval
OUTPUD_DIR="./output_test_$TASK"
rm -rf $OUTPUD_DIR
mkdir -p $OUTPUD_DIR

echo "Start running inference for $MODEL_NAME on $TASK"
INF_START_TIME=$(date +%s.%N)
accelerate launch --main_process_port=$PORT nl2code_inference.py \
    --model_name_or_path $MODEL_NAME \
    --prompt_file $PROMPT_FILE \
    --output_dir $OUTPUD_DIR \
    --gen_length $GEN_LEN \
    --batch_size $BS \
    --temperature $TMP \
    --num_return_sequences $NUM_SEQ \
    --overwrite_cache \
    2>&1 | tee $OUTPUD_DIR/inference_tmp${TMP}_seq${NUM_SEQ}.log
INF_END_TIME=$(date +%s.%N)
INF_ELAPSED_TIME=$(echo "$INF_END_TIME - $INF_START_TIME" | bc)
echo "Inference Time: $INF_ELAPSED_TIME" >> $OUTPUD_DIR/inference_tmp${TMP}_seq${NUM_SEQ}.log

echo "Done running inference for $MODEL_NAME on $TASK"
echo "Start evaluating inference results for $MODEL_NAME on $TASK"
MODEL_PRED_FILE=${OUTPUD_DIR}/prediction_tmp${TMP}_seq${NUM_SEQ}.jsonl
EXEC_RES_FILE=${OUTPUD_DIR}/exec_results_tmp${TMP}_seq${NUM_SEQ}.jsonl
python exec_eval/run_exec_eval.py \
    --prompt_file $PROMPT_FILE \
    --model_pred_file $MODEL_PRED_FILE \
    --res_file $EXEC_RES_FILE \
    2>&1 | tee ${OUTPUD_DIR}/exec_eval_tmp${TMP}_seq${NUM_SEQ}.log

COUNT=0
CONVERGE=False
PREV_SELF_REFINE_DIR=${OUTPUD_DIR}

while [ $CONVERGE == False ]; do
    echo "Build Prompt for Self-refinement for $MODEL_NAME on $TASK: $COUNT th iteration"
    SELF_REFINE_DIR=${OUTPUD_DIR}/self_refine_tmp${TMP_REFINE}_seq${NUM_SEQ}_count${COUNT}
    OT_PASS_PRED_FILE=${SELF_REFINE_DIR}/ot_passed_prediction_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl
    SELF_REFINE_INPUT=${SELF_REFINE_DIR}/self_refine_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl
    if [ $COUNT != 0 ]; then
        MODEL_PRED_FILE=${PREV_SELF_REFINE_DIR}/merged_prediction_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl
        EXEC_RES_FILE=${PREV_SELF_REFINE_DIR}/merged_exec_results_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl
    fi

    mkdir -p $SELF_REFINE_DIR
    python exec_eval/aggregate_exec_res.py \
        --task build_self_refine \
        --aggr_method $AGGR_METHOD \
        --num_test_cases $NUM_TEST_CASES \
        --pred_file $MODEL_PRED_FILE \
        --exec_res_file $EXEC_RES_FILE \
        --passed_pred_file $OT_PASS_PRED_FILE \
        --self_refine_input_file $SELF_REFINE_INPUT \

    echo "Start running self-refine inference for $MODEL_NAME on $TASK"
    INF_START_TIME=$(date +%s.%N)
    accelerate launch --main_process_port=$PORT nl2code_inference.py \
        --model_name_or_path $MODEL_NAME \
        --prompt_file $SELF_REFINE_INPUT \
        --output_dir $SELF_REFINE_DIR \
        --gen_length $GEN_LEN \
        --batch_size $BS \
        --temperature $TMP_REFINE \
        --num_return_sequences $NUM_SEQ \
        --overwrite_cache \
        2>&1 | tee ${SELF_REFINE_DIR}/self_refine_inference_tmp${TMP_REFINE}_seq${NUM_SEQ}.log
    INF_END_TIME=$(date +%s.%N)
    INF_ELAPSED_TIME=$(echo "$INF_END_TIME - $INF_START_TIME" | bc)
    echo "Inference Time: $INF_ELAPSED_TIME" >> ${SELF_REFINE_DIR}/self_refine_inference_tmp${TMP_REFINE}_seq${NUM_SEQ}.log

    echo "Start merging self-refine inference results for $MODEL_NAME on $TASK: $COUNT th iteration"
    python exec_eval/aggregate_exec_res.py \
        --task merge_self_refine \
        --orig_passed_pred_file $OT_PASS_PRED_FILE \
        --self_refine_preds_file ${SELF_REFINE_DIR}/prediction_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl \
        --merged_preds_file ${SELF_REFINE_DIR}/merged_prediction_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl \

    echo "Start evaluating self-refine inference results for $MODEL_NAME on $TASK: $COUNT th iteration"
    python exec_eval/run_exec_eval.py \
        --model_pred_file ${SELF_REFINE_DIR}/merged_prediction_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl \
        --res_file ${SELF_REFINE_DIR}/merged_exec_results_tmp${TMP_REFINE}_seq${NUM_SEQ}.jsonl \
        2>&1 | tee ${SELF_REFINE_DIR}/self_refine_exec_eval_tmp${TMP_REFINE}_seq${NUM_SEQ}.log
    CONVERGE=$(python exec_eval/verify_converge.py --prev_exec_res_file $PREV_SELF_REFINE_DIR/pass_at_k.json --curr_exec_res_file ${SELF_REFINE_DIR}/pass_at_k.json)
    PREV_SELF_REFINE_DIR=${SELF_REFINE_DIR}
    COUNT=$((COUNT+1))
    if [ $COUNT == 4 ]; then
        break
    fi
done
