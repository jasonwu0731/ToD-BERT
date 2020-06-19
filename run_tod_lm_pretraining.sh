gpu=$1
model_type=$2
bert_dir=$3
output_dir=$4
add1=$5
add2=$6
add3=$7
add4=$8
add5=$9

# ./run_tod_lm_pretraining.sh 0 bert bert-base-uncased save/pretrain/ToD-BERT-MLM --only_last_turn
# ./run_tod_lm_pretraining.sh 0 bert bert-base-uncased save/pretrain/ToD-BERT-JNT --only_last_turn --add_rs_loss

CUDA_VISIBLE_DEVICES=$gpu python my_tod_pretraining.py \
    --task=usdl \
    --model_type=${model_type} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir} \
    --do_train \
    --do_eval \
    --mlm \
    --do_lower_case \
    --evaluate_during_training \
    --save_steps=2500 --logging_steps=1000 \
    --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 \
    ${add1} ${add2} ${add3} ${add4} ${add5}