python3 run_softmax_ner.py --data_dir ../datasets/drone \
--labels ../datasets/drone/labels.txt --model_type bert \
--model_name_or_path bert-base-cased --output_dir ../output \
--max_seq_length 46 --overwrite_output_dir --loss_type ce \
--attention_type cosine --freeze_bert_params --num_train_epochs 3