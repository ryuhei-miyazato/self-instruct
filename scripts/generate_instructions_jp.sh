batch_dir=data/Qwen2_5_bakeneko-32b-instruct/

nohup python self_instruct/self_instaruct_japanese.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 5000 \
    --seed_tasks_path ichikara_data/merged_output.jsonl \
    --engine "davinci" \
    --model_name "rinna/qwen2.5-bakeneko-32b-instruct" \
    >> logs/qwen2-5_bakeneko-32b-instruct.log & 
