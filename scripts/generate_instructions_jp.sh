batch_dir=data/Qwen3-30B-A3B/

nohup python self_instruct/self_instaruct_japanese.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 5000 \
    --seed_tasks_path ichikara_data/merged_output.jsonl \
    --engine "davinci" \
    --model_name "Qwen/Qwen3-30B-A3B" \
    &