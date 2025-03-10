echo "gsm8k"
echo "step...1"
accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path='../llama-2-7b-chat-hf' \
  --dataset_name='gsm8k' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
  --output_dir='logs/Vanilla_FT/gsm8k/llama_2_7b/sft/lr_2e-5_False' \
  --logging_steps=1 --num_train_epochs=3 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' \
  --use_warmup=False ;
echo "step...2"
accelerate launch --num_processes=4 \
      eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='logs/Vanilla_FT/gsm8k/llama_2_7b/sft/lr_2e-5_False' \
      --dataset='gsm8k' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='gsm8k' \
      --save_path="logs/Vanilla_FT/gsm8k/llama_2_7b_sft_False.json" \
      --do_sample=False;
echo "step...3"
accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/Vanilla_FT/gsm8k/llama_2_7b/sft/lr_2e-5_False" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/Vanilla_FT/gsm8k/llama_2_7b_pure_bad_sft_False.json' \
      --eval_template='pure_bad' \
      --do_sample=False;
echo "step...4"
accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/Vanilla_FT/gsm8k/llama_2_7b/sft/lr_2e-5_False" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/Vanilla_FT/gsm8k/llama_2_7b_pure_bad_sft_False_jsr3.json' \
      --eval_template='pure_bad' \
      --num_perfix_tokens=3 \
      --do_sample=False;

echo "samsum"
echo "step...1"
accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path='../llama-2-7b-chat-hf' \
  --dataset_name='samsum' --model_family='llama2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
  --output_dir='logs/Vanilla_FT/samsum/llama_2_7b/sft/lr_2e-5_False' \
  --logging_steps=1 --num_train_epochs=3 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' \
  --max_seq_length=1024 \
  --use_warmup=False ;
echo "step...2"
accelerate launch --num_processes=4 \
      eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='logs/Vanilla_FT/samsum/llama_2_7b/sft/lr_2e-5_False' \
      --dataset='samsum' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='rouge_1' \
      --save_path="logs/Vanilla_FT/samsum/llama_2_7b_sft_False.json" \
      --do_sample=False;
echo "step...3"
accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/Vanilla_FT/samsum/llama_2_7b/sft/lr_2e-5_False" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/Vanilla_FT/samsum/llama_2_7b_pure_bad_sft_False.json' \
      --eval_template='pure_bad' \
      --do_sample=False;
echo "step...4"
accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/Vanilla_FT/samsum/llama_2_7b/sft/lr_2e-5_False" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/Vanilla_FT/samsum/llama_2_7b_pure_bad_sft_False_jsr3.json' \
      --eval_template='pure_bad' \
      --num_perfix_tokens=3 \
      --do_sample=False;