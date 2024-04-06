echo "Running experiment with torch_ddp"
echo
torchrun --standalone --nproc_per_node 1 --master_port 29800 finetune.py --target_f1 0.6 --plugin torch_ddp --model_type "gpt2"

echo
echo "Running experiment with torch_ddp_fp16"
echo
torchrun --standalone --nproc_per_node 1 --master_port 29800 finetune.py --target_f1 0.6 --plugin torch_ddp_fp16 --model_type "gpt2"

echo
echo "Running experiment with gemini"
echo
torchrun --standalone --nproc_per_node 1 --master_port 29800 finetune.py --target_f1 0.6 --plugin gemini --model_type "gpt2"

echo
echo "Running experiment with low_level_zero"
echo
torchrun --standalone --nproc_per_node 1 --master_port 29800 finetune.py --target_f1 0.6 --plugin low_level_zero --model_type "gpt2"