read -p "Enter the index of model to load: " load_index
python ../train.py --exp_name CrowdCounting --model_name CrowdCountingResnet --dataset_name CrowdCounting --load_dir ../models/CrowdCounting/${load_index} --dataset_dir ../data

