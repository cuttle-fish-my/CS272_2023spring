read -p "Enter the index of model to load: " load_index
python ../train.py --exp_name CrowdCounting --model_name CrowdCountingResnet --dataset_name CrowdCounting --batch_size 8 --save_interval 5 --epochs 200 --lr 5e-5 --load_dir ../models/CrowdCounting/${load_index} --save_dir ../models --dataset_dir ../data

