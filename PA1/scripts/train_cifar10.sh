read -p "Enter the index of model to load: " load_index

python ../train.py --exp_name CIFAR10 --model_name resnet34 --batch_size 128 --save_interval 5 --epochs 200 --lr 0.1 --load_dir ../models/CIFAR10/${load_index} --save_dir ../models --dataset_dir ../data

