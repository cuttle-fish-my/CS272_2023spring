read -p "Enter the index of model to load: " load_index

python ../test.py --exp_name CIFAR10 --model_name resnet34 --load_dir ../models/CIFAR10/${load_index} --dataset_dir ../data

