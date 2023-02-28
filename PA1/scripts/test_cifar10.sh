read -p "Enter the experiment name: " exp_name
read -p "Enter the index of model to load: " load_index

python ../test.py --exp_name ${exp_name} --model_name resnet34 --load_dir ../models/CIFAR10/${load_index} --dataset_dir ../data

