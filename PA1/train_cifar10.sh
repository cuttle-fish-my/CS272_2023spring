read -p "Enter the name of experiment: " exp_name
read -p "Enter the index of model to load: " load_index
read -p "Enter if using pretrained model (True/False): " pretrained
read -p "Enter lr: " lr
python train.py --exp_name ${exp_name} --model_name resnet34 --batch_size 128 --save_interval 5 --epochs 200 --lr ${lr} --load_dir models/${exp_name}/${load_index} \
--imagenet_pretrained ${pretrained}
