read -p "Enter the name of experiment: " exp_name
read -p "Enter the index of model to load: " load_index
read -p "Enter if using pretrained model (0/1): " pretrained
read -p "Enter lr: " lr

if ((${pretrained} == 1))
then
  python train.py --exp_name ${exp_name} --model_name resnet34 --batch_size 128 --save_interval 5 --epochs 200 --lr ${lr} --load_dir models/${exp_name}/${load_index} --imagenet_pretrained
else
  python train.py --exp_name ${exp_name} --model_name resnet34 --batch_size 128 --save_interval 5 --epochs 200 --lr ${lr} --load_dir models/${exp_name}/${load_index}
fi