read -p "Enter the index of model to load: " load_index
python ../train.py --exp_name CIFAR10_ft --model_name resnet34 --batch_size 128 --save_interval 5 --epochs 200 --lr 0.01 --load_dir models/CIFAR10_ft/${load_index} --imagenet_pretrained --freeze --freeze_epoch 50
