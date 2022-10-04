time_info=`date +%y%m%d-%H%M`
dir_name="model_"$time_info
echo $dir_name
# python3 Unet_train.py > $dir_name"_unet_train_log.txt"

for i in {1..10}
do
    target_dir="v1_"$dir_name"_"$i
    echo $target_dir
    mkdir $target_dir
    python3 train_unetv1.py > "unet_train_log.txt"
    mv "unet_train_log.txt"  $target_dir
    mv *h5 './'$target_dir'/'
done


for i in {1..10}
do
    target_dir="v2_"$dir_name"_"$i
    echo $target_dir
    mkdir $target_dir
    python3 train_unetv2.py > "unet_train_log.txt"
    mv "unet_train_log.txt" $target_dir
    mv *h5 './'$target_dir'/'
done
