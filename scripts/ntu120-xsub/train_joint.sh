
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_dualhead.py --config config/ntu120-xsub/train_joint.yaml \
    --work-dir work_dir/ntu120-xsub/train_joint \
    --base-lr 0.05 --device 0 1 2 3 \
    --step 40 60 80 \
    --seed 666 \
    --batch-size 64 --forward-batch-size 64 --test-batch-size 64 \
    --num-epoch 300
#