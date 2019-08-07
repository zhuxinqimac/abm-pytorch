## something_v1 rgb new_length=1, inner_abp, res50
#CUDA_VISIBLE_DEVICES=1,2,3 \
python main.py something RGB \
    --evaluate \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --arch res_iabp_50 \
    --weight-decay 5e-4 \
    --eval-freq 1 \
    --new_length 1 \
    --num_segments 16 \
    --gd 20 --lr 0.001 --lr_steps 25 40 \
    --epochs 50 -b 24 -j 8 --dropout 0.5 \
    --short_len 16 \
    --result_path sth_v1_results/iabp/rgb_res50 \
    --resume sth_v1_results/iabp/rgb_res50/_rgb_ep_35_checkpoint.pth.tar