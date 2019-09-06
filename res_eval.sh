## something_v1 rgb new_length=1, inner_abp, res50
#CUDA_VISIBLE_DEVICES=1,2,3 \
python main.py something RGB \
    --evaluate \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --arch res_iabp_50 \
    --new_length 1 \
    --num_segments 16 \
    -b 24 -j 8 \
    --short_len 16 \
    --result_path sth_v1_results/iabp/rgb_res50 \
    --resume sth_v1_results/iabp/rgb_res50/_rgb_ep_35_checkpoint.pth.tar
