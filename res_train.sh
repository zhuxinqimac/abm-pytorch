## something_v1 rgb new_length=3, top_abp, res34
#CUDA_VISIBLE_DEVICES=2,3 \
python main.py something RGB \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --arch res_top_34 \
    --weight-decay 5e-4 \
    --eval-freq 1 \
    --new_length 3 \
    --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 30 40 \
    --epochs 50 -b 24 -j 8 --dropout 0.5 \
    --short_len 8 \
    --result_path sth_v1_results/top_test/rgb_res34
    #--result_path sth_v1_results/top/rgb_res34

## something_v1 rgb new_length=3, inner_abp, res34
#CUDA_VISIBLE_DEVICES=2,3 \
#python main.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--arch res_iabp_34 \
    #--weight-decay 5e-4 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 25 40 \
    #--epochs 50 -b 24 -j 8 --dropout 0.5 \
    #--short_len 16 \
    #--result_path sth_v1_results/iabp/rgb_res34

## something_v1 rgb new_length=1, inner_abp, res50
#CUDA_VISIBLE_DEVICES=1,2,3 \
#python main.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--arch res_iabp_50 \
    #--weight-decay 5e-4 \
    #--eval-freq 1 \
    #--new_length 1 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 25 40 \
    #--epochs 50 -b 24 -j 8 --dropout 0.5 \
    #--short_len 16 \
    #--result_path sth_v1_results/iabp/rgb_res50

## something_v1 flow new_length=3, inner_adjacent_abp, res34
#CUDA_VISIBLE_DEVICES=1,2,3 \
#python main.py something Flow \
    #./something_v1_list/something_v1_flow_trainlist.txt \
    #./something_v1_list/something_v1_flow_vallist.txt \
    #--arch res_inabp_34 \
    #--weight-decay 5e-4 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 25 40 \
    #--epochs 50 -b 24 -j 8 --dropout 0.5 \
    #--short_len 16 \
    #--result_path sth_v1_results/inabp/flow_res34

## something_v2 rgb new_length=3, inner_abp, res34
#CUDA_VISIBLE_DEVICES=0 \
#python main.py something_v2 RGB \
    #./something_v2_list/something_v2_trainlist.txt \
    #./something_v2_list/something_v2_vallist.txt \
    #--arch res_iabp_34 \
    #--weight-decay 5e-4 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 25 40 \
    #--epochs 50 -b 24 -j 8 --dropout 0.5 \
    #--short_len 16 \
    #--result_path sth_v2_results/iabp/rgb_res34
