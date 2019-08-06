## something rgb new_length=1 split 1, abilip
#CUDA_VISIBLE_DEVICES=2,3 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--arch res_34 \
    #--eval-freq 1 \
    #--new_length 1 \
    #--num_segments 16 \
    #--gd 30 --lr 0.001 --lr_steps 30 35 40 \
    #--epochs 45 -b 10 -j 10 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_results/res/rgb_res34_sg16lon1sho16sparse_abilip_dp05_seplr_lr0001

## something rgb new_length=1 split 1, abilip
#CUDA_VISIBLE_DEVICES=2,3 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--arch res_34 \
    #--eval-freq 1 \
    #--new_length 1 \
    #--num_segments 9 \
    #--gd 30 --lr 0.001 --lr_steps 30 35 40 \
    #--epochs 45 -b 20 -j 20 --dropout 0.5 \
    #--long_len 1 --short_len 9 \
    #--result_path Something_results/res/rgb_res34_sg9lon1sho9sparse_abilip_dp05_seplr_lr0001

## Origin something rgb new_length=1 split 1, inner_abp
##CUDA_VISIBLE_DEVICES=2,3 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--gap 2 \
    #--arch res_iabp_34 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 20 25 30 \
    #--epochs 35 -b 16 -j 24 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_results/res_iabp/rgb_res34_iabp_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001

## something rgb new_length=1 split 1, inner_abp
#CUDA_VISIBLE_DEVICES=1,2,3 \
python main_i3d.py something RGB \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --gap 2 \
    --arch res_iabp_50 \
    --weight-decay 5e-4 \
    --eval-freq 5 \
    --new_length 1 \
    --num_segments 16 \
    --gd 20 --lr 0.001 --lr_steps 40 50 60 \
    --epochs 70 -b 24 -j 8 --dropout 0.5 \
    --long_len 1 --short_len 16 \
    --result_path Something_results/res_iabp/rgb_res50_iabp_wd5en4_new1_lay234full_allsg16lon1sho16sparse_dp05_lr0001

## something flow new_length=1 split 1, inner_nabp
##CUDA_VISIBLE_DEVICES=1,2,3 \
#python main_i3d.py something Flow \
    #./something_v1_list/something_v1_flow_trainlist.txt \
    #./something_v1_list/something_v1_flow_vallist.txt \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 30 40 45 \
    #--epochs 50 -b 24 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_results/res_inabp/flow_res34_iabp_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001

## something rgb new_length=1 split 1, inner_nabp
##CUDA_VISIBLE_DEVICES=2 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 20 30 35 \
    #--epochs 40 -b 16 -j 24 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_results/res_inabp/rgb_res34_inabp_beta025_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001

## something rgb new_length=1, inner_nabp, pretrained on Kinetics
##CUDA_VISIBLE_DEVICES=2 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--arch res_inabp_34 \
    #--resume_as_pretrain \
    #--eval-freq 1 \
    #--new_length 1 \
    #--num_segments 32 \
    #--gd 30 --lr 0.001 --lr_steps 20 30 35 \
    #--epochs 40 -b 16 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 32 \
    #--result_path Something_results/res_inabp/rgb_res34_inabp07070504_lay1234temp_pool1_allsg32lon1sho32sparse_dp02_lr001 \
    #--resume Kinetics_results/res_inabp/rgb_res34_inabp07070504_lay1234temp_pool1_allsg32lon1sho32sparse_dp02_lr001/_rgb_ep_58_checkpoint.pth.tar

### something_v2 rgb new_length=3 split 1, inner_abp
##CUDA_VISIBLE_DEVICES=0 \
#python main_i3d.py something RGB \
    #./something_v2_list/something_v2_trainlist.txt \
    #./something_v2_list/something_v2_vallist.txt \
    #--weight-decay 5e-4 \
    #--gap 2 \
    #--arch res_iabp_34 \
    #--eval-freq 1 \
    #--new_length 1 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 30 40 45 \
    #--epochs 50 -b 16 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_v2_results/res_iabp/rgb_res34_iabp_hw34_wd5en4_lay234full_newlen1_allsg16lon1sho16sparse_dp05_lr0001

### Good ONE !!! something_v2 rgb new_length=3 split 1, inner_abp
##CUDA_VISIBLE_DEVICES=0 \
#python main_i3d.py something_v2 RGB \
    #./something_v2_list/something_v2_trainlist.txt \
    #./something_v2_list/something_v2_vallist.txt \
    #--input_size 224 \
    #--weight-decay 5e-4 \
    #--gap 2 \
    #--arch res_iabp_50 \
    #--eval-freq 10 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 50 60 70 \
    #--epochs 80 -b 24 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_v2_results/res_iabp/rgb_res50_iabp_hw34_wd5en4_lay1234full_sg16lon1sho16sparse_realdp05_lr0001

### something_v2 rgb new_length=3 split 1, inner_nabp
##CUDA_VISIBLE_DEVICES=1,2,3 \
#python main_i3d.py something_v2 RGB \
    #./something_v2_list/something_v2_trainlist.txt \
    #./something_v2_list/something_v2_vallist.txt \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--resume_as_pretrain \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 10 13 \
    #--epochs 16 -b 24 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_v2_results/res_inabp/rgb_res34_iabp_beta05_lay234full_allsg16lon1sho16sparse_dp05_lr0001 \
    #--resume Something_results/res_inabp/rgb_res34_inabp_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001/_rgb_ep_25_checkpoint.pth.tar

### something_v2 rgb new_length=3 split 1, inner_abp
##CUDA_VISIBLE_DEVICES=1,2,3 \
#python main_i3d.py something_v2 RGB \
    #./something_v2_list/something_v2_trainlist.txt \
    #./something_v2_list/something_v2_vallist.txt \
    #--gap 2 \
    #--arch res_iabp_34 \
    #--eval-freq 5 \
    #--new_length 3 \
    #--resume_as_pretrain \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 20 30 \
    #--epochs 40 -b 32 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_v2_results/res_iabp/rgb_v1pre_res34_iabp_lay234full_allsg16lon1sho16sparse_dp05_lr0001 \
    #--resume Something_results/res_iabp/rgb_res34_iabp_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001/_rgb_ep_30_checkpoint.pth.tar
    ##--resume Something_results/res_iabp/rgb_res34_iabp_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001/_rgb_ep_24_checkpoint.pth.tar

## something rgb new_length=1 split 1, inner_nabp
##CUDA_VISIBLE_DEVICES=2 \
#python main_i3d.py something RGB \
    #./something_v1_list/something_v1_trainlist.txt \
    #./something_v1_list/something_v1_vallist.txt \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 1 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 20 30 35 \
    #--epochs 40 -b 16 -j 24 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_results/res_inabp/rgb_res34_inabp_beta025_lay2pool_lay234full_allsg16lon1sho16sparse_dp05_lr0001

### something_v2 flow new_length=3 split 1, inner_abp
##CUDA_VISIBLE_DEVICES=1,2,3 \
#python main_i3d.py something_v2 Flow \
    #./something_v2_list/something_v2_flow_noempty_trainlist.txt \
    #./something_v2_list/something_v2_flow_noempty_vallist.txt \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 8 \
    #--new_length 3 \
    #--num_segments 16 \
    #--gd 20 --lr 0.001 --lr_steps 20 30 35 \
    #--epochs 40 -b 24 -j 12 --dropout 0.5 \
    #--long_len 1 --short_len 16 \
    #--result_path Something_v2_results/res_iabp/flow_res34_iabp_lay234full_allsg16lon1sho16sparse_dp05_lr0001

## kinetics rgb new_length=1, inner_nabp
##CUDA_VISIBLE_DEVICES=1 
#python main_i3d.py kinetics RGB \
    #./kinetics_list/trainlist.txt \
    #./kinetics_list/vallist.txt \
    #--dense_sample \
    #--gap 2 \
    #--arch res_inabp_34 \
    #--eval-freq 8 \
    #--new_length 1 \
    #--num_segments 32 \
    #--gd 30 --lr 0.01 --lr_steps 80 100 \
    #--epochs 120 -b 32 -j 12 --dropout 0.2 \
    #--long_len 1 --short_len 32 \
    #--result_path Kinetics_results/res_inabp/rgb_res34_inabp07070504_lay1234temp_pool1_allsg32lon1sho32sparse_dp02_lr001
