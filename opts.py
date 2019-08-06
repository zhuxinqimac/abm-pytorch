import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Approximated Bilinear Modules")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 
                                        'kinetics', 'something', 'something_v2'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="res_iabp_34")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'lstm', 'conv_lstm', 'ele_multi', 
                        'bilinear_att', 'bilinear_multi_top', 
                        'temp_att_fusion', 'selective_bi_fusion', 
                        'combined_bilinear'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
# ====== Modified ======
parser.add_argument('--lstm_out_type', type=str, help='lstm fusion type', 
                    default='avg', choices=['last', 'max', 'avg'])
parser.add_argument('--lstm_layers', type=int, help='lstm layers', 
                    default=1)
parser.add_argument('--lstm_hidden_dims', type=int, help='lstm hidden dims', 
                    default=512)
parser.add_argument('--conv_lstm_kernel', type=int, help='convlstm kernel size', 
                    default=5)
parser.add_argument('--bi_out_dims', type=int, help='bilinear out dims, should equal to num classes when bi_add_clf is false', 
                    default=101)
parser.add_argument('--bi_rank', type=int, default=1, 
                    help='rank used to approximate bilinear pooling')
parser.add_argument('--bi_att_softmax', default=False, action='store_true', 
                    help='add softmax layer for bilinear attention maps')
parser.add_argument('--bi_filter_size', type=int, default=1, 
                    help='filter size used in bilinear pooling when generating attention maps')
parser.add_argument('--bi_dropout', type=float, default=0, help='dropout used in bilinear pooling')
parser.add_argument('--bi_conv_dropout', type=float, default=0, 
                    help='sep_conv dropout during bilinear att')
parser.add_argument('--bi_add_clf', default=False, action='store_true', 
                    help='add another classifier after bilinear fusion')
parser.add_argument('--train_reverse', default=False, action='store_true', 
                    help='train with frames reversed')
parser.add_argument('--train_shuffle', default=False, action='store_true', 
                    help='train with frames shuffled')
parser.add_argument('--val_reverse', default=False, action='store_true', 
                    help='validate (test) with frames reversed')
parser.add_argument('--val_shuffle', default=False, action='store_true', 
                    help='validate (test) with frames shuffled')
parser.add_argument('--contrastive_mode', default=False, action='store_true', 
                    help='train with contrastive loss')
parser.add_argument('--contras_m1', default=1, type=float, 
                    help='contrastive loss margin 1')
parser.add_argument('--contras_m2', default=1, type=float, 
                    help='contrastive loss margin 2')
parser.add_argument('--new_length', default=1, type=int, 
                    help='number of consecutive frames as input')
parser.add_argument('--gap', default=2, type=int, 
                    help='gap between two sampled frames in dense (3d) version')
parser.add_argument('--n_history', default=1, type=int, 
                    help='number of history frames used '+\
                            'for predicting affine Theta')
parser.add_argument('--side_weight_type', default='1', type=str, 
                    choices=['1', 'avg', 'attention'], 
                    help='choose if use attention or 1 or avg for both side')
parser.add_argument('--input_size', default=224, type=int, 
                    help='input size of frames')
parser.add_argument('--use_two_stream', default=False, action='store_true', 
                    help='use two stream structure')
parser.add_argument('--use_three_stream', default=False, action='store_true', 
                    help='use three stream structure')
parser.add_argument('--ts_name_1', default='res_cafb_34', type=str, 
                    help='name of model_1 for two stream architecture')
parser.add_argument('--ts_name_2', default='res_pb_34', type=str, 
                    help='name of model_2 for two stream architecture')
parser.add_argument('--ts_name_3', default='res_aafpb_34', type=str, 
                    help='name of model_3 for three stream architecture')
parser.add_argument('--ts_ckpt_1', 
                    default='Something_results/A/_rgb_model_best.pth.tar', 
                    type=str, 
                    help='path for ckpt_1 for two stream')
parser.add_argument('--ts_ckpt_2', 
                    default='Something_results/B/_rgb_model_best.pth.tar', 
                    type=str, 
                    help='path for ckpt_2 for two stream')
parser.add_argument('--ts_ckpt_3', 
                    default='Something_results/B/_rgb_model_best.pth.tar', 
                    type=str, 
                    help='path for ckpt_3 for three stream')
parser.add_argument('--n_cframes_1', default=8, type=int, 
                    help='number of frames for model1 to do ensemble')
parser.add_argument('--n_cframes_2', default=12, type=int, 
                    help='number of frames for model2 to do ensemble')
parser.add_argument('--n_cframes_3', default=16, type=int, 
                    help='number of frames for model3 to do ensemble')
parser.add_argument('--last_result_json', 
                    default=None, 
                    type=str, 
                    help='path for last val score results in json')
parser.add_argument('--this_result_json', 
                    default=None, 
                    type=str, 
                    help='path for this val score results in json')

parser.add_argument('--this_model_weight', default=1.0, type=float,
                    help='weight of this model to combine with others')
parser.add_argument('--num_samples_extract', default=32, type=int, 
                    help='number of samples to extract from a video in \
                    frame extract mode')
parser.add_argument('--something_v1_label_csv', type=str, 
                    help='something_v1 label csv', 
                    default='./something_v1_list/something-something-v1-labels.csv')
parser.add_argument('--resume_as_pretrain', default=False, action='store_true',
                    help='resume as pretrain and train start from start_epoch')
parser.add_argument('--use_10crop_eval', action="store_true")
parser.add_argument('--ft_idx', default=5, type=int)
parser.add_argument('--short_len', default=8, type=int)
parser.add_argument('--long_len', default=8, type=int)
parser.add_argument('--dense_sample', default=False, action="store_true")
parser.add_argument('--shift_val', default=None, type=int)
parser.add_argument('--sample_all', default=False, action="store_true")
parser.add_argument('--fix_back', default=False, action="store_true")


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0000001, type=float,
                    metavar='W', help='weight decay (default: 0.0000001)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--eval_segs', default=1, type=int, help='segments used for evaluation')
parser.add_argument('--result_path', default='result', type=str,
                    metavar='LOG_PATH', help='results and log path')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="flow_", type=str)
# parser.add_argument('--flow_prefix', default="", type=str)








