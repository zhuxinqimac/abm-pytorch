import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Approximated Bilinear Modules")
parser.add_argument('dataset', type=str, choices=['kinetics', 'something', 'something_v2'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="res_iabp_34")
parser.add_argument('--num_segments', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ====== Modified ======
parser.add_argument('--new_length', default=1, type=int, 
                    help='number of consecutive frames as input')
parser.add_argument('--gap', default=2, type=int, 
                    help='gap between two sampled frames in dense (3d) version')
parser.add_argument('--side_weight_type', default='1', type=str, 
                    choices=['1', 'avg', 'attention'], 
                    help='choose if use attention or 1 or avg for both side')
parser.add_argument('--input_size', default=224, type=int, 
                    help='input size of frames')
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
parser.add_argument('--something_v1_label_csv', type=str, 
                    help='something_v1 label csv', 
                    default='./something_v1_list/something-something-v1-labels.csv')
parser.add_argument('--resume_as_pretrain', default=False, action='store_true',
                    help='resume as pretrain and train start from start_epoch')
parser.add_argument('--use_10crop_eval', action="store_true")
parser.add_argument('--short_len', default=16, type=int)
parser.add_argument('--long_len', default=1, type=int)
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
