import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    starttime = datetime.now()

    
    parser = argparse.ArgumentParser(description='CAMEL')
    parser.add_argument('--samle_rate', type=float, default=0.1, help='down sampling rate')
    parser.add_argument('--sample_seed', type=int, default=7, help='down sampling seed')
    parser.add_argument('--train_seed', type=int, default=2024, help='train seed')
    parser.add_argument('--gap_day', type=int, default=365, help='gap days between x and y')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument(
        '--model',
        type=str,
        default='DLinear',
        help='model name, options include [Autoformer, DLinear, FEDformer, Informer, PatchTST, iTransformer, CAMEL, PhaseFormer, MixLinear, FreqCycle, stgcn, gwn, astgcn, pdformer]'
    )

    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../data/pems/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='pems03_all_common_flow.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--adj_path', type=str, default='', help='path to adjacency matrix')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    parser.add_argument('--enc_in', type=int, default=151, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=151, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=151, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--enable_camel', action='store_true', default=False,
                        help='deprecated switch for compatibility; CAMEL should be used via --model CAMEL')
    parser.add_argument('--camel_gap_years', type=float, default=0.0,
                        help='gap in years for CAMEL latent extrapolation')
    parser.add_argument('--camel_memory_size', type=int, default=1196,
                        help='memory bank size for CAMEL CEM')
    parser.add_argument('--camel_k_retrieve', type=int, default=8,
                        help='top-k memory retrieval count')
    parser.add_argument('--camel_latent_dim', type=int, default=32,
                        help='latent dimension for CAMEL LDE')
    parser.add_argument('--camel_d_model', type=int, default=32,
                        help='feature dimension used inside CAMEL memory/dynamics/fusion modules')
    parser.add_argument('--camel_use_nll', action='store_true', default=False,
                        help='use CAMEL uncertainty-aware NLL as prediction loss')
    parser.add_argument('--lambda_mem', type=float, default=0.10,
                        help='weight for CAMEL memory contrastive loss')
    parser.add_argument('--lambda_ode', type=float, default=0.05,
                        help='weight for CAMEL ODE reconstruction loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='weight for CAMEL ODE smoothness loss')
    parser.add_argument('--camel_min_year_gap', type=float, default=1.0,
                        help='minimum year difference used to form CAMEL positive contrastive pairs')
    parser.add_argument('--camel_gap_tolerance', type=float, default=0.25,
                        help='tolerance around target gap years for CAMEL memory retrieval and positives')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--steps_per_day', type=int, default=288,
                help='number of data points per day; 288 for 5-min, 24 for hourly, 1 for daily')
    parser.add_argument('--ablate_memory', action='store_true', help='remove memory branch')
    parser.add_argument('--ablate_ode', action='store_true', help='remove ODE branch')
    parser.add_argument('--ablate_atf', action='store_true', help='replace ATF gating with simple average')
    parser.add_argument('--use_which_ode',type=int, default=0, help='using which ode')
    parser.add_argument('--camel_tau', type=float, default=0.1)
    parser.add_argument('--camel_sim_threshold', type=float, default=0.5)
    parser.add_argument('--camel_sim_scale', type=float, default=10.0)
    parser.add_argument('--camel_tau_contrast', type=float, default=0.07)

    parser.add_argument('--period_len', type=int, default=24,
                        help='period length for phase decomposition')
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='latent dimension in PhaseFormer')
    parser.add_argument('--phase_encoder_hidden', type=int, default=32,
                        help='hidden dim of phase encoder MLP')
    parser.add_argument('--predictor_hidden', type=int, default=64,
                        help='hidden dim of phase predictor MLP')
    parser.add_argument('--phase_attn_heads', type=int, default=4,
                        help='number of heads in cross-phase routing')
    parser.add_argument('--phase_attn_dropout', type=float, default=0.0,
                        help='dropout in phase attention')
    parser.add_argument('--phase_attn_use_relpos', action='store_true', default=False,
                        help='use relative positional design flag')
    parser.add_argument('--phase_attn_window', type=int, default=0,
                        help='optional local window size, 0 means None')
    parser.add_argument('--phase_attention_dim', type=int, default=0,
                        help='attention dim override, 0 means None')
    parser.add_argument('--phase_num_routers', type=int, default=8,
                        help='number of routing tokens')
    parser.add_argument('--phase_use_pos_embed', action='store_true', default=False,
                        help='use phase positional embedding')
    parser.add_argument('--phase_pos_dropout', type=float, default=0.0,
                        help='dropout for phase positional embedding')
    parser.add_argument('--phase_layers', type=int, default=1,
                        help='number of cross-phase routing layers')
    parser.add_argument('--phase_encoder_use_mlp', action='store_true', default=False,
                        help='use MLP in phase encoder')
    parser.add_argument('--phase_encoder_dropout', type=float, default=0.0,
                        help='dropout in phase encoder')
    parser.add_argument('--predictor_use_mlp', action='store_true', default=False,
                        help='use MLP in predictor')
    parser.add_argument('--predictor_dropout', type=float, default=0.0,
                        help='dropout in predictor')
    parser.add_argument('--use_revin', action='store_true', default=False,
                        help='use RevIN normalization')
    parser.add_argument('--revin_affine', action='store_true', default=False,
                        help='use affine params in RevIN')
    parser.add_argument('--revin_eps', type=float, default=1e-5,
                        help='epsilon for RevIN')
    parser.add_argument('--lpf', type=int, default=16, help='low-pass filter truncation for FFT')

    parser.add_argument('--cycle', type=int, default=24,
                    help='cycle length for recurrent cycle component')
    parser.add_argument('--model_type', type=str, default='mlp',
                        help='options: [linear, mlp]')
    parser.add_argument('--seg_window', type=int, default=24,
                        help='window size for sliding segmentation')
    parser.add_argument('--seg_stride', type=int, default=12,
                        help='stride for sliding segmentation')
    parser.add_argument('--window_type', type=str, default='hann',
                        help='options: [rectangular, hamming, hann, gaussian]')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    fix_seed = args.train_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise ValueError(f"Unsupported task_name: {args.task_name}")

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_srate{}_sseed{}_trainseed{}_gap{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.samle_rate,
                args.sample_seed,
                args.train_seed,
                args.gap_day,
                args.des, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
            
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_srate{}_sseed{}_trainseed{}_gap{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.samle_rate,
            args.sample_seed,
            args.train_seed,
            args.gap_day,
            args.des, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    endtime = datetime.now()
    all_time = (endtime - starttime).seconds
    print('>>>>>>>Overall time: {} seconds<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(all_time))
    
    
