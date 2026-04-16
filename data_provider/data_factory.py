from torch.utils.data import DataLoader, Subset
import torch

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_M4,
    PSMSegLoader,
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
}


def create_dataloader(dataset, batch_size, shuffle, num_workers, drop_last, samle_rate=0.1, seed=42):
    torch.manual_seed(seed)

    full_size = len(dataset)
    subset_size = int(full_size * samle_rate)
    if shuffle:
        indices = torch.randperm(full_size)[:subset_size]
    else:
        indices = torch.arange(subset_size)

    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name in ['anomaly_detection', 'classification']:
            batch_size = args.batch_size
        else:
            batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(root_path=args.root_path, win_size=args.seq_len, flag=flag)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader

    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(root_path=args.root_path, flag=flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader

    if args.data == 'm4':
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        gap=args.gap_day,
        camel_mark=(args.model == 'CAMEL' or getattr(args, 'enable_camel', False)),
        steps_per_day=getattr(args, 'steps_per_day', 288),
    )
    print(flag, len(data_set))
    data_loader = create_dataloader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=1,
        drop_last=drop_last,
        samle_rate=args.samle_rate,
        seed=args.sample_seed,
    )
    return data_set, data_loader
