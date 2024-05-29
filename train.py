
import datetime
import os
from functools import partial
from nets.detr_training import ModelEMA
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from nets.FastDETR import DETR
from nets.detr_training import (build_loss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DetrDataset, detr_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":

    Cuda = True

    seed = 3407

    distributed = False

    fp16 = True
    
    classes_path = 'model_data/voc_classes.txt'
    
    model_path = 'model_data/van_tiny_754.pth.tar'
    
    input_shape = [640, 640]
    # ---------------------------------------------#
    #   resnet18
    #   VAN
    #   cspdarknet
    #   mobilenetone
    #   mobilenetv2
    #   mobilenetv3
    # ---------------------------------------------#
    backbone = "VAN"

    
    pretrained = False

    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 4
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    Freeze_Train = False


    Init_lr = 7.5e-4
    Min_lr = Init_lr * 0.5
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-4

    save_period = 10
    
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 10
    
    aux_loss = True
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    num_workers = 4

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    seed_everything(seed)
    # ------------------------------------------------------#
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    class_names, num_classes = get_classes(classes_path)

    model = DETR(backbone, num_classes,  aux_loss=aux_loss)


    if model_path != '':
        # ------------------------------------------------------#

        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)

        # for k,v in pretrained_dict.items():
        #     print(k,v.shape)
        # for k,v in model_dict.items():
        #     print(k,v.shape)

        load_key, no_load_key, temp_dict = [], [], {}
        new_dict =[]
        new_pretrained_dict={}
        new_pretrained_dict2={}


        if backbone=="cspdarknet":
            for k,v in pretrained_dict.items():
                if "backbone" in k:
                    new_dict.append(k)
            for k,v in pretrained_dict.items():
                    if k in new_dict:
                        new_pretrained_dict[k]=v

            for k, v in new_pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        if backbone=="mobilenetv2":
            pretrained_dictnew = {}
            for k in pretrained_dict:
                pretrained_dictnew["backbone.model." + k] = pretrained_dict[k]
            for k,v in pretrained_dictnew.items():
                if "backbone.model.feature" in k:
                    new_dict.append(k)
            for k,v in pretrained_dictnew.items():
                    if k in new_dict:
                        new_pretrained_dict[k]=v

            for k, v in pretrained_dictnew.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        if backbone=="VAN":
            pretrained_dictnew = {}
            pretrained_dict1 = {}
            pretrained_dict2 = {}
            for k ,v in pretrained_dict.items():
                if k == "state_dict":
                    pretrained_dict1[k] = v
                    for k, v in pretrained_dict1.items():
                        for k1,v1 in v.items():
                            pretrained_dict2[k1] = v1
            for k in pretrained_dict2:
                pretrained_dictnew["backbone.model." + k] = pretrained_dict2[k]
            for k,v in pretrained_dictnew.items():
                if "backbone.model.feature" in k:
                    new_dict.append(k)
            for k,v in pretrained_dictnew.items():
                    if k in new_dict:
                        new_pretrained_dict[k]=v

            for k, v in pretrained_dictnew.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        if backbone=="mobileone":
            pretrained_dictnew = {}
            for k in pretrained_dict:
                pretrained_dictnew["backbone." + k] = pretrained_dict[k]
            for k, v in pretrained_dictnew.items():
                if "backbone." in k:
                    new_dict.append(k)
            for k, v in pretrained_dictnew.items():
                if k in new_dict:
                    new_pretrained_dict[k] = v

            for k, v in pretrained_dictnew.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        elif backbone=="swintransformer-tiny":
            pretrained_dictnew={}
            for k in pretrained_dict:
                pretrained_dictnew["backbone."+k]=pretrained_dict[k]
            for k,v in pretrained_dictnew.items():
                if "backbone" in k:
                    new_dict.append(k)
            for k,v in pretrained_dictnew.items():
                    if k in new_dict:
                        new_pretrained_dict[k]=v

            for k, v in pretrained_dictnew.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        elif backbone=="convnext-tiny":
            pretrained_dictnew={}
            for k in pretrained_dict:
                pretrained_dictnew["backbone."+k]=pretrained_dict[k]
            for k,v in pretrained_dictnew.items():
                if "backbone" in k:
                    new_dict.append(k)
            for k,v in pretrained_dictnew.items():
                    if k in new_dict:
                        new_pretrained_dict[k]=v

            for k, v in pretrained_dictnew.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)

        elif backbone=="resnet18":
            for k, v in pretrained_dict.items():
                for k1, v1 in v.items():
                    for k2, v2 in v1.items():
                        # print(k2,v2.shape)
                        if "backbone" in k2:
                            new_dict.append(k2)
            for k, v in pretrained_dict.items():
                for k1, v1 in v.items():
                    for k2, v2 in v1.items():
                        if k2 in new_dict:
                            new_pretrained_dict[k2] = v2

            for k, v in new_pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict, strict=False)
      
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    detr_loss = build_loss(num_classes)
    
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if Cuda:
        if distributed:
            
            model_train = model_train.cuda(local_rank)
            detr_loss = detr_loss.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            detr_loss = detr_loss.cuda()

    ema = ModelEMA(model_train)
    
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, \
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            # lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
       
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
      
        #model.freeze_bn()

        
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        val_batch_size = batch_size
        
        if optimizer_type in ['adam', 'adamw']:
            Init_lr_fit = Init_lr
            Min_lr_fit = Min_lr
        else:
            nbs = 64
            lr_limit_max = 5e-2
            lr_limit_min = 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            #Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n],
                "lr": Init_lr_fit / 10,
            },
            # {
            #     "params": [p for n, p in model.named_parameters() if "encode" in n],
            #     "lr": Init_lr_fit / 10,
            # },
        ]
        # param_dicts = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if "backbone" in n],
        #         "lr": Init_lr_fit / 10,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if "encode" in n],
        #         "lr": Init_lr_fit / 10,
        #     },
        #     {"params": [p for n, p in model.named_parameters() if "transfomer" in n]},
        # ]
        optimizer = {
            'adam': optim.Adam(param_dicts, Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adamw': optim.AdamW(param_dicts, Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(param_dicts, Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay),
        }[optimizer_type]
        lr_scale_ratio = [ 1, 0.1]
        #lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        
        train_dataset = DetrDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = DetrDataset(val_lines, input_shape, num_classes, train=False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=detr_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))


        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=val_batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=detr_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                
                if optimizer_type in ['adam', 'adamw']:
                    Init_lr_fit = Init_lr
                    Min_lr_fit = Min_lr
                else:
                    nbs = 64
                    lr_limit_max = 5e-2
                    lr_limit_min = 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                #lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                # model.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=detr_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=detr_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
            #set_optimizer_lr(optimizer, lr_scheduler_func, epoch, lr_scale_ratio)
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1).step()
            fit_one_epoch(model_train, model, ema, detr_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                          local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
