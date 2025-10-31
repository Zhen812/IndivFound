import argparse
import os
from pretrain.src.model.encoder_ms import MultiMAE
from pretrain.src.utils.helper import set_seed, makedir
from pretrain.src.utils.log import create_logger
from pretrain.dataset import IC_Dataset
import yaml
import torch
from pretrain.train_test_pretrain import train
import torch.nn as nn
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="encoder_ms")
parser.add_argument('--experiment_run', type=str)
parser.add_argument('--gpuid', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
set_seed(args.seed)
print('experiment run: ', args.experiment_run)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)

if __name__ == '__main__':

    with open("config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)

    save_dir = os.path.join(opt['model_save_root'], args.experiment_run, args.model)
    config_dir = os.path.join(save_dir, 'config')
    log_dir = os.path.join(save_dir, 'log')
    makedir(config_dir)
    makedir(log_dir)

    shutil.copy(src=os.path.join(os.getcwd(), 'config.yaml'), dst=config_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_test_pretrain.py'), dst=config_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'pretrain_run.py'), dst=config_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'src/model/encoder_ms.py'), dst=config_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'dataset.py'), dst=config_dir)

    log, logclose = create_logger(os.path.join(log_dir, 'train.log'))

    print("load data...")
    train_dataset = IC_Dataset(opt, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])
    log("length of training set is:%d\n" % len(train_dataset))
    log("length of train loader is: %d\n" % len(train_loader))

    print("define model...")
    model = MultiMAE(opt)
    model = model.cuda().float()
    if torch.cuda.device_count() > 1:
        print("Lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    # load pretrained single-modality MAE
    print('load pre-trained VideoMAE...')
    videoMAE_path = os.path.join(os.getcwd(), "checkpoint/Video-MAE/adaptedVideoMAE-base.ckpt")
    videoMAE_dict = torch.load(videoMAE_path)
    model.load_state_dict(videoMAE_dict, strict=False)
    print('load pre-trained EEGMAE...')
    eegMAE_path = os.path.join(os.getcwd(), "checkpoint/EEG-MAE/best_model.pth")
    eegMAE_dict = torch.load(eegMAE_path)
    model.load_state_dict(eegMAE_dict, strict=False)
    print('load pre-trained ECGMAE...')
    ecgMAE_path = os.path.join(os.getcwd(), "checkpoint/ECG-MAE/best_model.pth")
    ecgMAE_dict = torch.load(ecgMAE_path)
    model.load_state_dict(ecgMAE_dict, strict=False)

    train(model=model, train_loader=train_loader, args=args, opt=opt, log=log)
    logclose()
