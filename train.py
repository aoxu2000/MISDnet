import argparse
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from IdeaTest import make_net, freeze_net, freeze_bn, set_bn_eval
from utils.load_data import DDTIDataset, STAGE2
from utils.diceloss import BinaryDiceLoss
from tqdm import tqdm
from utils.utils import dice_coeff
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dir_checkpoint = Path('./checkpoints/')
# image_dir = Path('../data/cropped')
# mask_dir = Path('../data/mask')
# label_path = Path('../data/diagnosis.csv')

image_dir = Path('../stage2/p_image')
mask_dir = Path('../stage2/p_mask')


def train(epoch):
    net.train()
    with tqdm(range(len(train_loader)), desc='Train') as pbar:
        for batch_id, batch in enumerate(train_loader):

            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            # targets = batch['label'].long().to(device)

            pred_masks = net(inputs)
            loss = dice_loss(pred_masks, masks)

            n_iter = (epoch - 1) * len(train_loader) + batch_id + 1
            writer.add_scalar('Train/loss', loss.item(), n_iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            train_scheduler.step(loss.item())
            pbar.update(1)


def eval(epoch):
    net.eval()
    dice_score = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            # targets = batch['label'].long().to(device)

            pred_masks = net(inputs)
            # pred_masks = F.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).float()
            dice_score += dice_coeff(pred_masks, masks)

        writer.add_scalar('Val/DICE SCORE', dice_score / len(val_loader), epoch)
        print(f'Val/IOU: {dice_score / len(val_loader)}')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
    parser.add_argument('--lr', '-l', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', '-e', default=50, type=int, help='max epoch')
    # parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--val-percent', '-v', type=float, default=0.1)
    parser.add_argument('--resize', '-r', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- tensorboard ---
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    writer = SummaryWriter(log_dir=str(Path('runs')/TIME_NOW))

    # --- data ---
    t_list = [transforms.ToTensor()]
    if args.resize != -1:
        t_list.append(transforms.Resize(args.resize))
    t = transforms.Compose(t_list)

    # dataset = DDTIDataset(image_dir, mask_dir, label_path, t)
    dataset = STAGE2(image_dir, mask_dir, t)
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # --- model ---

    net = make_net("UNet")  # Options: "ResUNet" "ResNet"
    net.to(device=device)
    total_num = sum(p.numel() for p in net.parameters())
    writer.add_scalar('Amount', total_num)

    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9)

    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # ce_loss = nn.CrossEntropyLoss()
    dice_loss = BinaryDiceLoss()

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        eval(epoch)
