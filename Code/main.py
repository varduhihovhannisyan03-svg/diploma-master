import os
import copy
import time
import datetime

import torch
import torch.utils.data

from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from efficientnet.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from efficientnet import utils
from efficientnet.loss import CrossEntropyLoss
from efficientnet.scheduler import StepLR
from efficientnet.optim import RMSprop
from efficientnet.utils import RandAugment


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch EfficientNet Training")

    parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--batch-size", default=64, type=int, help="images per GPU, total @num_gpu x batch_size")
    parser.add_argument("--epochs", default=450, type=int, help="number of total epochs to run")
    parser.add_argument("--workers", default=16, type=int, help="number of data loading workers (default: 16)")

    # Optimizer params
    parser.add_argument("--lr", default=0.048, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=1e-5, type=float, help="weight decay (default: 1e-4)")

    # Learning rate scheduler
    parser.add_argument('--lr-warmup-init', type=float, default=1e-6, help='warmup learning rate (default: 0.0001)')
    parser.add_argument("--lr-warmup-epochs", default=3, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-decay-epochs", default=2.4, type=float, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-decay-rate", default=0.97, type=float, help="decrease lr by a factor of lr-gamma")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    parser.add_argument("--sync-bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test", action='store_true', help='model testing')

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=0, type=int, help="number of distributed processes")

    # Exponential Moving Average
    parser.add_argument("--model-ema", action="store_true", help="Exponential Moving Average")
    parser.add_argument("--model-ema-decay", type=float, default=0.9999, help="Exponential Moving Average decay")

    # Data processing
    parser.add_argument("--random-erase", default=0.2, type=float, help="random erasing probability")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation")
    parser.add_argument("--val-crop-size", default=240, type=int, help="the central crop size used for validation")
    parser.add_argument("--train-crop-size", default=240, type=int, help="the random crop size used for training")

    return parser


def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    end = time.time()
    last_idx = len(train_loader) - 1

    time_logger = utils.AverageMeter()  # img/s
    loss_logger = utils.AverageMeter()  # loss
    top1_logger = utils.AverageMeter()  # top1 accuracy
    top5_logger = utils.AverageMeter()  # top5 accuracy

    header = f"Epoch: [{epoch}]"
    for batch_idx, (image, target) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        batch_size = image.shape[0]
        time_logger.update(time.time() - end)

        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if model_ema:
            model_ema.update(model)

        torch.cuda.synchronize()

        if last_batch or batch_idx % args.print_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            loss_logger.update(reduced_loss.item(), n=batch_size)
            top1_logger.update(acc1.item(), n=batch_size)
            top5_logger.update(acc5.item(), n=batch_size)

            if args.local_rank == 0:
                print('Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                      'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f})  '
                      'Acc@1: {acc1.val:>6.4f} ({acc1.avg:>6.4f}) '
                      'Acc@5: {acc5.val:>6.4f} ({acc5.avg:>6.4f}) '
                      'LR: {lr:.3e}  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(epoch, batch_idx, len(train_loader),
                                                                               100. * batch_idx / last_idx,
                                                                               loss=loss_logger,
                                                                               acc1=top1_logger,
                                                                               acc5=top5_logger,
                                                                               lr=lr,
                                                                               data_time=time_logger))


def validate(model, criterion, test_loader, device, args, log_suffix=""):
    time_logger = utils.AverageMeter()  # img/s
    loss_logger = utils.AverageMeter()  # loss
    top1_logger = utils.AverageMeter()  # top1 accuracy
    top5_logger = utils.AverageMeter()  # top5 accuracy

    model.eval()

    end = time.time()
    last_idx = len(test_loader) - 1

    header = f"Test: {log_suffix}"
    with torch.inference_mode():
        for batch_idx, (image, target) in enumerate(test_loader):
            last_batch = batch_idx == last_idx
            batch_size = image.shape[0]
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                loss = loss.data

            torch.cuda.synchronize()

            loss_logger.update(loss.item(), n=batch_size)
            top1_logger.update(acc1.item(), n=batch_size)
            top5_logger.update(acc5.item(), n=batch_size)

            time_logger.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.print_freq == 0):
                print('{0}: [{1:>4d}/{2}]  '
                      'Time: {batch_time.val:>4.3f} ({batch_time.avg:>4.3f})  '
                      'Loss: {loss.val:>4.4f} ({loss.avg:>6.4f})  '
                      'Acc@1: {top1.val:>4.4f} ({top1.avg:>4.4f})  '
                      'Acc@5: {top5.val:>4.4f} ({top5.avg:>4.4f})'.format(header, batch_idx, last_idx,
                                                                          batch_time=time_logger,
                                                                          loss=loss_logger,
                                                                          top1=top1_logger,
                                                                          top5=top5_logger))

    print(f"{header} Loss: {loss_logger.avg:.3f} Acc@1 {top1_logger.avg:.3f} Acc@5 {top5_logger.avg:.3f}")
    return loss_logger.avg, top1_logger.avg, top5_logger.avg


def load_data(train_dir, val_dir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    dataset = utils.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(magnitude=9, magnitude_std=0.5, num_operations=2),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=args.random_erase),
        ]),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = utils.ImageFolder(
        val_dir,
        transform=transforms.Compose([
            transforms.Resize(val_resize_size, interpolation=interpolation),
            transforms.CenterCrop(val_crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
    )
    print("Creating data loaders")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    os.makedirs('weights', exist_ok=True)
    utils.init_distributed_mode(args)
    print(args)
    utils.random_seed(args.local_rank)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    train_dataset, test_dataset, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              )

    print("Creating model")
    model = efficientnet_b1()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = CrossEntropyLoss()
    parameters = utils.add_weight_decay(model, args.weight_decay)
    optimizer = RMSprop(parameters,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        eps=0.0316,
                        alpha=0.9,
                        )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scheduler = StepLR(optimizer,
                       decay_epochs=args.lr_decay_epochs,
                       decay_rate=args.lr_decay_rate,
                       warmup_epochs=args.lr_warmup_epochs,
                       warmup_lr_init=args.lr_warmup_init,
                       )
    model_ema = None
    if args.model_ema:
        model_ema = utils.EMA(model, decay=1.0 - args.model_ema_decay)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

        if model_ema:
            model_ema.model.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test:
        print("Start testing")
        start_time = time.time()
        model_ema = torch.load('weights/last.pth', 'cuda')['model'].float()
        _, acc1, acc5 = validate(model_ema, criterion, test_loader, device=device, args=args, log_suffix='EMA')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Testing time: {total_time_str}")
    else:
        print("Start training")
        start_time = time.time()
        best = 0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, model_ema, scaler)
            scheduler.step(epoch)

            validate(model, criterion, test_loader, device=device, args=args)
            loss, acc1, acc5 = validate(model_ema.model, criterion, test_loader, device=device, args=args,
                                        log_suffix="EMA")

            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.model.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            state_ema = {
                'model': copy.deepcopy(model_ema.model).half()
            }

            torch.save(checkpoint, 'weights/last.ckpt')
            torch.save(state_ema, 'weights/last.pth')
            if acc1 > best:
                torch.save(checkpoint, 'weights/best.ckpt')
                torch.save(state_ema, 'weights/best.pth')
            best = max(acc1, best)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time: {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
