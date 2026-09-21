'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

###
from tqdm import tqdm
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
# from utils import progress_bar

###
import time
import wandb
from wandb import Html
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###
from comm_hooks.utils import register_comm_hook_for_ddp_model, add_comm_hook_args
from optimizers import add_muon_args, build_muon_optimizer


# import os
# from torchvision import datasets

# data_path = './data'
# if not os.path.exists(os.path.join(data_path, 'cifar-10-batches-py')):
#     print("Downloading CIFAR-10 dataset...")
#     datasets.CIFAR10(root=data_path, train=True, download=True)

def init_distributed_mode(args):
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(seconds=30)
    )
    print(f"Initialized distributed training (rank {args.rank})")
    # print(f"Rank {args.rank} running on CUDA device {args.local_rank} (visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')})")
    # print(f"Rank {args.rank} uses backend {dist.get_backend()}")




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', action='store_true',help='resume from checkpoint')

###
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")

parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer')
parser.add_argument("--momentum", type=float, default=0.9, help="Set momentum for msgd.")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for optimizer.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.",)

parser.add_argument('--use_wandb', default=0, type=int, help='use wandb or not')
parser.add_argument('--col_rank', default=0, type=int, help=' "--r" is ambiguous while use "torchrun" instead of "accelerate", so use "--col_rank" instead of "--r" ')

###
add_comm_hook_args(parser) 
add_muon_args(parser)
args = parser.parse_args()
args.r=args.col_rank
init_distributed_mode(args)

supported_optimizers = ['adamw', 'sgd', 'muon']
assert args.optimizer in supported_optimizers, "`optimizer` should be one of the following: " + ', '.join(supported_optimizers)


###
if args.rank == 0 and args.use_wandb:
    if args.optimizer=="adamw":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                project=f"cifar10_resnet18_group_topk_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                project=f"cifar10_resnet18_{args.compressor}_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )
    elif args.optimizer=="sgd":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                project=f"msgd_cifar10_resnet18_group_topk_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                project=f"msgd_cifar10_resnet18_{args.compressor}_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )
    elif args.optimizer == "muon":
        wandb.init(
            project=f"muon_cifar10_resnet18_{args.compressor}_{args.use_error_feedback}",
            name=f"muon_lr{args.lr}_mu{args.muon_mu}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_ratio{args.compress_ratio}",
        )




device = torch.device(f"cuda:{args.local_rank}")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# DistributedSampler
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_sampler = DistributedSampler(trainset) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, num_workers=2, pin_memory=True) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_sampler = DistributedSampler(testset, shuffle=False) 
testloader = torch.utils.data.DataLoader(testset, batch_size=args.per_device_train_batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

# Model
print(f"[Rank {args.rank} | Local Rank {args.local_rank}] Building model..")
net = ResNet18().to(device)
# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
#     checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=map_location)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

net = DDP(net, device_ids=[args.local_rank])


criterion = nn.CrossEntropyLoss()

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
if args.optimizer == "adamw":
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
elif args.optimizer == 'sgd':  # msgd(NAG)
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
elif args.optimizer == 'muon':
    optimizer = build_muon_optimizer(
        net, lr=args.lr, scalar_lr=args.muon_scalar_lr,
        mu=args.muon_mu, weight_decay=args.weight_decay,
        scalar_weight_decay=args.muon_scalar_weight_decay,
        scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
        scalar_epsilon=args.muon_scalar_eps,
        muon_epsilon=args.muon_epsilon,
        adjust_lr=None if args.muon_adjust_lr == "none" else args.muon_adjust_lr,
        compile_orthogonalization=args.muon_compile,
        distributed_orthogonalization=not args.muon_local_orthogonalization,
    )

# Compressor
process_group = dist.distributed_c10d._get_default_group()
register_comm_hook_for_ddp_model(net, process_group, args, optimizer=optimizer)

# lr_scheduler
if args.optimizer in ("adamw", "muon"):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)
    milestone=0.1 * args.num_train_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=milestone) 
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs-milestone)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[milestone]) 

elif args.optimizer == 'sgd':  # msgd(NAG)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)




# Training
def train(epoch):
    if args.rank == 0:
        logger.info(f"\n[Epoch {epoch+1}/{args.num_train_epochs}] Training...")

    train_sampler.set_epoch(epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1) 
        temloss = torch.tensor(loss.item(), device="cuda") 
        temtotal = torch.tensor(targets.size(0), device="cuda") 
        temcorrect = torch.tensor(predicted.eq(targets).sum().item(), device="cuda") 

        dist.all_reduce(temloss, op=dist.ReduceOp.SUM)  
        temloss = temloss / args.world_size
        dist.all_reduce(temtotal, op=dist.ReduceOp.SUM)
        dist.all_reduce(temcorrect, op=dist.ReduceOp.SUM)

        if args.rank==0:
            train_loss += temloss.detach().item()
            total += temtotal.item()
            correct += temcorrect.item()
            
    if args.rank==0:
        avg_loss = train_loss / len(trainloader)
        acc = 100. * correct / total
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.3f} | Acc: {acc:.3f}% (correct: {correct}/total: {total})")

        if args.use_wandb:
            wandb.log({
                "train_loss": avg_loss,
                "train_accuracy": acc,
                "epoch": epoch,
                "train_correct": correct,
                "train_total": total,
            },step=epoch)




def test(epoch):
    global best_acc
    test_sampler.set_epoch(epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            temloss=torch.tensor(loss.item(),device="cuda")
            temtotal=torch.tensor(targets.size(0),device="cuda")
            temcorrect=torch.tensor(predicted.eq(targets).sum().item(),device="cuda")

            dist.all_reduce(temloss, op=dist.ReduceOp.SUM)
            temloss = temloss / args.world_size
            dist.all_reduce(temtotal, op=dist.ReduceOp.SUM)
            dist.all_reduce(temcorrect, op=dist.ReduceOp.SUM)

            if args.rank==0:
                test_loss += temloss.detach().item()  
                total += temtotal.item()
                correct += temcorrect.item()

 
    if args.rank==0:
        acc = 100.*correct/total
        avg_loss = test_loss / len(testloader)
        logger.info(f"[Epoch {epoch+1}] Test  Loss: {avg_loss:.3f} | Acc: {acc:.3f}% (correct: {correct}/total: {total})")

        if acc > best_acc:
            best_acc = acc


        if args.use_wandb:
            wandb.log({
                "test_accuracy": acc,
                "test_loss": avg_loss,
                "best_acc": best_acc,
                "test_correct": correct,
                "test_total": total,
                "epoch": epoch,
            }, step=epoch)   

                

    

if __name__ == '__main__':
    start_time = time.time() 
    print('Training Begins!')
    

    if args.rank == 0:
        epoch_bar = tqdm(range(start_epoch, start_epoch + args.num_train_epochs), desc="Training Epochs")
    else:
        epoch_bar = range(start_epoch, start_epoch + args.num_train_epochs)

    for epoch in epoch_bar:
        train(epoch)
        test(epoch)
        scheduler.step()

    end_time = time.time()
    print('Traning Ends!')
    total_seconds = end_time - start_time
    minutes, seconds = divmod(int(total_seconds), 60)

    logger.info(f"Total training time: {minutes:02} :{seconds:02} ") 

    if args.rank == 0:
        wandb.log({
            "total_training_time": Html(f"<p>{minutes:02}:{seconds:02}</p>"),
            "total_training_time_minutes": minutes + seconds / 60,  
        })
    dist.destroy_process_group()
