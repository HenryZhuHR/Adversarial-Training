import os
import copy
from torchvision.models.resnet import resnet34
import tqdm
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch import cuda
from torch import optim
from torch import Tensor
from torch.utils import data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from modules import attack
from modules import models as Models


# --------------------------------------------------------
#   Args
# --------------------------------------------------------

parser = argparse.ArgumentParser(description='Adversarial Training')


parser.add_argument('--arch', type=str, choices=Models.model_zoo,
                    default=list(Models.model_zoo.keys())[0],
                    # default='resnet50',
                    )
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int)
parser.add_argument('--data', type=str, default='%s/datasets/custom' % os.path.expanduser('~'),
                    help='dataset folder')
parser.add_argument('--model_save_dir', type=str, default='server/checkpoints')
parser.add_argument('--model_save_name', type=str,
                    help='using arch name if not given')

parser.add_argument('--logdir', type=str, default='server',
                    help='train log save folder')
parser.add_argument('--model_summary', action='store_true',
                    help='if print model summary')
# adversarial training
parser.add_argument('--adversarial_training', action='store_true',
                    help='if adversarial training')
parser.add_argument('--attack_method', type=str, default='pgd')
parser.add_argument('--attack_args', nargs=argparse.REMAINDER)
args = parser.parse_args()


# Parse Args
ARCH: str = args.arch
DEVICE: str = args.device
BATCH_SIZE: int = args.batch_size    # 2
NUM_WORKERS: int = args.num_worker   # 0
MAX_EPOCH: int = args.max_epoch    # 100
LR: float = args.lr   # 0.01
SEED: int = args.seed
DATASET_DIR: str = args.data  # 'data'
MODEL_SAVE_DIR: str = args.model_save_dir  # 'checkpoints'
MODEL_SAVE_NAME: str = ARCH if args.model_save_name == None else args.model_save_name  # 'NONE'/ARCH

LOG_DIR: str = args.logdir    # 'where tensorboard data save (runs)'
IS_MODEL_SUMMARY: bool = args.model_summary

# adversarial training
IS_ADVERSARIAL_TRAINING: bool = args.adversarial_training
ATTACK_METHOD: str = args.attack_method
ATTACK_PARAMS: list = None if ATTACK_METHOD == None else args.attack_args


DATA_TRANSFORM = {
    'train': transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor()]),
    'valid': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])
}


def pgd_attack(model, images, labels, device='cpu', eps=4/255, alpha=0.01, iters=7):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
    images = torch.clamp(ori_images + 2 * eta, min=0, max=1).detach_()

    return images


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size(
        (true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    return true_dist


def one_hot(x, class_count=10):
    # 第一构造一个[class_count, class_count]的对角线为1的向量
    # 第二保留label对应的行并返回
    return torch.eye(class_count)[x, :]


def cross_entropy(input_, target, reduction='elementwise_mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res = -target * logsoftmax(input_)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res


if __name__ == '__main__':

    # create train log save folder
    os.makedirs('%s/%s' % (LOG_DIR, ARCH), exist_ok=True)

    # init Tensorborad SummaryWriter
    writer = SummaryWriter('%s/%s' % (LOG_DIR, ARCH))

    # ----------------------------------------
    #   Load dataset
    # ----------------------------------------
    train_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'),
                                     transform=DATA_TRANSFORM['train'])
    valid_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'valid'),
                                     transform=DATA_TRANSFORM['valid'])
    test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'),
                                    transform=DATA_TRANSFORM['test'])
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)
    num_class = len(train_set.classes)
    print('%s Load \033[0;32;40m%d\033[0m classes dataset' % (chr(128229),num_class))

    class_list = train_set.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(LOG_DIR,'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    # ----------------------------------------
    #   Load model and fine tune
    # ----------------------------------------
    print('%s Try to load model \033[0;32;40m%s\033[0m ...' % (chr(128229),ARCH))
    # model: nn.Module = Models.model_zoo[ARCH]()
    model=resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.to(DEVICE)

    if IS_ADVERSARIAL_TRAINING:
        model_attack: attack.BaseAttack = attack.GetAttackByName(
            ATTACK_METHOD)(model, DEVICE, ATTACK_PARAMS)
        print(model_attack.epsilon)
        print(model_attack.alpha)
        print(model_attack.num_steps)

    # ----------------------------------------
    #   tensorboard :   Add model graph
    #   torchsummary:   Summary model
    # ----------------------------------------
    input_tensor_sample: Tensor = train_set[0][0]
    writer.add_graph(model, input_to_model=(
        input_tensor_sample.unsqueeze(0)).to(DEVICE))
    if IS_MODEL_SUMMARY:
        try:
            from torchsummary import summary
        except:
            print('please install torchsummary by command: pip instsll torchsummary')
        else:
            print(summary(model, input_tensor_sample.size(),
                  device=DEVICE.split(':')[0]))

    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    loss_function.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----------------------------------------
    #   set train random seed
    # ----------------------------------------
    if SEED is not None:
        torch.manual_seed(SEED)  # set seed for current CPU
        torch.cuda.manual_seed(SEED)  # set seed for current GPU
        torch.cuda.manual_seed_all(SEED)  # set seed for all GPU

    # ----------------------------------------
    #   Train model
    # ----------------------------------------
    print('train model in device: \033[0;32;40m%s\033[0m ' % DEVICE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    train_log = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    for epoch in range(1, MAX_EPOCH + 1):
        print('\033[0;32;40m[train: %s]\033[0m' % ARCH, end=' ')
        print('[Epoch] %d/%d' % (epoch, MAX_EPOCH), end=' ')
        print('[Batch Size] %d' % (BATCH_SIZE), end=' ')
        print('[LR] %f' % (LR))

        # --- train ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0    # how many data has trained
        model.train()
        pbar = tqdm.tqdm(train_loader)
        # mini batch
        for images, labels in pbar:
            images: Tensor = images.to(DEVICE)
            labels: Tensor = labels.to(DEVICE)
            batch = images.size(0)
            num_data += batch

            is_mixup = True
            if IS_ADVERSARIAL_TRAINING and is_mixup:
                # images = model_attack.att ack(images, labels)
                images = pgd_attack(
                    model, images, labels, device=DEVICE, eps=4/255, alpha=0.01, iters=7)
                # Mixup
                alpha_ = 9999.0
                lam = np.random.beta(alpha_, alpha_)

                index = torch.randperm(images.size(0)).cuda()
                inputs: Tensor = lam*images.cuda() + (1-lam) * \
                    images[index, :].cuda()
                labels_a, labels_b = labels, labels[index]
                labels_a: Tensor = one_hot(labels_a, 10)
                labels_b: Tensor = one_hot(labels_b, 10)

                output: Tensor = model(inputs.to(DEVICE))
                _, pred = torch.max(output, 1)
                loss: Tensor = lam * cross_entropy(output, smooth_one_hot(labels_a, 10, 0.3).to(DEVICE))+(
                    1 - lam) * cross_entropy(output, smooth_one_hot(labels_b, 10, 0.3).to(DEVICE))
            else:
                output: Tensor = model(images)
                _, pred = torch.max(output, 1)
                loss: Tensor = loss_function(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == labels).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            if IS_ADVERSARIAL_TRAINING and is_mixup:
                pbar.set_description('loss:%.6f acc:%.6f lam:%.6f' %(epoch_loss / batch, epoch__acc / batch, lam))
            else:
                pbar.set_description('loss:%.6f acc:%.6f' %(epoch_loss / batch, epoch__acc / batch))
        train_loss = running_loss / num_data
        train_acc = running__acc / num_data

        # --- valid ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_loader)
            for images, labels in pbar:
                images: Tensor = images.to(DEVICE)
                labels: Tensor = labels.to(DEVICE)
                batch = images.size(0)
                num_data += batch

                output: Tensor = model(images)
                _, pred = torch.max(output, 1)
                # loss: Tensor = loss_function(output, labels)

                # epoch_loss = loss.item()
                epoch__acc = torch.sum(pred == labels).item()
                running_loss += epoch_loss
                running__acc += epoch__acc

                # pbar.set_description('loss:%.6f acc:%.6f' %(epoch_loss / batch, epoch__acc / batch))
                pbar.set_description('acc:%.6f' % (epoch__acc / batch))
            # valid_loss = running_loss / num_data
            valid_acc = running__acc / num_data

        print('Train Loss:%f Accuracy:%f' % (train_loss, train_acc))
        print('Valid Accuracy:%f' % (valid_acc))
        # print('Valid Loss:%f Accuracy:%f' % (valid_loss, valid_acc))

        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/Accuracy', train_acc, global_step=epoch)
        # writer.add_scalar('Valid/Loss', valid_loss, global_step=epoch)
        writer.add_scalar('Valid/Accuracy', valid_acc, global_step=epoch)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(
            MODEL_SAVE_DIR, '%s.pt' % ARCH))

        train_log.append([epoch, train_loss, train_acc,
                          #  valid_loss,
                          valid_acc])

    # save the best model
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(), os.path.join(
    MODEL_SAVE_DIR, '%s-best.pt' % ARCH))

    # ----------------------------------------
    #   Test model
    # ----------------------------------------
    print('\033[0;32;40m[Test: %s]\033[0m' % ARCH)
    running_loss, running__acc = 0.0, 0.0
    num_data = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader)
        for images, labels in pbar:
            images: Tensor = images.to(DEVICE)
            labels: Tensor = labels.to(DEVICE)
            batch = images.size(0)
            num_data += batch

            output: Tensor = model(images)
            _, pred = torch.max(output, 1)
            loss: Tensor = loss_function(output, labels)

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == labels).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            pbar.set_description('loss:%.6f acc:%.6f' %
                                 (epoch_loss / batch, epoch__acc / batch))
        test_loss = running_loss / num_data
        test_acc = running__acc / num_data
        print('Test Loss:%f Accuracy:%f' % (test_loss, test_acc))

    # ----------------------------------------
    #   Write logs
    # ----------------------------------------
    hparam_dict = {'batch size': BATCH_SIZE, 'lr': LR}
    metric_dict = {
        'train loss': train_loss, 'train accuracy': train_acc,
        # 'valid loss': valid_loss,
        'valid accuracy': valid_acc,
        'test accuracy': test_acc
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    os.makedirs('logs', exist_ok=True)
    import time
    finished_time = time.strftime("%m%d_%H%M", time.localtime())
    with open(os.path.join(LOG_DIR, '%s-%s.txt' % (ARCH, finished_time)), 'w') as f:
        f.write('bacth size =%d\n' % BATCH_SIZE)
        f.write('lr         =%f\n' % LR)
        f.write('train epoch=%d\n' % epoch)
        f.write('device     =%s\n' % DEVICE)
        f.write('test_loss:%f,test_acc:%f\n' % (test_loss, test_acc))
        f.write('epoch\ttrain loss\ttrain accuracy\tvalid loss\tvalid accuracy\n')
        for item in train_log:
            f.write('{epoch:.6f}\t{train_loss:.6f}\t{train_acc:.6f}\t{valid_acc:.6f}\n'.format(
                epoch=epoch, train_loss=train_loss, train_acc=train_acc,  valid_acc=valid_acc
            ))
