import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim

from torchvision.models.resnet import resnet34

import time
import matplotlib
import numpy as np
# print("sleep........\n")
# time.sleep(24150)
# PGD Attack
def pgd_attack(model, images, labels, eps=4/255, alpha=0.01, iters=7) :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
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
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    return true_dist


def one_hot(x, class_count=10):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]

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
    res  =-target * logsoftmax(input_)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res


matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 #transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()
                                 ]),#来自官网参数
    "val": transforms.Compose([transforms.Resize(256),               #将最小边长缩放到256
                               transforms.CenterCrop(224),
                               transforms.ToTensor()])}

train_dataset = datasets.ImageFolder(root="~/datasets/custom/train",transform=data_transform["train"])
train_num = len(train_dataset)

# {'CG_DDG': 0, 'CVN': 1, 'airplane': 2, 'automobile': 3, 'cargo_ship': 4,
#  'castle': 5, 'char': 6, 'cruise_ship': 7, 'monitor': 8, 'tanker_ship': 9}
ship_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in ship_list.items())

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 128 ##################
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8)

validate_dataset = datasets.ImageFolder(root="~/datasets/custom/valid",transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8)
#net = resnet101()
net = resnet34(num_classes=10) ##################
#net = torch.nn.DataParallel(net).cuda()
# load pretrain weights

# model_weight_path = "./resnet101-pre.pth"
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)#载入模型参数

# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure

# inchannel = net.fc.in_features
# net.fc = nn.Linear(inchannel, 5)

net.to(device)

#loss_function = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4) ##################

#定义两个数组
Loss_list = []
Accuracy_list = []

best_epoch = 0
best_acc = 0.0
save_path = 'PGD-7AVmixup_resNet34_ls0.5_0.5' ################## 模型保存路径
print(save_path)
n = 2  #训练次数

is_mixup = True

for epoch in range(200):
    # train
    time_start=time.time()

    # if epoch > 200:
    #     is_mixup = False
    # else:
    #     is_mixup = True

    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        #adv_images = images
        adv_images = pgd_attack(net,images,labels)
        # 2.mixup
        if is_mixup:
            alpha=9999.0
            lam = np.random.beta(alpha,alpha)
            #print("\nlam:",lam)
            index = torch.randperm(adv_images.size(0)).cuda()
            inputs = lam*adv_images.cuda() + (1-lam)*images[index,:].cuda()
            labels_a, labels_b = labels, labels[index]
            labels_a = one_hot(labels_a,10)
            labels_b = one_hot(labels_b,10)
            logits = net(inputs.to(device))
            loss = lam * cross_entropy(logits, smooth_one_hot(labels_a,10,0.3).to(device))+(1 - lam) * cross_entropy(logits, smooth_one_hot(labels_b,10,0.3).to(device))
        else:
            logits = net(adv_images.to(device))
            loss = loss_function(logits, labels.to(device))

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    time_end=time.time()
    print('    time cost:%.2f'%(time_end-time_start))
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        # if (val_accurate > best_acc and epoch <= 100):
        #     torch.save(net.state_dict(), "./saver/" + str(n) +"_100epoch_"+save_path + '.pth')
        if (val_accurate > best_acc and epoch > 100):
            best_acc = val_accurate
            best_epoch = epoch
            torch.save(net.state_dict(), "./saver/" + str(n) +"_240epoch_" + save_path + '.pth')
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.4f' %
              (epoch + 1, running_loss / step, val_accurate))
        Loss_list.append(running_loss / step)
        Accuracy_list.append(100 * val_accurate)
    # if epoch == 100:
    #     print("\n\n\n\n\n\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #     print("100 epoch:\n")
    #     print("best_acc:",best_acc,"   best_epoch:",best_epoch,"    model:",save_path)
    #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n\n\n\n\n\n")
    #     x1 = range(0, len(Accuracy_list))
    #     x2 = range(0, len(Accuracy_list))
    #     y1 = Accuracy_list
    #     y2 = Loss_list
    #     plt.subplot(2, 1, 1)
    #     plt.plot(x1, y1)
    #     plt.title("Val_set accuracy vs. epoches")
    #     plt.ylabel("Val_set accuracy")
    #     plt.subplot(2, 1, 2)
    #     plt.plot(x2, y2)
    #     plt.xlabel("Train_set loss vs. epoches")
    #     plt.ylabel("Train_set loss")
    #     plt.savefig("./saver/" +str(n) + '_100epoch_'  + save_path + '.jpg')  ##################

plt.clf()
print("\n\n\n\n\n\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("240 epoch:\n")
print("best_acc:",best_acc,"   best_epoch:",best_epoch,"    model:",save_path)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n\n\n\n\n\n")
x1 = range(0, len(Accuracy_list))
x2 = range(0, len(Accuracy_list))
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title("Val_set accuracy vs. epoches")
plt.ylabel("Val_set accuracy")
plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.xlabel("Train_set loss vs. epoches")
plt.ylabel("Train_set loss")

plt.savefig("./saver/" +str(n) + '_240epoch_' + save_path + '.jpg')  ##################
#plt.show()
print('Finished Training')