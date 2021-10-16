# Adversarial-Training


# TB
```bash
tensorboard --logdir . --host 192.168.1.161
```
# Update files

project
```bash
scp -r Adversarial-Training zhr@192.168.1.161:~/project 
scp -r Adversarial-vertex-mixup-pytorch zhr@192.168.1.161:~/project 
```

dataset
```bash
scp -r gc10_none_mask_divided zhr@192.168.1.161:~/datasets 
```

checkpoint
```bash
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/server C:/Users/henryzhu/Projects/Adversarial-Training
```

