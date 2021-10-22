# Adversarial-Training

# Train
```bash
bash scripts/train-avmixup.sh
```

# Connection to Remote
```bash
ssh zhr@192.168.1.161 -A
```

# TensorBoard
```bash
tensorboard --logdir ./server/runs --host 192.168.1.161
```

# Update files
project
```bash
scp -r  C:/Users/henryzhu/Projects/Adversarial-Training zhr@192.168.1.161:~/project 
scp -r  E:/Projects/Adversarial-Training zhr@192.168.1.161:~/project 

scp api_robustModel/models/resnet34.pt ubuntu@192.168.1.161:~/Robust_AI_2021/api_robustModel
```

dataset
```bash
scp -r gc10_none_mask_divided zhr@192.168.1.161:~/datasets 
```

# Download from remote
server
```bash
<<<<<<< HEAD
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/server C:/Users/henryzhu/Projects/Adversarial-Training
=======
>>>>>>> f9ed4aca33e27478e8f2824767c1d04193fcb3e9
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/server E:/Projects/Adversarial-Training
```