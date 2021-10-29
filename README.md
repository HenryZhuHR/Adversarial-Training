# Adversarial-Training


# Github
✅ update
📝 new article
🚀 deploy
🙈 ignore 
⚡ fix problem
⚡ fix conflict
🛠️ hexo scripts

⚙️ Added ( 新加入的需求 )
🛠️ Fixed ( 修复 bug )
📝 Changed ( 完成的任务 )
📤 Updated ( 完成的任务，或者由于第三方模块变化而做的变化 )

# Files
```bash
├─api_robustModel
│  ├─models
├─modules
│  ├─attack
│  ├─dataset
│  ├─models
│  └─utils
├─generate-adv_exp.py   # generate-adv_exp
├─predict.py            # 
├─test.py               # test robust model
├─train-adv.py          # adversarial training by AVMixup
├─train-pgd-avmixup.py  # adversarial training by AVMixup (old)
├─train-pure.py         # train without any adversarial method
├─write_data-table.py   # write test result to data.xlsx file
└─scripts
```

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
scp -r  ../Adversarial-Training zhr@192.168.1.161:~/project 

scp api_robustModel/models/resnet34.pt ubuntu@192.168.1.161:~/Robust_AI_2021/api_robustModel

scp train-adv.py zhr@192.168.1.161:~/project/Adversarial-Training
scp scripts/train-avmixup.sh zhr@192.168.1.161:~/project/Adversarial-Training/scripts
```

dataset
```bash
scp -r gc10_none_mask_divided zhr@192.168.1.161:~/datasets 
scp -r E:/datasets/new_dataset zhr@192.168.1.161:~/datasets 
```

# Download from remote
server
```bash
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/server C:/Users/henryzhu/Projects/Adversarial-Training
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/server E:/Projects/

# xlsx
scp -r zhr@192.168.1.161:~/project/Adversarial-Training/data-res.xlsx E:/Projects/Adversarial-Training/data--res.xlsx
```