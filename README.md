FedMPS: https://github.com/wenxinyang1026/FedMPS

SFD: https://github.com/zhb2000/SFD

```bash
python exps/federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --train_shots_max 110 --test_shots 15 --stdev 1 --alph 0.1 --beta 0.02 --gama 5 --rounds 500 --gpu 0 --enable_safs 1
```

## Data Preparation
* Manually download the training and test datasets from the provided links; some datasets can be downloaded directly from torchvision.datasets.
* Experiments are run on Flowers (https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), DeFungi (https://www.kaggle.com/datasets/joebeachcapital/defungi), RealWaste (https://archive.ics.uci.edu/dataset/908/realwaste), CIFAR-10 (using the defalt links in torchvision), Fashion-MNIST (using the defalt links in torchvision) and Femnist (https://s3.amazonaws.com/nist-srd/SD19/by_class.zip).  
For RealWaste, please use the following code to split the downloaded files into training and test sets.
```
import splitfolders
splitfolders.ratio(input='D:\\codes\\data\\realwaste-main', output='D:\\codes\\data\\realwaste', seed=1337, ratio=(0.7, 0.3))
```
For Femnist, please place downloaded by_class.zip under data/femnist/data/raw_data/ and then unzip it.

## Running the experiments
The entry point of a single experiment is exps/federated_main.py
* To train the FedMPS on Flowers with n=3:
```
python federated_main.py --alg ours --dataset flowers --num_classes 5 --num_users 20 --ways 3 --shots 28 --train_shots_max 30 --test_shots 10 --stdev 2 --gama 1 --beta 1 --rounds 400
```
* To train the FedMPS on DeFungi with n=3:
```
python federated_main.py --alg ours --dataset defungi --num_classes 5 --num_users 20 --ways 3 --shots 27 --train_shots_max 29 --test_shots 7 --stdev 2 --gama 0.01 --beta 1 --rounds 800
```
* To train the FedMPS on RealWaste with n=3:
```
python federated_main.py --alg ours --dataset realwaste --num_classes 9 --num_users 20 --ways 3 --shots 9 --train_shots_max 10 --test_shots 4 --stdev 1 --gama 0.01 --beta 1 --rounds 6000
```
* To train the FedMPS on CIFAR-10 with n=3:
```
python federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --train_shots_max 110 --test_shots 15 --stdev 1 --gama 5 --beta 0.2 --rounds 500
```
* To train the FedMPS on Fashion-MNIST with n=3:
```
python federated_main.py --alg ours --dataset fashion --num_classes 10 --num_users 20 --ways 3 --shots 100 --train_shots_max 110 --test_shots 15 --stdev 2 --gama 1 --beta 1 --rounds 800
```
* To train the FedMPS on Femnist with n=3:
```
python federated_main.py --alg ours --dataset femnist --num_classes 62 --num_users 20 --ways 3 --shots 95 --train_shots_max 96 --test_shots 15 --stdev 1 --gama 10 --beta 1 --rounds 1000
```

----

目前我通过如下命令来训练FedMPS：

python exps/federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --train_shots_max 110 --test_shots 15 --stdev 1 --alph 0.1 --beta 0.02 --gama 5 --rounds 300 --gpu 0

现在我想要对该训练方法进行进一步的改进：

想要将其向着SFD中的解耦多阶段训练靠拢，

想要使用上方的方式，让每个客户端先训练一下各自的本地模型，保存成 低级编码器 高级编码器 投影器 分类头 4个组件。

在一阶段训练结束后，我想要每个客户端将本地的图片经过 低级编码器 编码成 低级特征 ，通过添加 一点 基于全局的均值和协方差 生成的 噪声 来实现隐私保护，然后汇聚到全局，来训练一个GAN网络，用它来生成 虚拟合成的低级特征，然后将其分发给每个客户端（补充缺失类 或 样本稀少的类），然后再将 虚拟合成特诊和本地真实的编码低级特征 混合 构建一个训练集，来训练 本地的 高级编码器 投影器 分类头

目前我要进行一阶段的代码调整，然后进行一阶段的训练，目前先为我制定一个一阶段的代码调整方案

---

断点恢复训练：

```bash
python exps\run_stage1.py --log_dir "<LOGDIR>" --resume_ckpt_path "<LOGDIR>\stage1_ckpts\latest.pt" --reuse_split 1 --rounds 300 --latest_ckpt_interval 20
```

建议（避免 latest.pt 被误覆盖）：
- 默认情况下，如果 `<LOGDIR>\stage1_ckpts\latest.pt` 已存在但你没有传 `--resume_ckpt_path`，程序会直接报错停止，防止误覆盖。
- 如果你确认要从头重跑并允许覆盖 latest.pt，请显式加 `--allow_restart 1`。
- 稳妥起见，程序会额外按间隔保存不覆盖快照：`latest_rXXXX.pt`（可用 `--latest_history_interval` 调整）。

----

