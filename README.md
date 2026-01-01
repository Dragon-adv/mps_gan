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
