# CS-DRO
Official implementation of **Causal Structure-guided Distributionally Robust Optimization under Domain Shifts** ([KDD 2026](https://kdd2026.kdd.org/)).

<p align="center">
  <img src="main_fig.png" width="700">
</p>

For detailed derivations, proofs, additional experimental results, and extended analyses, please refer to the supplementary material. 

📄 [Supplementary Material](Supplementary_Material.pdf)

##

We use iDAG (DomainBed), the link below has details. 

https://github.com/lccurious/iDAG
https://github.com/facebookresearch/DomainBed

Here is the file we modified:

* domainbed/algorithm/algorithms.py
* domainbed/networks.py
* domainbed/trainer.py


## Preparation

### Dependencies
```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=./dataset
```

### Environments

Environment details used for the main experiments. Every experiment is conducted on a single NVIDIA GeForce RTX3090 GPU.

```
Environment:
	Python: 3.10.18
	PyTorch: 2.4.0+cu118
	Torchvision: 0.19.0+cu118
	CUDA: 11.8
	CUDNN: 9.0.1
	NumPy: 1.19.5
```

## How to Run

First, download PACS dataset in `dataset` folder. (You can find the download script in "/domainbed/script/download.py")

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

Run command with hyperparameters (HPs):

```sh
sh launch_train.sh
```



## License

This project is released under the MIT license, included [here](./LICENSE).

This project include some code from [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) (MIT license), 
[khanrc/swad](https://github.com/khanrc/swad) (MIT license), and [iDAG](https://github.com/lccurious/iDAG) (MIT license).
