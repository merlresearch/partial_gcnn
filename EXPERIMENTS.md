<!--
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Experiments

We provide the commands used to run the experiments published in the paper. For each experiment we provide a single command. For experiments where results are reported over multiple runs one should use incremental integer seeds starting at zero to reproduce the original results. For example, for an experiment with three runs we used `seed=0`, `seed=1` and `seed=2`.

Please note that due to randomness in certain PyTorch operations on CUDA, it may not be possible to reproduce certain results with high precision. Please see [PyTorch's manual on deterministic behavior](https://pytorch.org/docs/stable/notes/randomness.html) for more details, as well as `run_experiments.py::set_manual_seed()` for specifications on how we seed our experiments.

## MNIST6-180

Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=True base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=MNIST6-180 kernel.no_hidden=32 kernel.no_layers=3 kernel.size=7 kernel.type=MAGNet net.dropout=0 net.no_blocks=2 net.no_hidden=10 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=MNIST6-180 kernel.no_hidden=32 kernel.no_layers=3 kernel.size=7 kernel.type=Fourier net.dropout=0 net.no_blocks=2 net.no_hidden=20 net.norm=LayerNorm net.pool_blocks=[1,2] net.type=CKResNet seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001
```

## MNIST6-M

Partial equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=MNIST6-M kernel.no_hidden=32 kernel.no_layers=3 kernel.size=7 kernel.type=MAGNet net.dropout=0 net.no_blocks=2 net.no_hidden=10 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001
```

Fully equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=MNIST6-M kernel.no_hidden=32 kernel.no_layers=3 kernel.size=7 kernel.type=MAGNet net.dropout=0 net.no_blocks=2 net.no_hidden=10 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001
```

## RotMNIST

### CKResNet

##### T2
```
python main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0 train.lr_probs=1e-4
```

##### SE2
Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.monotonic_decay_loss=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.monotonic_decay_loss=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

Augerino
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

##### E2
Partial equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0 train.lr_probs=1e-4
```

Fully equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0 train.lr_probs=1e-4
```

Augerino
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1] net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

### GCNN

##### T2
```
python main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,7] net.dropout=0.3 net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm net.pool_blocks=[2,5] net.dropout_blocks=[2,5,8] net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.01 train.lr_probs=1e-3 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0 debug=False
```

##### SE2
Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,7] net.dropout=0.3 net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm net.pool_blocks=[2,5] net.dropout_blocks=[2,5,8] net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.01 train.lr_probs=1e-3 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0 debug=False
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Augerino
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=rotMNIST kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=AugerinoGCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

## CIFAR10

### CKResNet

##### T2
```
python main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

##### SE2
Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.final_spatial_dim=[2,2] net.learnable_final_pooling=True net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Augerino
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 1]" net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm "net.pool_blocks=[1, 2]" net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

##### E2
Partial equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

Fully equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.final_spatial_dim=[2,2] net.learnable_final_pooling=True net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Augerino
```
main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 1]" net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm "net.pool_blocks=[1, 2]" net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

### GCNN

##### T2
```
main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=30 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.1 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

##### SE2
Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.lr_probs=0.0001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=30 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

Augerino
```
main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=AugerinoGCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.lr_probs=0.0001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

## CIFAR100

### CKResNet

##### T2
```
python main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

##### SE2
Partial equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=2 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

Fully equivariant
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=2 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

Augerino
```
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 1]" net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm "net.pool_blocks=[1, 2]" net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

##### E2
Partial equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4
```

Fully equivariant
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.block_width_factors=[1,1,2,1] net.dropout=0 net.last_conv_T2=True net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Augerino
```
python main.py base_group.name=E2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 1]" net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm "net.pool_blocks=[1, 2]" net.type=AugerinoCKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

### GCNN

##### T2
```
python main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=30 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.1 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

##### SE2
Partial equivariant
```
main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.lr_probs=0.0001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001
```

Fully equivariant
```
main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=GCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

Augerino
```
main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR100 kernel.learn_omega0=False kernel.no_hidden=8 kernel.no_layers=3 kernel.omega0=10 kernel.size=3 kernel.type=SIREN kernel.weight_norm=False "net.block_width_factors=[1, 1, 2, 7]" net.dropout=0.3 "net.dropout_blocks=[2, 5, 8]" net.no_blocks=8 net.no_hidden=128 net.norm=BatchNorm "net.pool_blocks=[2, 5]" net.type=AugerinoGCNN no_workers=4 seed=0 train.batch_size=128 train.do=True train.epochs=205 train.lr=0.001 train.lr_probs=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0
```

## Additional Experiments

In the paper we additionally report results with:
1. a learnable final pooling layer, and
2. a T2, i.e., translation equivariant , layer at the end of the network

These experiments can be replicated by replacing the flags `net.learnable_final_pooling` (1) and `net.last_conv_t2` (2) to `True`.
