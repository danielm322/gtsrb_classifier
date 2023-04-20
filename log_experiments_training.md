# GTSRB Classifier Experiments Log

## Exp version 122771

```yaml
arch_name: resnet18
num_classes: 43
input_channels: 3
loss_fn: cross_entropy
optimizer_lr: 0.0001
optimizer_weight_decay: 0.0001
max_nro_epochs: 200
data_path: /home/users/farnez/Projects/data/gtsrb-data/
img_size:
- 32
- 32
batch_size: 64
num_workers: 10
seed: 10
shuffle: true
pin_memory: true
custom_transforms: false
train_transforms: null
valid_transforms: null
test_transforms: null
```

## Exp version 122801

```yaml
arch_name: resnet18
num_classes: 43
input_channels: 3
loss_fn: cross_entropy
optimizer_lr: 0.0001
optimizer_weight_decay: 0.0001
max_nro_epochs: 500
data_path: /home/users/farnez/Projects/data/gtsrb-data/
img_size:
- 32
- 32
batch_size: 64
num_workers: 10
seed: 10
shuffle: true
pin_memory: true
custom_transforms: false
train_transforms: null
valid_transforms: null
test_transforms: null
```

## Exp version 122805

```yaml
arch_name: resnet18
num_classes: 43
input_channels: 3
loss_fn: cross_entropy
optimizer_lr: 0.0001
optimizer_weight_decay: 0.0001
max_nro_epochs: 500
data_path: /home/users/farnez/Projects/data/gtsrb-data/
img_size:
- 32
- 32
batch_size: 64
num_workers: 10
seed: 10
shuffle: true
pin_memory: true
custom_transforms: false
train_transforms: null
valid_transforms: null
test_transforms: null
```

Here I changed the `DropBlock2D` scheduler inside Resnet architecture.
The `LinearScheduler` has `nr_steps=25e3`; previous experiments had `nr_steps=5e3`

```python
self.dropblock2d = LinearScheduler(
                    DropBlock2D(drop_prob=self.drop_prob, block_size=3),
                    start_value=0.0,
                    stop_value=self.drop_prob,
                    nr_steps=int(25e3)
                )
```
