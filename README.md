# KerasDemo

Playground

## Datasets

[Dogs v.s. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

```
input/
    iris.csv
    dogcat/
        preview/
        test/
            0001.jpg
            ...
        train/
            dogs/
                dog.001.jpg
                ...
            cats/
                cat.001.jpg
                ...
        train_small/
            dogs/
            cats/
        valid_small/
            dogs/
            cats/
```

## Dogcat

```
pip3 install h5py
pip3 install keras
ipython3 dogcat.py
ipython3 dogcat_img_arg.py
```

可以观察到，`dogcat_img_arg.py` 在做了 data argumentaion 后，很好地抑制了过拟合，使得准确率由 70% 提升到 72%。

进一步，`dogcat_img_arg_vgg.py` 把神经网络扩展到 VGG 的深度，准确率停留在 50%

而 `dogcat_pretrain_vgg.py` 使用迁移学习，使准确率提高到 85%