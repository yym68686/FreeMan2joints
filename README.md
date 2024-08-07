# FreeMan2joints

下载 SMPL 模型，然后将其转换为 FreeMan2joints 模型。

## 人体模型

- 主要使用[SMPL](https://smpl.is.tue.mpg.de/) v1.1.0.
  - 可以在[SMPLify](https://smplify.is.tue.mpg.de/)下载 neutral model.
  - 所有的人体模型需要重命名为 `SMPL_{GENDER}.pkl` 格式。例如, `SMPL_NEUTRAL.pkl` `SMPL_MALE.pkl` `SMPL_FEMALE.pkl`

## References

SMPL 输入形状参考：

https://github.com/google/aistplusplus_api/blob/main/demos/run_vis.py

模型下载链接：

https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip

模型下载网页：

https://smpl.is.tue.mpg.de/download.php

FreeMan dataloader 参考：

https://wangjiongw.github.io/freeman/index.html

smpl 依赖包：

https://github.com/vchoutas/smplx