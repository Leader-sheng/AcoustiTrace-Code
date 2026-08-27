# 第三方素材声明

## Greatest Hits / Visually Indicated Sounds

下列目录中的文件衍生自 *Visually Indicated Sounds* 工作发布的
**Greatest Hits** 数据集：

- `i2av_conditioning_assets/`：143 张用于 I2AV 条件输入的首图；
- `references/i2av_lat/`：143 份供 Log Attack Time evaluator 使用的对应五秒参考音频。

Greatest Hits 官方以
[知识共享署名 4.0 国际许可协议（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)
发布。数据集主页和官方下载入口位于
[andrewowens.com/vis](https://andrewowens.com/vis/)。

AcoustiTrace 发布准备过程对原始素材进行了以下修改：

- 选取与公开 benchmark 对应的 143 条记录；
- 抽取静态图像，并按发布条件图的需要进行裁剪或缩放；
- 提取五秒音频，并转换为单声道、22,050 Hz、16-bit PCM WAV。

`references/i2av_lat_sources.jsonl` 提供了每份发布素材与原始记录之间的映射。
这些衍生素材独立遵循 CC BY 4.0，不随 AcoustiTrace 源代码许可证变更。
本项目不暗示原数据集作者对 AcoustiTrace 的认可或背书。

使用这些素材时请引用原数据集论文：

```bibtex
@inproceedings{owens2016visually,
  title     = {Visually Indicated Sounds},
  author    = {Owens, Andrew and Isola, Phillip and McDermott, Josh H. and
               Torralba, Antonio and Adelson, Edward H. and Freeman, William T.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2016},
  pages     = {2405--2413}
}
```

