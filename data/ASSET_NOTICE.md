# Third-party asset notice

## Greatest Hits / Visually Indicated Sounds

The files in the following directories are derived from the **Greatest Hits**
dataset released with *Visually Indicated Sounds*:

- `i2av_conditioning_assets/`: 143 selected still images used as I2AV
  conditioning inputs;
- `references/i2av_lat/`: 143 corresponding five-second reference-audio
  clips used by the Log Attack Time evaluator.

The dataset authors publish Greatest Hits under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The official dataset page and downloads are available at
[andrewowens.com/vis](https://andrewowens.com/vis/).

AcoustiTrace release preparation modified the source material as follows:

- selected a benchmark-specific subset of 143 recordings;
- extracted still images, with cropping or resizing where needed for the
  released conditioning inputs; and
- extracted five-second audio clips and converted them to mono, 22,050 Hz,
  16-bit PCM WAV.

The mapping from every released asset to its source recording is provided in
`references/i2av_lat_sources.jsonl`. These derived assets remain subject to
CC BY 4.0 independently of the license applied to AcoustiTrace source code.
No endorsement by the original dataset authors is implied.

Please cite the source dataset when using these assets:

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

