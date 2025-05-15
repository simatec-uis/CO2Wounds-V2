<div align="center">
  <img src="https://github.com/simatec-uis/CO2Wounds-V2/blob/main/figures/logo.png" alt="Logo" width="160"/>
  <h1>CO2Wounds-V2</h1>
  <h2 style="font-size: 14px;">Extended Chronic Wounds Dataset From Leprosy Patients</h2>
  <h3>
    <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10647641">
      <img src="https://img.shields.io/badge/Paper-pdf-blue" alt="Paper"/>
    </a>
  </h3>
</div>

This [paper](https://ieeexplore.ieee.org/abstract/document/10647641) has been accepted at [ICIP 2024](https://2024.ieeeicip.org/). This repository provides the benchmark code and checkpoints to evaluate our CO2Wounds-V2 dataset, which contains 764 RGB images of chronic wounds acquired from 96 leprosy patients, with wound semantic segmentation annotations provided in COCO and image formats.

## Dataset Description

Download the CO2Wounds-V2 dataset free, easy, and fast [here](https://ieee-dataport.org/open-access/co2wounds-v2-extended-chronic-wounds-dataset-leprosy-patients-segmentation-and).

## Code and Library Information

The main code for this project is available in the **main.py** file within this repository. This file contains the core implementation necessary to reproduce the results presented in our work (see Table in the following section).

We have utilized the Python library [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) for this project, which provides a wide range of state-of-the-art segmentation models. Please refer to the linked repository for detailed installation instructions and additional information on the library.

The **explore_dataset.py** script included in this repository is designed to visualize the _CO2Wounds-V2_ dataset, overlaying the masks and bounding boxes as in Figure below.

If you prefer to download the pre-trained checkpoints for each architecture and encoder directly, rather than training the models yourself using the provided scripts, you can find the download links in the last column of the table below.

## Results and Evaluation

| Architectures  | Encoder      | mIoU(%) | F1(%) | Accuracy(%) | Precision(%)  | Recall(%) | Checkpoints |
|----------------|--------------|-------|-------|-------|-------|------|-------|
| DeepLabV3      | ResNeXt-50   | 68.48 | 77.86 | 98.58 | 85.81 | 78.20 | [Download](https://osf.io/2p8yz) |
| DeepLabV3+     | ResNeXt-50   | 68.23 | 78.04 | 98.51 | 81.61 | 81.69 | [Download](https://osf.io/bvkqw) |
| U-Net          | ResNeXt-50   | 69.94 | 79.44 | 98.65 | 84.35 | 80.31 | [Download](https://osf.io/gncqf) |
| FPN            | ResNeXt-50   | 68.99 | 78.36 | 98.53 | 82.94 | 81.17 | Download |
| DeepLabV3+     | ResNet-101   | 66.88 | 76.55 | 98.40 | 83.04 | 79.02 | [Download](https://osf.io/ukgmq) |
| U-Net          | ResNet-101   | 66.96 | 76.61 | 98.28 | 80.87 | 81.10 | [Download](https://osf.io/2kstu) |
| FPN            | ResNet-101   | 66.81 | 76.52 | 98.45 | 81.78 | 79.61 | [Download](https://osf.io/jmt2u) |
| DeepLabV3+     | EfficientNet | 66.98 | 76.78 | 98.49 | 79.88 | 81.86 | [Download](https://osf.io/vxwsj) |
| U-Net          | EfficientNet | 67.71 | 77.20 | 98.51 | 83.29 | 77.80 | [Download](https://osf.io/z7f8p) |
| FPN            | EfficientNet | 67.49 | 76.84 | 98.59 | 82.63 | 80.34 | [Download](https://osf.io/d9tg4) |
| U-Net          | SegFormer    | 70.13 | 79.26 | 98.59 | 84.70 | 81.99 | [Download](https://osf.io/edawc) |
| FPN            | SegFormer    | 69.90 | 79.36 | 98.56 | 82.02 | 84.35 | Download |


![Figure 1: Visualization of CO2Wounds-V2 Dataset](figures/dataset_co2wounds.png)

## :warning: Data Citation Required

If you use this dataset/code in your research, please cite:

```bibtext
@inproceedings{sanchez2024co2wounds,
  title={CO2Wounds-V2: Extended Chronic Wounds Dataset From Leprosy Patients},
  author={Sanchez, Karen and Hinojosa, Carlos and Mieles, Olinto and Zhao, Chen and Ghanem, Bernard and Arguello, Henry},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={69--75},
  year={2024},
  organization={IEEE}
}

@article{monroy2023automated,
  title={Automated chronic wounds medical assessment and tracking framework based on deep learning},
  author={Monroy, Brayan and Sanchez, Karen and Arguello, Paula and Estupi{\~n}{\'a}n, Juan and Bacca, Jorge and Correa, Claudia V and Valencia, Laura and Castillo, Juan C and Mieles, Olinto and Arguello, Henry and others},
  journal={Computers in Biology and Medicine},
  volume={165},
  pages={107335},
  year={2023},
  publisher={Elsevier}
}
```

## License

The authors make data publicly available according to open data standards and license datasets under the Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) license.

## Contact

Linkedin: [https://www.linkedin.com/in/karenyanethsanchez/](https://www.linkedin.com/in/karenyanethsanchez/)
Twitter: [@karensanchez119](https://x.com/karensanchez119)
Email: karen.sanchez@kaust.edu.sa

