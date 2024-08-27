<div align="center">
  <img src="https://github.com/simatec-uis/CO2Wounds-V2/blob/main/figures/logo.png" alt="Logo" width="160"/>
  <h1>CO2Wounds-V2</h1>
  <h2 style="font-size: 16px;">Extended Chronic Wounds Dataset From Leprosy Patients</h2>
  <h3>
    <a href="https://arxiv.org/pdf/2408.10827">
      <img src="https://img.shields.io/badge/Paper-pdf-blue" alt="Paper"/>
    </a>
  </h3>
</div>

This paper has been accepted at [ICIP 2024](https://2024.ieeeicip.org/). This repository provides the benchmark code and checkpoints to evaluate our CO2Wounds-V2 dataset, which contains 764 RGB images of chronic wounds acquired from 96 leprosy patients, with wound semantic segmentation annotations provided in COCO and image formats.

## Dataset Description

Download the CO2Wounds-V2 dataset free, easy, and fast [here](https://ieee-dataport.org/open-access/co2wounds-v2-extended-chronic-wounds-dataset-leprosy-patients-segmentation-and-detection/).

- Collection
- Curation
- Annotation process

## Getting Started

Clone our repo to your local machine using the following command:

**Prerequisites**

Create a new conda environment using the provided environment.yml file.


## Results and Evaluation

| Architectures  | Encoder      | mIoU(%) | F1(%) | Accuracy(%) | Precision(%)  | Recall(%) | Checkpoints |
|----------------|--------------|-------|-------|-------|-------|------|-------|
| DeepLabV3      | ResNeXt-50   | 68.48 | 77.86 | 98.58 | 85.81 | 78.20 | Link |
| DeepLabV3+     | ResNeXt-50   | 68.23 | 78.04 | 98.51 | 81.61 | 81.69 | Link |
| U-Net          | ResNeXt-50   | 69.94 | 79.44 | 98.65 | 84.35 | 80.31 | Link |
| FPN            | ResNeXt-50   | 68.99 | 78.36 | 98.53 | 82.94 | 81.17 | Link |
| DeepLabV3+     | ResNet-101   | 66.88 | 76.55 | 98.40 | 83.04 | 79.02 | Link |
| U-Net          | ResNet-101   | 66.96 | 76.61 | 98.28 | 80.87 | 81.10 | Link |
| FPN            | ResNet-101   | 66.81 | 76.52 | 98.45 | 81.78 | 79.61 | Link |
| DeepLabV3+     | EfficientNet | 66.98 | 76.78 | 98.49 | 79.88 | 81.86 | Link |
| U-Net          | EfficientNet | 67.71 | 77.20 | 98.51 | 83.29 | 77.80 | Link |
| FPN            | EfficientNet | 67.49 | 76.84 | 98.59 | 82.63 | 80.34 | Link |
| U-Net          | SegFormer    | 70.13 | 79.26 | 98.59 | 84.70 | 81.99 | Link |
| FPN            | SegFormer    | 69.90 | 79.36 | 98.56 | 82.02 | 84.35 | Link |

## Contributing



## How to cite

If you use this dataset/code in your research, please cite:

```bibtext
@misc{your_last_name2024triplet,
  author = {Your First Name Last Name and Co-author First Name Last Name},
  title = {Triplet Detection in Laparoscopy Videos},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/Triplet-Detection-Laparoscopy}},
}
```

## License

The authors make data publicly available according to open data standards and license datasets under the Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) license.

## Contact

- Karen Sanchez

Linkedin: [https://www.linkedin.com/in/karenyanethsanchez/](https://www.linkedin.com/in/karenyanethsanchez/)
Twitter: [@karensanchez119](https://x.com/karensanchez119)
Email: karen.sanchez@kaust.edu.sa

- Carlos Hinojosa

Linkedin: [https://www.linkedin.com/in/phdcarloshinojosa/](https://www.linkedin.com/in/phdcarloshinojosa/)
Twitter: [@CarlosH_93](https://x.com/CarlosH_93)
Email: carlos.hinojosa@kaust.edu.sa


