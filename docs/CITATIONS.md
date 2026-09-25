# 数据集引用规范（Citations）

本作品使用 4 个开源数据集，均已在 `data/README.md` 记录下载来源与许可。以下为核对后的**官方引用（BibTeX 与许可）**，用于技术报告、申报书与代码仓库引用规范。

> 核对日期：2026-09-25。所有引用均已通过 PubMed Central / MDPI / ACM DL / Zenodo 等官方渠道核实。

---

## 1. UP-Fall Detection Dataset（主性能数据集）

**论文**：Lourdes Martínez-Villaseñor, Hiram Ponce, Jorge Brieva, Ernesto Moya-Albor, José Núñez-Martínez, Carlos Torres-Rodríguez. *UP-Fall Detection Dataset: A Multimodal Approach.* Sensors, 2019, 19(9), 1988.

- DOI：`10.3390/s19091988`
- 许可：MDPI 开放获取（CC BY 4.0）
- 下载：Kaggle 镜像（`kaggle/api/v1/datasets/download/pragyachandak/upfalldataset`）；官方增强版骨架见 Zenodo 12773013
- 说明：本作品使用其穿戴传感器（5 处 IMU）+ 脑电（EEG）通道

```bibtex
@article{martinez2019upfall,
  title   = {UP-Fall Detection Dataset: A Multimodal Approach},
  author  = {Mart{\'\i}nez-Villase{\~n}or, Lourdes and Ponce, Hiram and Brieva, Jorge and Moya-Albor, Ernesto and N{\'u}{\~n}ez-Mart{\'\i}nez, Jos{\'e} and Torres-Rodr{\'\i}guez, Carlos},
  journal = {Sensors},
  volume  = {19},
  number  = {9},
  pages   = {1988},
  year    = {2019},
  doi     = {10.3390/s19091988}
}
```

## 2. UR Fall Detection Dataset（双模态数据集）

**论文**：Bogdan Kwolek, Michal Kepski. *Human fall detection on embedded platform using depth maps and wireless accelerometer.* Computer Methods and Programs in Biomedicine, 2014, 117(3), 489–501.

- DOI：`10.1016/j.cmpb.2014.09.005`
- 数据下载页：https://fenix.ur.edu.pl/mkepski/ds/uf.html （70 视频 + 70 加速度 CSV）
- 说明：本作品使用其 RGB 视频（YOLOv8-Pose 提取骨架）+ 加速度

```bibtex
@article{kwolek2014human,
  title   = {Human fall detection on embedded platform using depth maps and wireless accelerometer},
  author  = {Kwolek, Bogdan and Kepski, Michal},
  journal = {Computer Methods and Programs in Biomedicine},
  volume  = {117},
  number  = {3},
  pages   = {489--501},
  year    = {2014},
  doi     = {10.1016/j.cmpb.2014.09.005}
}
```

## 3. 老年腕戴数据集（BITS Geriatric Fall Dataset，真实老人+心率）

**论文**：Purab Nandi, K. R. Anupama, Himanish Agarwal, Kishan Patel, Vedant Bang, Manan Bharat, Madhen Vyas Guru. *Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device.* Data in Brief, 2023, 52, 109812.

- DOI（论文）：`10.1016/j.dib.2023.109812`
- DOI（数据）：`10.5281/zenodo.10013090`
- 数据网站：https://shamanx86.github.io/fall_detection_data/
- 许可：CC BY 4.0（开放获取）
- 内容：41 名志愿者、16 ADL + 8 跌倒 ×5 次；腕戴三轴加速度（20Hz）+ 医用级心率（1Hz）；Snapdragon 820c 定制腕带设备
- 说明：本作品使用其加速度 + 1Hz 真实心率作为"真实老人数据 + 跨域挑战"样本

```bibtex
@article{nandi2023inertial,
  title   = {Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device},
  author  = {Nandi, Purab and Anupama, K. R. and Agarwal, Himanish and Patel, Kishan and Bang, Vedant and Bharat, Manan and Guru, Madhen Vyas},
  journal = {Data in Brief},
  volume  = {52},
  pages   = {109812},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109812}
}
```

## 4. PPG-DaLiA（附加素材：PPG→心率）

**论文**：Attila Reiss, Ina Indlekofer, Philip Schmidt, Kristof Van Laerhoven. *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks.* Sensors, 2019, 19(14), 3079.

- DOI：`10.3390/s19143079`
- Zenodo 数据页：https://zenodo.org/record/3902728 （官方要求引用上述论文）
- 下载：Kaggle 窗口化镜像（`aiforiot/ppg-dalia`），65MB
- 说明：本作品仅作为 PPG→心率建模的附加素材，不参与主实验指标

```bibtex
@article{reiss2019deep,
  title   = {Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks},
  author  = {Reiss, Attila and Indlekofer, Ina and Schmidt, Philip and Van Laerhoven, Kristof},
  journal = {Sensors},
  volume  = {19},
  number  = {14},
  pages   = {3079},
  year    = {2019},
  doi     = {10.3390/s19143079}
}
```

---

## 依赖工具引用

- **YOLOv8-Pose**（ultralytics）：Jocher, G., Chaurasia, A., & Qiu, J. *Ultralytics YOLOv8.* 2023. https://github.com/ultralytics/ultralytics （AGPL-3.0，竞赛/研究用途免费，商业化需许可）

## 申报材料使用规范

- 技术报告/申报书须在"数据集"一节逐条列出上述 4 项引用（含 DOI）；
- 许可提醒：三个数据集均为研究用途许可；UP-Fall 与老年数据集为 CC BY 4.0，UR Fall 需注明出处（论文引用即满足）；
- 代码仓库 LICENSE：本作品代码采用 MIT（`LICENSE`），与数据集的各自许可相互独立。
