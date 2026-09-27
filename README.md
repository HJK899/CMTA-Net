# CMTA-Net：跨模态时空注意力融合网络

面向老年群体「生理 × 行为」联合异常识别的算法模型创新作品（TASK 01 赛道）。

核心创新：跨模态互证注意力融合（mutual-verification attention）——用生理与行为两路信号的
模态间一致性做双重确认，一致时置信上调、冲突时触发复核，从网络结构层面抑制误报。

## 工程结构

```
算法模型/
├── README.md               # 本文件
├── requirements.txt        # Python 依赖
├── config.py               # 全局配置（数据路径、窗口、超参）
├── data/
│   ├── README.md           # 数据集下载指引与许可说明
│   ├── extract_pose.py     # [Day1] YOLOv8-Pose 提取姿态特征 → .npy
│   ├── preprocess_physio.py# [Day1] 生理时序 → 窗口特征 .npy
│   ├── make_synthetic_physio.py # [Day1] 合成生理数据（管线自测用，非最终实验数据）
│   ├── split_by_subject.py # [Day1] 按受试者划分 train/val/test
│   └── build_dataset.py    # [Day1] 对齐管线 + PyTorch Dataset
├── models/
│   ├── physio_branch.py    # 生理支路：轻量 1D-CNN
│   ├── behavior_branch.py  # 行为支路：GRU 姿态序列编码
│   ├── fusion.py           # ★ 跨模态互证注意力融合（核心创新）
│   └── cmta_net.py         # CMTA-Net 主模型组装
├── utils/
│   ├── losses.py           # Focal Loss（类别不平衡）
│   └── metrics.py          # 指标：混淆矩阵 / P-R-F1 / ROC+AUC / 阈值选择
├── train.py                # 训练脚本（warmup+cosine、早停、AMP）
├── evaluate.py             # 评估脚本（指标+图表）
├── checkpoints/            # 训练产物
├── outputs/                # 图表/日志输出
└── data/                   # 数据（raw / features / splits）
```

## 快速开始（按顺序跑，真实数据管线）

> 图形界面（推荐）：双击 `启动可视化界面.bat` 打开可视化验证平台，8 项验证任务一键操作。
> **Web 推理演示（产品形态）**：双击 `启动Web演示.bat`，浏览器自动打开 http://localhost:8000
> ——上传照片序列或视频即可实时四态判定（手机同 WiFi 可访问），详见下方"Web 演示"。
> 网页版界面预览：<https://hjk899.github.io/CMTA-Net/>

```bash
# 0) 安装依赖（venv 已装 torch/ultralytics，其余按需）
pip install -r requirements.txt

# 1) 下载数据集（详见 data/README.md，全部开源直链；9/25 已完成）
#    UP-Fall      → data/raw/upfall/CompleteDataSet (1).csv
#    UR Fall      → data/raw/urfall/（70 视频 + 70 acc CSV）
#    老年腕戴     → data/raw/zenodo_ger/（41 人，984 段，含 1Hz 心率）
#    PPG-DaLiA(镜像) → data/raw/ppg_dalia/dalia_window/（495 万条 8 秒 PPG 窗口）

# 2) 提取姿态特征（行为支路：UR Fall 视频 → YOLOv8-Pose 骨架）
python data/extract_pose.py --input data/raw/urfall --output data/features/pose

# 3) 构建传感/生理特征 + 样本表（UP-Fall 默认 16 通道多部位融合；UR Fall/老年 4 通道）
python data/build_features.py --raw data/raw --out data/features/physio --splits data/splits --upfall-channels full

# 4) 按受试者划分（防数据泄漏）
python data/split_by_subject.py --samples data/splits/all_samples.csv --split-dir data/splits

# 5) 训练（双模态：UR Fall；多部位传感流：UP-Fall 16通道）
python train.py --dataset urfall --stream both --epochs 50
python train.py --dataset upfall --stream physio --physio-channels 16 --epochs 60

# 6) 评估 + 出图（混淆矩阵/ROC/指标表）
python evaluate.py --checkpoint checkpoints/best_upfall_v2.pt --dataset upfall --stream physio --split test --out outputs/eval_upfall

# 7) 推理时延/参数量评测（边缘部署论证）
python benchmark_latency.py --checkpoint checkpoints/best_upfall_v2h.pt

# 8) 端到端推理接口（模块④四态诊断 + 可解释输出）
python inference.py --checkpoint checkpoints/best_urfall_v2mt.pt

# 9) 照片/视频行为判定演示（可解释可视化，答辩演示用）
python photo_demo.py -d outputs/demo_photos/fall         # 8张跌倒照片
python photo_demo.py -d 我的测试照片                      # 多组照片（子文件夹=一组）
python photo_demo.py --video 视频.mp4                    # 视频直接检测（150帧真实时序，最可靠）
```

## 数据集主线（全部开源，9/25 已全部落地）

| 数据集 | 内容 | 本作品角色 |
|---|---|---|
| UP-Fall | 17人、11类活动（5跌倒+6日常），脑电EEG+5处穿戴IMU+红外 | 传感流规模训练（按人 6:2:2） |
| UR Fall | 70段（30跌倒+40日常）视频 + 配对加速度CSV | 双模态联合主阵地（视频→骨架 × 传感）⚠官方源失效 |
| 老年腕戴 | 41名真实老人，984段，三轴加速度 + 1Hz真实心率 | 真实老人数据（含HR），可分性挑战样本 |
| PPG-DaLiA(镜像) | 495万条8秒PPG窗口（S1，HR回归标签） | PPG→心率附加素材（镜像实为3.1GB完整包，已放弃） |

> **数据状态（2026-09-26 恢复后）**：UP-Fall（78.4MB CSV）与老年腕戴（984段）已恢复并重建特征
> （UP-Fall 559段 + 老年腕戴 984段），任务1/2 复跑指标与训练记录一致（F1=0.9310）。
> **UR Fall 官方数据服务器迁移**（fenix.ur.edu.pl 数据文件404、旧域名DNS注销、Kaggle 4.5GB镜像限速不可行），
> 原始视频/CSV 无法重下；双模态四态诊断改由 `urfall_demo.py` 用真实视频骨架（fall10k）×真实传感窗口（UP-Fall）现场演示，
> UR Fall 测试集指标以已记录实验为准（见下节消融与局限）。

详细地址、放置目录、许可说明见 `data/README.md`；**完整引用规范（BibTeX + DOI + 许可）见 `docs/CITATIONS.md`**。

## Web 推理演示（产品形态）

浏览器界面，适合答辩现场与手机演示：上传**连续照片**（按动作顺序 ≥6 张）或**一段视频**，
实时输出四态诊断 + 行为异常概率 + 模态一致性 + 骨架可视化图。

```bash
pip install flask            # 已加入 requirements.txt
双击 启动Web演示.bat         # Windows 一键启动（自动打开浏览器）
# 或命令行：python web_demo.py --port 8000
# 手机同 WiFi：浏览器访问 http://<电脑IP>:8000
```

- 判定的是"动态事件"（站立→倒地的过程），照片需覆盖完整动作；视频推荐 5-15 秒
- 模型启动时加载一次，每次请求仅推理（CPU 单窗口约几十毫秒 + YOLO 骨架提取）
- 注意：本应用在本地运行模型（GitHub Pages 无法执行 Python），仓库仅托管代码

## 代码仓库与许可

- 网页版可视化界面（GitHub Pages）：<https://hjk899.github.io/CMTA-Net/>
- 代码：MIT License（见 `LICENSE`）；依赖 ultralytics (YOLOv8) 为 AGPL-3.0（竞赛/研究免费，商业化需企业许可）
- 数据：4 个开源数据集（UP-Fall / UR Fall / BITS 老年腕戴 / PPG-DaLiA），研究用途许可，引用与 DOI 见 `docs/CITATIONS.md`
- 复现：`data/raw` 与 `data/features` 不入库（可下载/重建），`checkpoints/` 与 `outputs/` 保留实验产物

## 实验结果（真实数据，2026-09-25 基准）

**主实验：CMTA-Net 传感流（UP-Fall，按受试者 6:2:2，132 个未见受试者测试样本）**

V2h = 多部位融合 + 单流诊断头（16通道：EEG + 5处穿戴加速度15通道，当前主模型）；
V2 = 旧结构（仅融合头，16通道）；V1 = 4通道（EEG+腕戴acc3）。

| 版本 | F1（3 seed） | F1 均值±std | AUC 均值±std |
|---|---|---|---|
| V2h（新结构，当前主模型） | 0.9310 / 0.9508 / 0.9298 | **0.9372 ± 0.011** | **0.9842 ± 0.002** |
| V2（旧结构） | 0.9217 / 0.9194 / 0.9134 | 0.9182 ± 0.004 | 0.9793 ± 0.001 |
| V1（4通道，基线） | 0.8750 / 0.8376 / 0.8421 | 0.8516 ± 0.021 | 0.9282 ± 0.009 |

**配置敏感性（UP-Fall，3 seed）**

| 配置 | F1 均值±std | AUC 均值±std | 结论 |
|---|---|---|---|
| V2h 16通道+诊断头（当前默认） | **0.9372 ± 0.011** | **0.9842 ± 0.002** | 最优 |
| V2 16通道（旧结构） | 0.9182 ± 0.004 | 0.9793 ± 0.001 | 结构对照 |
| V2 + 数据增强（时间扭曲+噪声） | 0.9173 ± 0.004 | 0.9810 ± 0.002 | F1 持平、AUC 微升 → 配置已近饱和 |
| V3 31通道（16+5处角速度15） | 0.8966（seed1） | 0.956（seed1） | 角速度冗余 → 退化，弃用 |
| d_model=192 | 0.9206（seed1） | 0.9711（seed1） | 与128相当，保持128 |

**部署可行性（CPU 推理评测，V2 模型）**

| 指标 | 数值 |
|---|---|
| 参数量 | 0.325 M |
| 参数内存 / checkpoint | 1.30 MB / 1.32 MB |
| 单窗口(5s)推理时延（CPU, p50/p95） | 6.0 / 9.7 ms |
| 理论吞吐 | 154 窗口/秒（20Hz 实时流的 154 倍余量） |

**双模态消融（UR Fall，14测试样本，段级划分，3 seed 报告）**

| 输入流 | F1（3 seed） | AUC（3 seed） | 说明 |
|---|---|---|---|
| CMTA-Net 双模态（传感+骨架） | 1.0000 / 1.0000 / 1.0000 | 1.0 | 跨模态互证融合 |
| 仅传感（physio-only，消融） | 0.9333 | 0.9796 | 消融：去掉行为流 |
| 仅行为（behavior-only，消融） | 1.0000 | 1.0 | 消融：去掉生理流 |

**模块④：四态联合诊断（正常/生理异常/行为异常/联合异常）**

机制：融合头给出联合异常判断；单流诊断头（physio_head/behavior_head，多任务训练）做模态归因；
双阈值（验证集 Youden 校准）组合成四态；同时输出模态一致性分数 s 供复核。

- **UR Fall 双模态（3 seed）**：融合头 F1 全 1.0；正常态识别率 3 seed 全 1.000（日常→正常）；
  联合异常召回 0.14~0.29（真实跌倒部分被归为"行为异常"）——根因是 UR Fall 加速度生理信号
  fall/adl 可分性弱（生理头 AUC≈0.86、概率贴近0.5），如实定位为数据局限而非机制缺陷。
- **UP-Fall 传感流（"生理异常"态验证）**：physio_head 16 通道 F1=0.937±0.011 / AUC=0.984±0.002，
  证明"生理异常"态在主数据上判别力强。
- 结论：四态归因能力已完整实现——行为异常（UR Fall 行为头 AUC=1.0）+ 生理异常（UP-Fall 生理头
  AUC=0.984）+ 正常（100%）+ 联合异常（两模态强判别时）；中间两态在"双模态同异常/同正常"标注的
  数据集上无 ground-truth，如实报告为开放诊断能力。

推理接口（演示/部署用）：

```bash
python inference.py --checkpoint checkpoints/best_urfall_v2mt.pt   # 双模态四态冒烟
python inference.py --checkpoint checkpoints/best_upfall_v2h.pt    # 传感流（生理异常）冒烟
```

**老年腕戴（真实老人+心率，41人）**：测试 AUC≈0.54——该数据集模拟跌倒标注噪声大、fall/adl 可分性弱（线性与深度方法均≈0.55），作为"真实老人数据获取/处理能力 + 跨域挑战"定位，不作为性能证据。

> 说明：UR Fall 仅 70 段（文献采用段级划分，无受试者标注），14 个测试样本下双模态与单流差异不显著；UP-Fall 按受试者划分、132 个未见样本，是主性能证据。跨库（UR Fall→老年）迁移 AUC=0.47，如实报告为开放挑战。

## 环境说明

当前 venv（D:\PyTorch\venv）为 **CPU 版 torch（2.14.0+cpu）**。CPU 可跑通全部流程，
但训练较慢；如有 NVIDIA 显卡，建议安装 CUDA 版：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

代码自动选择设备：`device="auto"` → 有 GPU 用 GPU，否则 CPU。

## 许可提醒

- YOLOv8（ultralytics）为 AGPL-3.0：竞赛/研究用途免费；商业化需企业许可。
- 三个数据集均为研究用途许可，申报材料需注明出处。

## 14 天计划对照

| 阶段 | 时间 | 对应文件 |
|---|---|---|
| 数据与基线 | 9/25-9/28 | data/*、train.py（先跑合成数据） |
| 核心模型 | 9/29-10/2 | models/fusion.py（核心）、models/cmta_net.py |
| 实验与材料 | 10/3-10/6 | evaluate.py、消融实验 |
| 验收上交 | 10/7-10/8 | 申报书/视频 |
