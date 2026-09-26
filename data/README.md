# 数据集下载指引与许可说明

主线数据集全部开源、直链下载，无需申请。**已在 9/25 完成下载**（详见各节状态标注）。

## 1. UP-Fall Detection（行为支路主数据：加速度/EEG/光照）✅ 已就位

- 内容：17 名受试者、11 类活动（5 种跌倒 + 6 种日常活动），每人 3 次尝试；含 5 个穿戴传感器
  （加速度/角速度/光照）、脑电、6 路红外。**注意：当前 Kaggle 镜像无视频**（行为支路视频由 UR Fall 补齐）。
- 已下载：`data/raw/upfall/CompleteDataSet (1).csv`（82.2MB，294,680 行）
- 下载源：https://www.kaggle.com/api/v1/datasets/download/pragyachandak/upfalldataset
- 增强版 3D 骨架（Zenodo，可选加分）：https://zenodo.org/records/12773013
- 许可：公开数据集，研究用途免费；申报材料注明出处（UP-Fall Detection Dataset, Martinez-Villaseñor et al.）

## 2. PPG-DaLiA（生理支路：PPG/加速度/ECG/呼吸）✅ 已就位（窗口化镜像）

- 内容：15 名受试者、每人约 2.5 小时日常活动；腕戴 Empatica E4（PPG 64Hz、加速度 32Hz）+
  胸戴 RespiBAN（ECG/呼吸 700Hz），含心率真值。
- 已下载（窗口化镜像 65MB → 解压 338MB）：`data/raw/ppg_dalia/dalia_window/combined_data.csv`
  （约 495 万条 8 秒窗口，wrist_BVP/心率回归标签/patient_id；当前镜像含 1 名受试者 S1）
- 下载源（窗口化，速度快）：https://www.kaggle.com/api/v1/datasets/download/aiforiot/ppg-dalia
  - 原始包约 3.1GB（15 受试者），⚠ UCI 官方直链 200KB/s、Kaggle 大文件限速 85KB/s，
    实测需 4-8 小时，已放弃（本作品以窗口化镜像作为 PPG→心率附加素材，不阻塞主线）。
- 放置目录：`data/raw/ppg_dalia/`
- 许可：研究用途免费，注明出处（Reiss et al., "Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks", Sensors 2019, 19(14), 3079, DOI: 10.3390/s19143079）

## 3. 老年腕戴数据（BITS Geriatric Fall Dataset，41 名真实老人）✅ 已就位

- 内容：41 名真实老年受试者腕戴三轴加速度（列=t,x,y,z,a）+ 1Hz 心率（hrt 行），每 user 16 段日常活动 + 8 段跌倒（每段 5 次尝试）。
- 已下载：`data/raw/zenodo_ger/`（986 个 CSV，adl.zip + fall.zip）
- 下载源：https://shamanx86.github.io/fall_detection_data/adl.zip 与 /fall.zip
  （Zenodo 页 403 时用此 GitHub 镜像，同样开源；官方 DOI: 10.5281/zenodo.10013090）
- 论文：Nandi et al., "Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device", Data in Brief 2023, 52, 109812, DOI: 10.1016/j.dib.2023.109812（CC BY 4.0）
- 用途：跨库泛化测试 + 真实老人数据（含心率）处理能力（申报加分项）。

## 4. UR Fall Detection（行为支路视频：YOLOv8-Pose 姿态提取）✅ 已就位

- 内容：70 个视频（30 跌倒 fall-01~30 双视角 + 40 日常活动 adl-01~40）+ 70 个对应加速度 CSV。
- 已下载：`data/raw/urfall/`（140 文件，93.9MB，全部校验无空文件）
- 下载源：https://fenix.ur.edu.pl/mkepski/ds/uf.html （页面 140 个直链并发下载）
- 许可：研究用途免费，注明出处（Kwolek & Kepski, 2014）

## 5. 跨库泛化测试素材（非训练集视频）✅ 已就位

- 内容：Kaggle unidpro/fall-detection（页面称 10,000 视频，实际压缩包含 2 个高清示例视频
  sample_1/sample_2.mp4，1080×1920@30fps，户外冬季场景跌倒）；**与训练集（UR Fall 室内）完全无关**，
  用作跨库泛化测试素材（照片序列形式）。
- 已抽帧：`outputs/demo_photos_cross/fall/`（8帧，弯腰跑动段）、`lying/`（8帧，趴地静态段）
- 下载源：https://www.kaggle.com/api/v1/datasets/download/unidpro/fall-detection （47.1MB）
- 实测（如实）：
  - **8 张照片抽帧**：户外跌倒段 0.687（低于室内校准阈值0.76漏检）——照片插值丢失运动速度信息；
  - **视频完整时序（150帧真实抽帧）**：sample_1 户外跌倒 **0.781 检出**、sample_2 趴地 **0.839 检出** →
    跨域视频在完整时序下可检测；照片模式漏检主因是"静态抽帧信息量不足"，已如实归因。
  - 结论：跨域仍存在分布差异（阈值0.76为室内校准），属"局限与未来工作：域自适应"素材；
    视频输入（真实时序）比照片序列更可靠，推荐演示用 `photo_demo.py --video`。
- 引用注意：unidpro 示例视频具体拍摄来源页面未注明，申报时标注"Kaggle Fall Detection 示例视频（unidpro）"即可。

## 备选（不阻塞主线，周期不可控）

## 目录约定

```
data/raw/upfall/         # 视频/传感器原始数据（Kaggle 解压后）
data/raw/ppg_dalia/      # 生理原始数据
data/raw/zenodo_ger/     # 老年腕戴数据
data/features/pose/      # extract_pose.py 输出：每样本一个 .npy (T,17,3)
data/features/physio/    # preprocess_physio.py 输出：每样本一个 .npy (C,T)
data/splits/             # split_by_subject.py 输出：train.csv / val.csv / test.csv
```

## 常见问题

- Kaggle 需要账号：注册后页面直接下载即可，不需要 API key。
- UP-Fall Kaggle 镜像的目录结构与官方略有差异：`extract_pose.py` 会递归遍历，
  只要视频在 `data/raw/upfall/` 下即可；`split_by_subject.py` 从路径正则提取
  subject 编号（支持 `subject1`、`subject_01` 等写法）。
- 数据集没下载完也能自测：`make_synthetic_physio.py` 生成合成生理特征，
  先用合成数据跑通 train.py → evaluate.py 全流程，再替换真实数据。
