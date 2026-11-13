# photo_retouching 人像精修

纯 OpenCV 实现的人像精修流水线，不需要下载模型，CPU 上 1000×1500 的图约 3 秒。

## 用法

```bash
pip install -r requirements.txt

python main.py                          # 处理 data/raw/ 下所有图片，输出到 output/
python main.py photo.jpg                # 单张，输出 output/photo_retouched.jpg
python main.py photo.jpg -o out.jpg --compare --save-mask
python main.py data/raw --smooth 0.9 --whiten 0.4 --slim 0.5
```

## 处理步骤

| 顺序 | 步骤 | 参数（默认） | 实现 |
|---|---|---|---|
| 0 | 人脸检测 + 皮肤掩码 | — | Haar 人脸框；在脸中心采样 Cr/Cb 建自适应肤色模型，只保留与人脸相连的皮肤区域 |
| 1 | 祛痘祛痣 | `--blemish 1.0` | 皮肤内 blackhat 找暗色小圆点，`cv2.inpaint` 修补；眉眼、鼻孔、嘴的区域不处理 |
| 2 | 色彩校正 | `--color 0.3` | 灰度世界白平衡 + 全局色阶拉伸 |
| 3 | 高低频分离匀肤 | `--even 0.5` | 低频层做保边平滑（去色斑、明暗不均），高频纹理原样加回 |
| 4 | 磨皮 | `--smooth 0.7` | 双边滤波，保留 15% 细节防塑料感，按羽化掩码混合 |
| 5 | 美白 | `--whiten 0.2` | LAB 亮度通道对数曲线提亮 |
| 6 | 锐化 | `--sharpen 0.5` | USM，只作用于非皮肤区域（眼睛、眉毛、头发） |
| 7 | 瘦脸 | `--slim 0`（关闭） | 下颌两侧局部液化；位置由人脸框估计，没有关键点 |

强度都设为 0 即可关闭对应步骤。`--compare` 额外输出前后对比图，`--save-mask` 输出皮肤掩码，便于调参。

## 局限

- 人脸检测只认正脸；侧脸或没检出时退回固定 YCrCb 阈值，可能把木头、沙子等近肤色背景也磨掉。
- 祛痘只处理明显的深色小点，雀斑、细纹交给匀肤和磨皮弱化，不会完全去除。
- 瘦脸只靠人脸框估计下颌位置，强度建议不超过 0.6。要更准需要接入人脸关键点（如 mediapipe）。
