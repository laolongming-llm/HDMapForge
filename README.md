# HDMapForge

点云强度图道路要素提取：本地深度学习负责几何，外部视觉语言模型（VLM）按同质路段赋予京东标准类型。

| 结果 | 几何与属性 |
| --- | --- |
| `BOUNDARY.shp` | 连续线；在类型变化处分段；整数 `TYPE` 使用京东标准 |
| `DIRECTION.shp` | 每个箭头的最小旋转外包矩形面，无箭头类型字段 |
| `CROSSWALK.shp` | 每个斑马线整体区域的外包面，保留 L 形凹部，无类型分类 |
| `STOPLINE.shp` | 停止线线要素，无类型分类 |

当前是可运行的**几何训练与区域推理基线**。已用合成数据验证CPU训练、checkpoint回载和离线API协议；没有真实数据训练的权重或工程精度结论。完整大范围的实例分离、跨源图拼接和生产验收仍需实现与实测。旧版思路和预处理/审查代码不属于此仓库。

## 运行

Python 3.12，依赖通过uv锁定。默认训练extra安装CPU版PyTorch，用于可复现的功能验证。

```powershell
uv sync --locked --extra train
uv run --extra train pytest -q
uv run --extra train hdmapforge train --config configs/train.yaml
uv run --extra train hdmapforge predict --raster path/to/intensity.tif --checkpoint runs/geometry/checkpoints/best.ckpt --window 1000 1000 2048 2048 --output outputs/probabilities.tif
uv run --extra train hdmapforge export --probabilities outputs/probabilities.tif --raster path/to/intensity.tif --crs EPSG:YOUR_VERIFIED_CODE --offset 0 0 --output outputs/vectors
```

省略 `--vlm-config` 时，车道线输出 `TYPE=0, REVIEW=1`。启用已配置的外部视觉服务：

```powershell
$env:HDMAPFORGE_API_KEY = 'YOUR_KEY'
uv run --extra train hdmapforge export --probabilities outputs/probabilities.tif --raster path/to/intensity.tif --crs EPSG:YOUR_VERIFIED_CODE --offset 0 0 --output outputs/typed_vectors --vlm-config configs/vlm.example.yaml
```

先将示例的endpoint、固定模型版本和预算改为实际配置。客户端使用兼容OpenAI Chat Completions的图像/JSON协议，支持此协议的Qwen部署等服务可接入；其他协议需要独立适配。API key占位符不应写进Git。客户端有请求次数上限、重试、缓存与token计数，默认不会创建微调任务。

GPU训练需按驱动和服务器配置选择经过验证的PyTorch/CUDA组合。项目使用命名索引 `pytorch`；可按 [uv官方PyTorch说明](https://docs.astral.sh/uv/guides/integration/pytorch/)覆盖索引并为服务器重新锁定环境。本机CPU测试通过不代表CUDA环境已经验证。若本机环境变量 `PROJ_LIB/PROJ_DATA` 指向旧PostGIS，应在独立进程中使用本环境自带PROJ数据，避免错误CRS库。

## 必要数据契约

本仓库的训练入口消费**已离线完成配准、标注和空间分组的样本**，不提供原始数据检查或预处理脚本。

manifest为JSON列表或 `{"samples": [...]}`，每条包含 `path`、`split`（train/val/test）、`group_id`。NPZ路径相对manifest，字段为：

- `image`：HW或1HW、float、范围[0,1]；当前推理按uint8/255归一化，离线训练须保持一致。
- `targets`：4HW、[0,1]，固定顺序 `lane, arrow, crosswalk, stopline`。
- `valid`：4HW、二元掩膜；未标注、无覆盖或标签位置不确定带设0。

车道与停止线目标是连续制图几何；箭头目标为外包矩形区域；斑马线为整体面。虚线空隙可以属于连续几何监督，不能把它冒充实际油漆监督。`valid`掩膜用于减少错误监督，**不能校正原始标签偏移**。系统偏移与局部偏移需在线下用有约束配准处理，并保留配准前后的人工依据。

训练只输出四类几何概率，不含车道或箭头类型分类头。相同物理路段、原线族和重叠采集区必须同一空间组；loader拒绝group跨split。当前是轻量Residual U-Net + masked BCE/Dice、Lightning、AdamW、best/last checkpoint与CSV日志。没有精细油漆真值时不启用paint辅助头。

## 车道线类型与过渡分段

每条候选线按里程采样，将原始强度条带和独立定位参考图一起交给VLM。原图不画满连续线；上下文保留多段虚线、双线和相邻结构。VLM只判定中央决策区间，返回京东 `TYPE` 或 `mixed/unknown`。类型冲突与未知区间加密取样，最后沿原有几何切分，同质子段保留连续线。

默认采样步距2m、边界加密目标0.25m；这是算法参数，**不是已达到的定位精度**。短于步距的过渡可能漏掉；工程版应结合完整滚动证据检测，验证最短类型段与转折点误差后决定采样密度。当前不会用多数票抹掉已经观测到的短段。VLM自报score不视作校准概率；未知、方向约定不清、调用失败和未收敛边界保留复核状态。

客户确认的原始粗码映射：`1→京东{1,2}`，`2→{3,4}`，`3→7`，`4→8`。该对应关系保存在`line_types.py`。粗码不能自动生成单双线细分类的微调真值，微调需使用有证据支持的逐段细分类标签和过渡点。左虚右实/左实右虚的左右约定须与交付方向一致，未确认时拒绝赋5/6。

当前示例配置通过 `provider.enabled_types` 暂停双虚线（JD2），其余编码保留，包括少样本双实线（JD4）；完整京东字典仍保留。请求中的允许类型取此配置与原始粗码对应类型的交集。模型返回未开放的合法类型时归为 `unknown/TYPE=0`，不会强行映射成单虚线。省略该配置项时兼容完整字典。少样本可用于先行学习，实际泛化能力需通过独立路段验证。

外部模型先在独立空间测试集上比较零/少样本基线，再决定视觉SFT/LoRA。Qwen的云视觉SFT或官方开源LoRA是候选；实际可用性和账户地区需确认。OpenAI的[视觉微调说明](https://developers.openai.com/api/docs/guides/vision-fine-tuning)目前说明平台收缩、不向新用户开放，不能预设购买token即获得微调权限。训练与评估图片不得以同一路段或相邻窗口泄漏；人工金标准保留在私有数据系统。

## 推理与交付范围

推理采用磁盘支持的重叠概率融合，保留原始affine，不整图缩放。当前矢量化限单个ROI最多1600万像素，建议先使用2048像素区域验证。箭头/斑马线采用连通区域基线，邻接实例可能合并；线交叉节点和图边缘实例进入复核。**当前没有完整跨区块实例拼接**，不应直接把各ROI的SHP拼成生产交付。

SHP导出需要经确认的真实投影米制CRS与局部坐标平移量，不能信任与数值范围矛盾的PRJ。车道含整数TYPE，其他图层没有TYPE/TURN_TYPE；全部带REVIEW。未验证模型默认 `model_validated=false`，输出仅供复核；`REVIEW=1`不是验收通过。记录包含沿线观测、错误、请求数及已报告token用量，便于按道路公里测算成本。

实现入口：`model.py/training.py/data.py/losses.py` 训练，`inference.py/pipeline.py` 推理，`crops.py/vlm.py/segmentation.py` 外部类型与分段，`vectorize.py/shapefiles.py` 矢量和SHP输出。测试只覆盖必要运行行为；真实数据、配准工具、审查脚本、研究文档和技能安装记录均不进入公开仓库。

训练工程参考了 [Scientific Agent Skills](https://arxiv.org/abs/2609.00065) 的PyTorch Lightning流程；微调设计采用vision-sft与eval-harness-first技能的图文配对、分辨率预算及独立基线原则，超参数仍由官方模型文档和实测决定。
