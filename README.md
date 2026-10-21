# Deta Generator 2.0

**材料核对（2026-10-21）：2.1.1 默认 Au/Palik 的原生拟合几乎丢失欧姆损耗。当前数据可用于数值流程验证，暂不建议据此扩大用于真实器件结论的数据集。** [完整证据与修正候选](https://github.com/CypherZzzyer05/Report/blob/main/reports/2026-10-21_gold_film_validation/README.md)。研究候选尚未达到全部材料误差门槛，生产物理配置未修改，未发布 2.2.0。

新版独立数据生成器，层叠从下到上固定为 **SiO2 → Au → As2Se3 四聚体**。四聚体直接接触金膜。原项目 [Deta_Generator](https://github.com/CypherZzzyer05/Deta_Generator) 保留，旧版任务和数据不在本项目中续算或混合。

Python 包为 `deta-generator2`，运行模块为 `deta_generator2`，可与旧版 `deta-generator` / `deta_generator` 同环境共存。

## 结构和求解设置

2.1.1 默认 Au 厚 100 nm，底层为延伸穿过下方 PML 的厚 SiO2 基板。四聚体底面在 z=0，Au 范围 −100 至 0 nm，SiO2 顶面在 −100 nm；层间直接接触，基板下方没有空气背界面。上方仍为空气入射区。

`substrate_depth_nm=350` 在 `substrate_termination=pml` 模式中只是计算域深度参考，不是物理玻璃厚度。默认下边界为 −1650 nm，玻璃对象延伸至 −2850 nm，透射面位于 −1050 nm 的均匀 SiO2 内。改动计算域深度后这些坐标随配置调整。

七个参数为 P、L、H、dX、dY、奇数象限角度、偶数象限角度。I/III 同角、II/IV 同角，各方柱绕自身中心旋转，正角逆时针。dX/dY 是未旋转时的名义边缘间隙。光谱范围 700–1700 nm，1000 个波长点。

默认并行任务只在金属附近设置 z 向 8 nm 网格，其余采用自动网格。原生布局检查会逐项核对金膜、基板、四聚体、金属网格和 R/T 监视器，确认玻璃覆盖下方 PML 和透射面；网格审核检查新金膜位置的实际步长，要求低于 10 nm。T 为进入玻璃的功率透射率，取原生监视器通量并按入射源功率归一化，不是电场振幅。

## 安装

服务器需要 Windows、已安装且可用的 Lumerical FDTD 和许可、Python 3.10 或更新版本。生成数据只用 CPU，不需要 CUDA 或 PyTorch。

在已有 conda 环境或新环境安装：

```powershell
conda create -n fdtd_gen2 python=3.10 -y
conda activate fdtd_gen2
python -m pip install --upgrade .\artifacts\deta_generator2-2.1.1-py3-none-any.whl
python -m deta_generator2 --version
```

也可在项目根目录执行 `python -m pip install .`。包依赖仅为 numpy 和 psutil。不要把 `lumapi` 当作 PyPI 包安装，应使用已安装 Lumerical 自带的 API。

服务器 v251 API 环境变量示例：

```powershell
$env:LUMERICAL_LUMAPI_PATH = 'E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py'
```

## PyCharm 同时计算 10 个样本

Run/Debug Configuration 新建 Python 配置：

| 项目 | 设置 |
| --- | --- |
| Python 解释器 | 安装新版包的 conda 环境 |
| 运行方式 | Module name |
| 模块名 | `deta_generator2` |
| 参数 | `parallel --job-dir jobs/pml_topmetal_au100_metal8_1000_p10 --samples 1000 --jobs 10` |
| 工作目录 | 新版项目根目录 |
| 环境变量 | `LUMERICAL_LUMAPI_PATH=E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py;PYTHONUNBUFFERED=1` |

点击运行后依次准备新任务、原生预检查、启动最多 10 个独立样本。`--jobs 10` 代表同时计算 10 个不同结构；每个结构分配的线程默认是可用 CPU 预算除以 10。64 个可用逻辑处理器对应 10×6 线程，128 个对应 10×12。不要给每个任务都分配整台机器的全部线程。

2.1.1 改为共享样本队列。一个样本完成并校验后，空位立即接下一个样本，不再把剩余样本固定在少数分片中。当未完成的样本总数少于 10 时，实际并发数自然下降；启动阶段的加载或许可等待也会影响实际引擎数。运行窗口区分 Python 工作进程和实际观察到的 FDTD 引擎。

复制此配置并在参数末尾加 `--parallel-action status`，即可只读查看最新状态快照及其时间。逐样本日志位于 `tail_resume/logs`，汇总结果仍位于 `merged/dataset.npz`。任务中断后使用同一条启动命令续算；已成功保存的样本经校验后复用。旧版 `shard*` 目录也可在同一任务中续算，原采样计划、标签和划分保持不变。任务的结构/数量/原分片布局固定；并发槽数 `--jobs` 可以调整，不受原分片数量限制。

每个样本使用独立的 Python 工作进程和 FDTD API 会话，存在启动开销。并行控制测试使用模拟求解器验证调度，尚未在服务器上测量新版的真实 FDTD 吞吐量。原有 `--pack-every` 参数为兼容命令保留；单样本工作目录总是在样本完成后保存，最终统一合并。

## 从 2.1.0 更新

先让当前任务完成，或停止该任务的控制器及其求解器，再更新解释器中的包。不要在旧控制器运行期间替换包。新任务直接使用上面的 PyCharm 配置；续算兼容的 2.1.0 任务时，保持原任务目录和样本总数。

```powershell
python -m pip install --upgrade .\artifacts\deta_generator2-2.1.1-py3-none-any.whl
python -m deta_generator2 --version
```

无需单独运行尾段加速脚本或合并修复脚本，正常 `parallel` 入口已包含这两项修复。最终合并对 Windows 短暂的文件占用（错误 5/32/33）进行有限重试，覆盖 manifest、样本复制和 dataset.npz 写入；持续权限错误会明确报错并保留成功样本。单次写入最多尝试 101 次，重试间隔 0.1 秒。

中断任务可能留有部分汇总文件。以进程退出码 0、`success=样本总数` 且 `pending=0` 为完成依据；文件存在本身不表示整批完成。

2.1.1 只修改调度和文件写入。`config.py`、`lumerical.py`、`pipeline.py` 与 2.1.0 完全相同，物理实现标识仍为 `deta-generator2/2.1.0`，用于保持旧任务指纹一致；安装包版本由 `--version` 查看，为 2.1.1。

## 先试一个样本

```powershell
python -m deta_generator2 init --job-dir jobs/pml_topmetal_pilot --preset au-metal --metal-mesh-nm 8 --samples 1 --threads 8
python -m deta_generator2 preflight --config jobs/pml_topmetal_pilot/job.json
python -m deta_generator2 generate --config jobs/pml_topmetal_pilot/job.json --limit 1
```

并行入口默认 Au100 / metal8。串行 `generate` 每次只处理一个结构；多样本同时计算请使用 `parallel`。`doctor` 和 `preflight` 不执行时间推进求解。

## 数据与验证边界

新任务必须显式记录 `metal_position=above_substrate` 和 `substrate_termination=pml`。2.0.0 的有限厚基板任务缺少后者，会被拒绝；请新建任务目录，不能把旧数据改标签后续算。独立对照可明确使用 `substrate_termination=finite`，并行默认启动器只接受 `pml`。元数据始终保留层序、基板终止方式、金层厚度、材料数据哈希和实现哈希。合并时检查物理配置、材料拟合和求解器版本的一致性。

已有材料模型沿用旧版：As2Se3 实测数据到 1688 nm，末端 12 nm 为显式常数延拓；Au Palik 宽带拟合误差尚未认证。数据保持 `pipeline_validation` 级别。自动关断成功说明达到当前衰减阈值，不等于完成材料拟合、网格或场增强的物理精度验证。

本机测试报告发布在 [Report 仓库](https://github.com/CypherZzzyer05/Report)。代码单元测试中的测试专用求解器只用于检验数据流程和并发控制，不代表真实 FDTD 结果。光谱窄峰的 Q 值需要足够时间衰减、频率分辨率及合理线型拟合，不能把反射谷的半突出度宽度直接当作模式 FWHM。
