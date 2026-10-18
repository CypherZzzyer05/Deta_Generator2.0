# Deta Generator 2.0

新版独立数据生成器，层叠从下到上固定为 **SiO2 → Au → As2Se3 四聚体**。四聚体直接接触金膜。原项目 [Deta_Generator](https://github.com/CypherZzzyer05/Deta_Generator) 保留，旧版任务和数据不在本项目中续算或混合。

Python 包为 `deta-generator2`，运行模块为 `deta_generator2`，可与旧版 `deta-generator` / `deta_generator` 同环境共存。

## 结构和求解设置

默认 Au 厚 100 nm，SiO2 厚 350 nm；四聚体底面在 z=0。SiO2 范围 −450 至 −100 nm，Au 范围 −100 至 0 nm，四聚体范围 0 至 H。基板采用有限厚度，底部保留空气与 PML，便于与旧层叠作受控对照。

七个参数为 P、L、H、dX、dY、奇数象限角度、偶数象限角度。I/III 同角、II/IV 同角，各方柱绕自身中心旋转，正角逆时针。dX/dY 是未旋转时的名义边缘间隙。光谱范围 700–1700 nm，1000 个波长点。

默认并行任务只在金属附近设置 z 向 8 nm 网格，其余采用自动网格。原生布局检查会逐项核对金膜、基板、四聚体、金属网格和 R/T 监视器；网格审核检查新金膜位置的实际步长，要求低于 10 nm。T 监视器位于完整有限层叠下方的空气中。

## 安装

服务器需要 Windows、已安装且可用的 Lumerical FDTD 和许可、Python 3.10 或更新版本。生成数据只用 CPU，不需要 CUDA 或 PyTorch。

在已有 conda 环境或新环境安装：

```powershell
conda create -n fdtd_gen2 python=3.10 -y
conda activate fdtd_gen2
python -m pip install .\artifacts\deta_generator2-2.0.0-py3-none-any.whl
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
| 参数 | `parallel --job-dir jobs/topmetal_au100_metal8_1000_p10 --samples 1000 --jobs 10` |
| 工作目录 | 新版项目根目录 |
| 环境变量 | `LUMERICAL_LUMAPI_PATH=E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py;PYTHONUNBUFFERED=1` |

点击运行后依次准备新任务、原生预检查、启动最多 10 个独立样本。`--jobs 10` 代表同时计算 10 个不同结构；每个结构分配的线程默认是可用 CPU 预算除以 10。64 个可用逻辑处理器对应 10×6 线程，128 个对应 10×12。不要给每个任务都分配整台机器的全部线程。

调度器采用多组小分片，某个分片完成后补上下一个。初始少于 10 个原生引擎可能是加载和许可等待；运行窗口会区分 Python 工作进程和实际观察到的 FDTD 引擎。

复制此配置并在参数末尾加 `--parallel-action status`，即可只读查看状态。日志位于任务的 `parallel_logs`；汇总结果位于 `merged/dataset.npz`。任务中断后使用同一条启动命令续算。任务的结构/数量/分片布局固定；要改变这些内容请使用新目录。

## 先试一个样本

```powershell
python -m deta_generator2 init --job-dir jobs/topmetal_pilot --preset au-metal --metal-mesh-nm 8 --samples 1 --threads 8
python -m deta_generator2 preflight --config jobs/topmetal_pilot/job.json
python -m deta_generator2 generate --config jobs/topmetal_pilot/job.json --limit 1
```

并行入口默认 Au100 / metal8。串行 `generate` 每次只处理一个结构；多样本同时计算请使用 `parallel`。`doctor` 和 `preflight` 不执行时间推进求解。

## 数据与验证边界

新任务必须显式记录 `metal_position=above_substrate`。旧版缺少此字段的 job.json 会被拒绝，防止在旧数据目录里无意更换物理结构。元数据始终保留金层位置、金层厚度、材料数据哈希和实现哈希。合并时检查采样、物理配置、材料拟合和求解器版本的一致性。

已有材料模型沿用旧版：As2Se3 实测数据到 1688 nm，末端 12 nm 为显式常数延拓；Au Palik 宽带拟合误差尚未认证。数据保持 `pipeline_validation` 级别。自动关断成功说明达到当前衰减阈值，不等于完成材料拟合、网格或场增强的物理精度验证。

本机原生单样本测试结果与结构对照将整理到 `docs/local_stack_test/`。代码单元测试中的测试专用求解器只用于检验数据流程和并发控制，不代表真实 FDTD 结果。
