# Deta Generator 2.0

**2.2.0 修正 Au 损耗拟合，并要求新建任务目录。** 2.1.1 及更早版本的默认 Au/Palik 拟合几乎丢失欧姆损耗；已有数据和旧权重保留为数值流程基线，不与新数据混合。

**网格核对（2026-10-22）：完整四聚体的金属局部 8/4 nm 对照未通过既定精度门槛。** 一个代表结构的全波段最大 R 差为 0.12182，第一谷半显著度宽度相差约 26%（以 8 nm 为分母），部分场强统计变化超过 10%。默认 8 nm 只用于流程验证，暂勿据此扩大用于器件结论的数据集；4 nm 还需更细网格核对。材料拟合通过与器件网格收敛分别判断。

本项目的层序从下到上为 **厚 SiO2 → Au100 nm → As2Se3 四聚体**，四聚体直接接触金膜，玻璃延伸穿过下方 PML。原项目 [Deta_Generator](https://github.com/CypherZzzyer05/Deta_Generator) 独立保留。

## 2.2.0 的修正

- Au 新任务使用 `Au (Gold) - JC passive NIR v1`，以 Johnson/Christy 数据拟合正强度、正阻尼的 Drude/Lorentz 模型，再导入原生 FDTD。数据来源明确更换，不把它描述为仅调整旧 Palik 参数。[原始论文](https://doi.org/10.1103/PhysRevB.6.4370)
- 每次实际求解前核对原生拟合。Au 的 n/k 最大绝对误差门槛分别为 0.02/0.05，介电常数实部/虚部最大相对误差门槛分别为 2%/5%；不得在源波段内出现增益。旧 Au/Palik 拟合若几乎消除损耗，会在启动引擎前报错。
- Au 默认使用 `conformal variant 1` 处理金属界面；只在金属附近设置 z 向 8 nm 网格，其余自动。预检查读取实际网格及 PML 厚度，确认 source、R、T 均位于 PML 外。
- 材料模型内容及实现哈希写入任务指纹。新旧任务不兼容，禁止修改旧 `job.json` 后继续混算。旧 wheel 留在 `artifacts/`，可在独立环境查看旧任务。

当前新 Au 模型覆盖 700–1700 nm。只用 numpy、psutil 和已安装的 Lumerical API，不需要在服务器安装 scipy、PyTorch 或 CUDA。Ag 仍使用原有 Palik 数据，本次没有完成 Ag 的新材料模型验证。

## 安装更新

先停止旧任务及其求解器并保留任务目录，再在 PyCharm 使用的 conda 环境中执行：

```powershell
python -m pip install --upgrade --no-deps .\artifacts\deta_generator2-2.2.0-py3-none-any.whl
python -m deta_generator2 --version
```

输出应为 `2.2.0`。如果只有旧项目目录，也可以直接安装已发布的 wheel：

```powershell
python -m pip install --upgrade --no-deps https://raw.githubusercontent.com/CypherZzzyer05/Deta_Generator2.0/main/artifacts/deta_generator2-2.2.0-py3-none-any.whl
```

新环境需要 Python 3.10+、numpy>=1.24,<3、psutil>=5.9；新安装可执行 `python -m pip install .` 自动安装依赖。`lumapi` 使用 Lumerical 随附文件，不从 PyPI 安装。

服务器 v251 的环境变量：

```powershell
$env:LUMERICAL_LUMAPI_PATH = 'E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py'
```

## PyCharm 同时计算 10 个样本

| 项目 | 设置 |
| --- | --- |
| 解释器 | 安装 2.2.0 的 conda 环境 |
| 运行方式 / 模块 | Module name / `deta_generator2` |
| 参数 | `parallel --job-dir jobs/jc220_topmetal_au100_metal8_1000_p10 --samples 1000 --jobs 10` |
| 工作目录 | 项目根目录 |
| 环境变量 | `LUMERICAL_LUMAPI_PATH=E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py;PYTHONUNBUFFERED=1` |
| PYTHONPATH | 关闭“添加内容根”和“添加源根”，避免旧 src 覆盖新安装包 |

启动器依次建立新任务、检查服务器实际材料与网格、派发计算。预检查失败就停止，不会先跑完一批再忽略错误。服务器必须自己通过材料检查，本机 v241 的成功不能替代服务器 v251 的实际结果。

`--jobs 10` 表示最多同时计算十个不同结构。线程默认按可用 CPU 预算分配：64 个可用逻辑处理器对应 10×6 线程。共享样本队列在每个样本完成后填补空位，保留 2.1.1 的尾段调度与 Windows 文件写入修复。

监控配置使用相同参数，末尾加 `--parallel-action status`。逐样本日志在 `tail_resume/logs`，最终结果在 `merged/dataset.npz`。同一 **2.2.0 新任务**中断后，可用原命令续算；以退出码 0、全部样本 success、pending=0 为完成依据。

## 先试一个样本

```powershell
python -m deta_generator2 init --job-dir jobs/jc220_pilot --preset au-metal --metal-mesh-nm 8 --samples 1 --threads 8
python -m deta_generator2 preflight --config jobs/jc220_pilot/job.json
python -m deta_generator2 generate --config jobs/jc220_pilot/job.json --limit 1
```

预检查应显示 `material_validation_passed: true`、金材料的 `material_fit_passed: true` 以及 `all_planes_outside_pml: true`。`doctor`、`preflight` 不进行时间推进求解。包内 `accuracy_certified: false` 表示未认证真实制备薄膜的实验精度，与上述数值拟合门槛分别记录。

## 结构、数据与验证范围

七参数为 P、L、H、dX、dY、奇数象限角和偶数象限角，I/III 同角、II/IV 同角，各柱绕自身中心旋转。长度单位 nm，角度 deg；dX/dY 表示未旋转时的名义边缘间隙。输出包含 700–1700 nm 的 1000 点 R/T/A 光谱。

默认 Au 为 z=−100 至 0 nm，四聚体从 z=0 开始。`substrate_depth_nm=350` 只表示计算域参考，不是物理玻璃厚度。新 Au 配置下边界 −2050 nm，玻璃对象延伸至 −3650 nm，T 监视面 −1250 nm；空气余量 3000 nm。T 是进入玻璃的归一化功率。

As2Se3 实测数据到 1688 nm，末端 12 nm 仍作显式常数延拓；数据保持 `pipeline_validation`。自动关断和材料门槛通过，不代表完整四聚体的 8/4 nm、热点或模态 Q 已完成精度认证。`bare-*` 是固定几何删除金膜的对照，不自动移动柱体或玻璃界面。

本机原生检查、谱线和测试记录见 `docs/release_2_2_0/`；报告集中发布在 [Report 仓库](https://github.com/CypherZzzyer05/Report)。并发测试中的模拟求解器仅验证程序流程；真实求解结果单独记录。原有 1.0/2.1 数据、训练结果和权重均保留。
