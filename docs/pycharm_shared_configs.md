# PyCharm 共享配置

打开这个新版项目后，运行菜单应出现两个共享配置：

- `Server TopMetal 1000 samples 10 jobs`：新任务目录、1000 个样本、最多 10 个同时计算。
- `Server TopMetal monitor`：只读查看同一个任务的完成数量、活动进程和求解日志进度，不启动计算。

先把项目解释器设为已安装 `deta-generator2` 的 conda 环境。两份配置使用项目解释器；不会自动创建环境或替换旧版包。启动配置中预填了服务器已提供的 v251 API 路径 `E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py`，若安装位置不同，在运行配置的环境变量中修改。

这份 10 并发配置针对服务器，不用于低内存本机测试。本机单样本测试使用独立脚本和一台原生求解器。生成器没有 CUDA 依赖。

运行前建议先按主 README 的单样本命令确认许可、模型和资源。正式启动时运行窗口会先执行原生布局预检查，再启动工作进程。`--jobs` 是同时计算的不同结构数，`--threads` 是每个结构的线程数，省略后自动按可用 CPU 预算分配。

若修改样本数或任务路径，也请同步修改监视配置。2.1.1 两份配置均使用 `jobs/pml_topmetal_au100_metal8_1000_p10` 目录；兼容同一物理实现的 2.1.0 任务。2.0.0 有限厚基板数据不能在此续算。请确认 `job.json` 同时包含 `metal_position=above_substrate`、`substrate_termination=pml`。

2.1.1 默认是逐样本动态队列；监视配置读取约每 30 秒更新一次的控制器快照并打印时间。单个样本的原生详细日志位于任务目录下 `tail_resume/sample_XXXXXX/projects`，Python 输出位于 `tail_resume/logs`。原分片任务的旧日志仍保留在原目录。
