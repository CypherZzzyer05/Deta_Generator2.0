# 服务器打不开 GitHub 时安装

仓库与安装包为公开资源。网页加载失败时，可以直接下载源码 ZIP 或 wheel；若服务器对 GitHub 各域名都无法访问，在这台电脑下载后复制到服务器即可。

- [源码 ZIP](https://codeload.github.com/CypherZzzyer05/Deta_Generator2.0/zip/refs/heads/main)
- [独立安装包](https://raw.githubusercontent.com/CypherZzzyer05/Deta_Generator2.0/main/artifacts/deta_generator2-2.0.0-py3-none-any.whl)

离线压缩包包含代码、材料文件、wheel、文档和 PyCharm 共享配置，不包含 Python、conda、Lumerical 或其许可。

服务器已经有能运行旧版的 `fdtd_gen` 环境，可以直接复用。新版使用不同的包名和模块名，安装后旧版仍可使用。解压后进入有 `README.md`、`pyproject.toml` 和 `artifacts` 的目录，在 conda PowerShell 中运行：

```powershell
conda activate fdtd_gen
python -c "import numpy,psutil; print('dependencies OK')"
python -m pip install --no-index --no-deps .\artifacts\deta_generator2-2.0.0-py3-none-any.whl
python -m deta_generator2 --version
```

最后一条应输出 `2.0.0`。`--no-index --no-deps` 保证这一步不访问 PyPI；它要求当前环境已经有 numpy 和 psutil。若依赖检查失败，先联网安装这两个依赖，或准备对应服务器 Python/Windows 版本的离线依赖包。已能运行旧生成器的同一环境应已具备它们。

在 PyCharm 打开新版目录，将项目解释器选为该 `fdtd_gen` 环境。运行模块名为 `deta_generator2`。共享的服务器启动配置预填了 v251 API 路径；新任务使用 `jobs/topmetal_...`，不要指向旧任务目录。

页面错误本身不能区分 GitHub 页面服务异常、浏览器状态或服务器网络问题。上述直接下载与离线复制方式可绕开仓库网页，不要求在服务器安装 Git。
