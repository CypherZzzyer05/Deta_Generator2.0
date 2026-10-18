# As₂Se₃ 光学常数

`Joseph-700nm.yml` 为 refractiveindex.info 的 CC0 数据，原始测量对象为玻璃上的 700 nm 厚薄膜。`as2se3_joseph_700nm.csv` 仅转换单位（μm→nm），保留实测 n 和 k，不设 k=0。

- 来源：https://refractiveindex.info/?shelf=main&book=As2Se3&page=Joseph-700nm
- 原论文：https://doi.org/10.1021/acsami.0c12525
- 数据库原文件：https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/As2Se3/nk/Joseph-700nm.yml
- 实测范围：245.8–1688 nm。
- 700–1700 nm 探索配置显式允许最多 12 nm 的端点常数延拓；1688–1700 nm 不是实测数据。最终结构若依赖此区间，须补充材料测量或更换覆盖完整波段的数据。
- 薄膜厚度、制备工艺可能改变 n、k，当前数据仅用于设计探索。

`Joseph-400nm.yml` 保留为材料工艺敏感性对照，默认不混用。

FDTD 将 `(n+ik)^2` 转为复介电常数，以频率和复介电常数两列导入 Sampled data；方法见 https://optics.ansys.com/hc/en-us/articles/360034409654-setmaterial-Script-command 。材料内容的 SHA256 写入配置指纹，修改 CSV 会使旧结果无法被复用。
