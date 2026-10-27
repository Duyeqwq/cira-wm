# CIRA-WM

完整 PointWorld 上的关系几何反馈与持续交互约束。

[项目展示](https://duyeqwq.github.io/cira-wm/) · [English](README.md) · [阶段记录](mark.md)

![项目展示卡](docs/project-card.png)

## 核心思路

按双臂来源和掌/指角色建立机器人到场景点的局部关系，
用逐点循环状态更新预测，并将模型自身预测的位置反馈给后续关系查询。
持续约束版本进一步携带交互参考、学习约束强度和释放。

保留完整 PointWorld 的编码与原位移/方差头，不是旧局部刚体预测器。
已知未来机器人轨迹仍是条件输入，不能把几何反馈称为已证明的因果识别。

## 这个精简仓库包含什么

- CIRA 反馈模块、共享关系工具、持续约束几何模块及其独立 CPU 测试。
- 简洁展示页、项目卡、两个结果摘要。
- 仅三个已有 MP4：长杆干预、非对称接触、抽屉转移。

完整源码、训练和闭环协议在私有 `D-Star-Robotics/CIRA-WM` 仓库，
需要访问权限。此精简仓库没有完整骨干、数据或权重，不是一键复现包。

## 已有证据

八任务、每组 23,068 窗口、两训练 seed，按 episode/task 宏平均：
CIRA 的动态/静态/全场景 EPE 为 10.145/0.726/0.911 mm；
持续约束版本为 10.028/0.918/1.096 mm。
持续约束动态略好，但静态和全场景更差，不宣称全面提升。

长杆闭环小试验只有三个独立初态：PointWorld、CIRA、持续约束 CIRA
分别为 4/6、4/6、6/6 次成功尝试。两 seed 复用初态，选模规则不同，
不能推导统计显著、广泛泛化或真机安全。

三个 MP4 都明确是仿真任务/干预预览，**不是 CIRA 预测对比或闭环成绩录像**。

安装 CPU PyTorch 后，可以独立运行约束几何检查：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m unittest discover \
  -s outputs/cira_constraint_research_20261021 -p test_transport.py
```

不启动仿真，不加载权重，不占用 GPU。完整实验复现仍需原始授权资产与环境。
上游归属和许可证见 `NOTICE`、`LICENSE`。
