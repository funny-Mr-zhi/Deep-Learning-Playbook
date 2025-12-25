# 🌸 Deep-Learning-Playbook

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/your-username/your-repo-name/graphs/commit-activity)

> **重读经典，从核心出发。** > 本项目致力于以更直观、更具实践性的方式重构《深度学习》（花书）。从深度前馈网络（第6章）切入，提供 **"理论总结文档 + 演示PPT + 代码复现"** 的全栈学习方案。

## 💡 项目愿景 (Vision)

《深度学习》被称为 AI 领域的圣经，但其数学密度往往让人望而却步。本仓库旨在通过以下方式降低学习门槛，并增强理论与工程的连接：

* **📘 理论重述**: 拒绝简单的原文摘抄，用更通俗的语言配合 LaTeX 公式重写核心逻辑。
* **🎨 视觉化呈现**: 为每一章提供总结性 PPT，关键概念辅以自绘架构图。
* **💻 双层代码实践**:
    * `From Scratch`: 使用 `Numpy` 手写核心算法（如反向传播），彻底理解黑箱内部。
    * `Framework`: 使用 `PyTorch` 进行现代化的工程实现，对接工业界标准。

## 🗺️ 学习路线与进度 (Roadmap)

我们采用“最小可行性”原则，优先攻克深度学习的核心腹地。

| 章节 | 章节名称 | 理论笔记 (Doc) | 演示幻灯片 (Slides) | 代码实践 (Code) | 状态 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **06** | **深度前馈网络** | [📝 Read](./chapters/Ch06_deep_feedforward_network/README.md) | [📊 PPT coming soon]() | [Coming Soon]() | 🚧 进行中 |
| **07** | **深度学习中的正则化** | - | - | - | ⏳ 待定 |
| **08** | **深度模型中的优化** | - | - | - | ⏳ 待定 |
| **09** | **卷积网络** | - | - | - | ⏳ 待定 |
| **10** | **序列建模: 循环和递归网络** | - | - | - | ⏳ 待定 |

## 🛠️ 环境配置 (Getting Started)

为了确保代码的可复现性，建议使用 `conda` 创建独立环境。

```bash
# 1. 克隆仓库
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# 2. 创建虚拟环境
conda create -n dl-flower python=3.10
conda activate dl-flower

# 3. 安装依赖
pip install -r requirements.txt
