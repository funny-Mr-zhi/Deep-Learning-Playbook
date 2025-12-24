"""
Docstring for projects.HelloGithub.Deep-Learning-Playbook.src.env_test.pytorch_test

test pytorch
"""
import torch

print(torch.__version__)
print(f"GPU {torch.cuda.is_available()}")

