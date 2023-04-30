'''
@Author: Kai Li
@Date: 2020-05-03 20:50:16
@LastEditors: Kai Li
@LastEditTime: 2020-07-19 20:49:22
@FilePath: /Conv-TasNet/test.py
'''
import torch

loss = torch.nn.MSELoss()

a = torch.tensor([[1,2,3],[4,5,6]]).type(torch.float32)
b = torch.tensor([[2,3,4],[5,6,7]]).type(torch.float32)


print(loss(a,b))