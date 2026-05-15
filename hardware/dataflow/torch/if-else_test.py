import torch
import torch.nn as nn
import allo

class StaticIf(nn.Module):
    def __init__(self, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        self.relu = nn.ReLU()
    def forward(self, x):
        if self.use_relu:
            return self.relu(x)
        else:
            return x + 1

class Test_If(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

x = torch.randn(2,3)

print('=== StaticIf ===')
try:
    m = StaticIf(True).eval()
    mod = allo.frontend.from_pytorch(m, example_inputs=[x], verbose=False)
    y = mod(x.numpy())
    print('StaticIf converted OK, output shape:', y.shape)
except Exception as e:
    print('StaticIf failed:', type(e).__name__, e)

print('=== Test_If ===')
try:
    m = Test_If().eval()
    mod = allo.frontend.from_pytorch(m, example_inputs=[x], verbose=False)
    y = mod(x.numpy())
    print('Test_If converted OK, output shape:', y.shape)
except Exception as e:
    print('failed:', type(e).__name__)
    print(e)