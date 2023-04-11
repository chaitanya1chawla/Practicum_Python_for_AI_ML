import torch
import numpy as np
import matplotlib.pyplot as plt

def polynomial(t:torch.Tensor) -> torch.Tensor:
    return -0.0533*t[:,0]**3 + 0.2*t[:,0]**2 + 10*t[:,0]

def breit_wigner(t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    a= params[0]
    b= params[1]
    c= params[2]
    return a/((b - (t[:,0])**2)+c)

def eval_jacobian_torch(x: torch.Tensor, func: callable, params: torch.Tensor)-> torch.Tensor:
    fixed_x = lambda p: func(x,p)
    return torch.autograd.functional.jacobian(fixed_x, params)

if __name__ == "__main__":
    print(polynomial(torch.tensor([[0.0],[1.0],[2.0],[3.0],[4.0]])))
    print(breit_wigner(torch.tensor([[0.0],[1.0],[2.0],[3.0],[4.0]]), torch.tensor([[0.5],[0.2],[1]])))
    px = torch.tensor(np.linspace(-20,20, num = 100).reshape(-1,1), dtype=torch.float, requires_grad=True)
    dpx1 = torch.autograd.grad(polynomial(px).sum(), px, create_graph=True)
    dpx2 = torch.autograd.grad(dpx1[0].sum(), px, create_graph = True)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(px.detach(), polynomial(px).detach(), label = "polynomial")
    ax.plot(px.detach(), dpx1[0].detach(), label = "first derivative")
    ax.plot(px.detach(), dpx2[0].detach(), label = "second derivative")
    ax.legend()
    fig.savefig("plot.pdf")

    bx = torch.tensor([[0],[1]])
    print(eval_jacobian_torch(bx, breit_wigner, torch.tensor([0.5, 0.2, 1])))