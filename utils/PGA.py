import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PGA(torch.nn.Module):
    def __init__(self, Wgt, device, nitm=1500, prec=1e-4):
        super(PGA, self).__init__()
        self.W = torch.tensor(Wgt, dtype=torch.float32, device=device)
        self.max_iters = nitm
        self.tol = prec

    def forward(self, V, H, Lambda):
        V = torch.tensor(V, dtype=torch.float32, device=device)
        H = torch.tensor(H, dtype=torch.float32, device=device)
        step_size = 1.995 / (torch.norm(self.W) ** 2 + Lambda)
        for nit in range(self.max_iters):
            Hold = H.clone()
            H_grad = self.W.t() @ (self.W @ H - V) + Lambda * H
            H = (H - step_size * H_grad).clamp_(min=1e-8)
            if nit > 0 and torch.norm(H - Hold) < torch.norm(Hold) * self.tol:
                break
        return H.detach().cpu().numpy()
