import torch
import torch.nn.functional as F


class SquareAlignmentBaseLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] > pred.shape[1]:
            target = target[:, : pred.shape[1], :]
        else:
            pred = pred[:, : target.shape[1], :]

        if self.normalize:
            target = F.normalize(target, dim=2)
            pred = F.normalize(pred, dim=2)

        alignment_pred = torch.bmm(pred, target.permute(0, 2, 1))
        alignment_target = torch.bmm(target, target.permute(0, 2, 1))

        return self.loss(alignment_pred, alignment_target)


class SquareAlignmentL1Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.normalize = False


class SquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = False


class NormalizedSquareAlignmentL1Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.normalize = True


class NormalizedSquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = True
