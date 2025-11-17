import torch


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k):
    with torch.inference_mode():
        max_k = max(top_k)
        batch_size = target.size(0)
        num_classes = output.size(1)
        
        # Adjust max_k if it exceeds number of classes
        max_k = min(max_k, num_classes)
        
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, prediction = output.topk(max_k, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target[None])

        res = []
        for k in top_k:
            # Use the minimum of k and num_classes
            k_adjusted = min(k, num_classes)
            correct_k = correct[:k_adjusted].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
