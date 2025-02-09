
import torch

def softmax_cross_entropy_with_logits(y_true, y_pred):

	p = y_pred.clone()
	pi = y_true

	zero = torch.zeros_like(pi)
	where = pi == zero

	negatives = torch.full_like(pi, -100.0) 
	p = torch.where(where, negatives, p)

	loss = torch.nn.functional.cross_entropy(p, pi, reduction='none')

	return loss


