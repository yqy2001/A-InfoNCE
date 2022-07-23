import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def trades_loss(model,
                classifier,
                x_natural,
                x_adv,
                y,
                optimizer,
                beta=6.0,
                distance='l_inf',
                step_size=2./255.,
                epsilon=8./255.,
                perturb_steps=10,
                trainmode='adv',
                fixmode='',
                trades=True
        ):
    if trainmode == "adv":
        batch_size = len(x_natural)
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()

        if trades:
            # generate adversarial example
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            if distance == 'l_inf':
                for _ in range(perturb_steps):
                    x_adv.requires_grad_()
                    with torch.enable_grad():
                        model.eval()
                        loss_kl = criterion_kl(F.log_softmax(classifier(model(x_adv)), dim=1),
                                                F.softmax(classifier(model(x_natural)), dim=1))
                    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
            elif distance == 'l_2':
                assert False
            else:
                assert False

            x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    model.train()

    if fixmode == 'f1':
        for name, param in model.named_parameters():
                param.requires_grad = True
    elif fixmode == 'f2':
        # fix previous three layers
        for name, param in model.named_parameters():
            if not ("layer4" in name or "fc" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, param in model.named_parameters():
            if not ("fc" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        assert False

    logits = classifier(model(x_natural))
   
    loss = F.cross_entropy(logits, y)

    if trainmode == "adv":
        logits_adv = classifier(model(x_adv))
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits, dim=1))
        loss += beta * loss_robust
    return loss, logits