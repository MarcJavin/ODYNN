
import torch
import pickle
from tqdm import tqdm


# Loss functions

def loss_mse(y, target):
    return (y[:,:,:,-1] - target[:,:,:,-1]).pow(2).sum((0,1,2))


def corr(y):
    # compute correlation matrix
    vs = y[:, 0, 0, :] - torch.mean(y[:, 0, 0, :], dim=0)
    stdev = torch.sqrt(torch.sum(vs ** 2, dim=0))

    costd = torch.einsum('ijn,jkn->ikn', stdev[:, None], stdev[None, :])
    corr = torch.einsum('ijn,jkn->ikn', vs.permute(1, 0, 2), vs) / (costd + 1e-8)
    return corr

def loss_correlation(y, target):
    cor = corr(y)
    loss = (cor - target).pow(2).sum((0, 1))
    return loss





def optim(opt, params, input, loss_func, target, lr=0.1, plot_func=lambda:None, plot_period=20, init=None, n_epochs=500):
    losses = []
    # Define loss function
    learning_rate = lr
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for t in tqdm(range(501)):
        y = opt.calculate(input, init)

        loss = loss_func(y, target)

        losses.append(loss)
        # Upgrade variables
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        opt.apply_constraints()

        if t%plot_period == 0:
            plot_func(y, target, loss)
            if loss.min() <= losses[-1].min():
                with open('params', 'wb') as f:
                    p = {k: v.detach().numpy() for k,v in opt.parameters.items()}
                    pickle.dump(p, f)


    return opt, losses