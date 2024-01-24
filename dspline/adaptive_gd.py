import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import itertools


def _S(X):
    return X.detach().cpu().numpy()


class AdaptiveStepGradientDescent(torch.optim.Optimizer):
    """ 
    1. Compute loss and gradients using closure
    
    2. If less than previous loss, accept the current set of parameters. 
        Update gradients using computed gradients
    
    3. If more, reset parameters to values with lower loss.
        Ignore computed gradients and keep old ones

    4. Make a step with current gradients
    
    [Optimization: If 3, we don't have to call loss.backward()]
    """

    def __init__(self, params, initial_step=1e-4, step_multiplier=2, max_reject=12, max_step=1):
        self.stepsize = initial_step
        self.step_multiplier = step_multiplier
        defaults = dict(
            initial_step=initial_step,
            step_multiplier=step_multiplier,
            max_reject=max_reject,
            max_step=max_step)
        super(AdaptiveStepGradientDescent, self).__init__(params, defaults)

        self.last_loss = None
        self.last_params = None
        self.reject_count = 0
        self.max_reject = max_reject
        self.finished = False
        self.iteration = 0
        self.max_step = max_step

    def _params(self):
        for group in self.param_groups:
            for p in group['params']:
                yield p

    def finalize(self):
        """
        Assign the best parameter values
        """
        for param, (best_state, best_state_grad) in zip(self._params(), self.last_params):
            param.data[:] = best_state

            # print(best_state)

        self.finished = True

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError('This optimizer needs access to the loss')

        if self.reject_count == self.max_reject:
            raise ValueError('Optimization should be stopped once optimizer.finished is true')

        # step parameters with current gradient and step dist
        # call closure
        # if better, accept

        with torch.enable_grad():
            loss = closure()

        if self.last_loss is None or loss < self.last_loss:
            # accept the new state
            self.last_params = [(d.data.detach().clone(), d.grad.data.detach().clone()) for d in self._params()]
            # print(self.last_params[0][0].data)
            self.last_loss = loss

            for param in self._params():
                param.data -= param.grad.data * self.stepsize

            self.stepsize = min(self.stepsize * self.step_multiplier, self.max_step)
            self.reject_count = 0
            # print(f'{self.iteration}. accept: stepsize={self.stepsize}')

        else:
            for param, (best_state, best_state_grad) in zip(self._params(), self.last_params):
                param.data[:] = best_state - best_state_grad * self.stepsize

            self.stepsize /= self.step_multiplier
            # print(f'{self.iteration}. reject: stepsize={self.stepsize}')
            self.reject_count += 1

            if self.reject_count == self.max_reject:
                self.finalize()
                return loss

        # for p in self._params():
        #    print(p)

        self.iteration += 1

        return loss


def test_spline_lsq(optim=None):
    from splines import CatmullRomSpline1D, HermiteSpline1D

    # random data
    x = np.random.uniform(-1, 11, size=500)
    y = 10 + (x - 5) ** 2 + np.random.normal(0, 4, size=x.shape)

    fig = plt.figure()
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
    plt.scatter(x, y)
    plt.show()

    spline = HermiteSpline1D(np.zeros((12, 1)))
    # spline = CatmullRomSpline1D(np.zeros((6,1)))

    if optim is None:
        optimizer = torch.optim.SGD(spline.parameters(), lr=1, momentum=0)
    else:
        optimizer = optim(spline.parameters())

    x = Tensor(x)
    y = Tensor(y)

    epochs = 400
    line = None
    for i in range(epochs):

        def loss_():
            optimizer.zero_grad()
            output = spline(x * 0.5)[:, 0]

            loss = torch.mean((output - y) ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(loss_)

        xl = np.linspace(0, 10)
        result = _S(spline(xl * 0.5)[:, 0])

        if line is None:
            line = plt.plot(xl, result, 'k', label='result')[0]
            plt.legend()
        else:
            line.set_ydata(result)

        fig.gca().set_title(f"epoch {i} - loss={_S(loss):.1f} - stepsize={optimizer.stepsize:.1f}")
        fig.canvas.draw()
        fig.canvas.flush_events()

        if optimizer.finished:
            print('aborting')
            break

    xl = np.arange(0, 6)
    result = _S(spline(xl)[:, 0])
    plt.plot(xl * 2, result, 'or')
    fig.canvas.draw()

    return spline


def test_custom_optim():
    s = test_spline_lsq(lambda params: AdaptiveStepGradientDescent(params, 1, 1.1, 20, max_step=10))


if __name__ == '__main__':
    test_custom_optim()
