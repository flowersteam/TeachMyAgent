from TeachMyAgent.teachers.utils.torch import get_gradient, zero_grad
import numpy as np
import torch


def _fisher_vector_product_t(p, kl_fun, param_fun, cg_damping):
    kl = kl_fun()

    grads = torch.autograd.grad(kl, param_fun(), create_graph=True, retain_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_v = torch.sum(flat_grad_kl * p)
    grads_v = torch.autograd.grad(kl_v, param_fun(), create_graph=False, retain_graph=True)
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_v]).data

    return flat_grad_grad_kl + p * cg_damping


def _fisher_vector_product(p, kl_fun, param_fun, cg_damping, use_cuda=False):
    p_tensor = torch.from_numpy(p)
    if use_cuda:
        p_tensor = p_tensor.cuda()

    return _fisher_vector_product_t(p_tensor, kl_fun, param_fun, cg_damping)


def _conjugate_gradient(b, kl_fun, param_fun, cg_damping, n_epochs_cg, cg_residual_tol, use_cuda=False):
    p = b.detach().cpu().numpy()
    r = b.detach().cpu().numpy()
    x = np.zeros_like(p)
    r2 = r.dot(r)

    for i in range(n_epochs_cg):
        z = _fisher_vector_product(p, kl_fun, param_fun, cg_damping, use_cuda=use_cuda).detach().cpu().numpy()
        v = r2 / p.dot(z)
        x += v * p
        r -= v * z
        r2_new = r.dot(r)
        mu = r2_new / r2
        p = r + mu * p

        r2 = r2_new
        if r2 < cg_residual_tol:
            break
    return x


def cg_step(loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter, cg_damping, n_epochs_cg,
            cg_residual_tol, n_epochs_line_search, use_cuda=False):
    zero_grad(param_fun())
    loss = loss_fun()
    prev_loss = loss.item()
    loss.backward(retain_graph=True)

    g = get_gradient(param_fun())
    if np.linalg.norm(g) < 1e-10:
        print("Gradient norm smaller than 1e-10, skipping gradient step!")
        return
    else:
        if torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)):
            raise RuntimeError("Nans and Infs in gradient")

        stepdir = _conjugate_gradient(g, kl_fun, param_fun, cg_damping, n_epochs_cg,
                                      cg_residual_tol, use_cuda=False)
        if np.any(np.isnan(stepdir)) or np.any(np.isinf(stepdir)):
            raise RuntimeError("Computation of conjugate gradient resulted in NaNs or Infs")

        _line_search(prev_loss, stepdir, loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter, cg_damping,
                     n_epochs_line_search, use_cuda=use_cuda)


def _line_search(prev_loss, stepdir, loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter,
                 cg_damping, n_epochs_line_search, use_cuda=False):
    # Compute optimal step size
    direction = _fisher_vector_product(stepdir, kl_fun, param_fun, cg_damping, use_cuda=use_cuda).detach().cpu().numpy()
    shs = .5 * stepdir.dot(direction)
    lm = np.sqrt(shs / max_kl)
    full_step = stepdir / lm
    stepsize = 1.

    # Save old policy parameters
    theta_old = weight_getter()

    # Perform Line search
    violation = True
    for _ in range(n_epochs_line_search):
        theta_new = theta_old + full_step * stepsize
        weight_setter(theta_new)

        new_loss = loss_fun()
        kl = kl_fun()
        improve = new_loss - prev_loss
        if kl <= max_kl * 1.5 or improve >= 0:
            violation = False
            break
        stepsize *= .5

    if violation:
        print("WARNING! KL-Divergence bound violation after linesearch")
        weight_setter(theta_old)
