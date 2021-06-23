import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import time
import pickle


def make_step(loss, o, s, params):
    o.zero_grad()
    loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(params, 2.0)
    o.step()
    if s is not None:
        s.step()


def find_acc(true_probs,
             true_labs,
             false_probs=None,
             false_labs=None,
             threshold=0.5):
    if threshold < 0:
        assert len(true_probs) == len(true_labs)
        preds = true_probs.argmax(dim=1)
        return (torch.sum(preds == true_labs) / len(true_labs)).item()

    if true_labs.shape != true_probs.shape:
        assert len(true_labs) == true_probs.size(0)
        labs = torch.zeros_like(true_probs)
        for i in range(true_probs.size(0)):
            labs[i][true_labs[i]] = 1
        true_labs = labs
    true_acc = ((true_probs > threshold) * 1) == true_labs
    true_acc = torch.sum(true_acc) / true_acc.nelement()

    false_acc = None
    if false_probs != None:
        if false_labs.shape != false_probs.shape:
            assert len(false_labs) == false_probs.size(0)
            labs = torch.zeros_like(false_probs)
            for i in range(false_probs.size(0)):
                labs[i][false_labs[i]] = 1
            false_labs = labs
            print(false_labs)
        false_acc = ((false_probs > threshold) * 1) == false_labs
        false_acc = torch.sum(false_acc) / false_acc.nelement()

        return true_acc.item(), false_acc.item()

    return true_acc.item()


eps = 1e-5


def evaluate(model, val_dl, device, pretraining=False):
    model.eval()

    val_metrics = {
        'reconst': [],
        'disc_y': [],
        'disc_z': [],
        'enc': [],
        'y_': [],
        'z_d_acc': [],
        'y_d_acc': [],
        'y_acc': []
    }

    for _, (x_unlab, xs, ys) in enumerate(val_dl):
        x_unlab = x_unlab.to(device)
        xs = xs.to(device)
        ys = ys.to(device)

        with torch.no_grad():
            x_, z, y_ = model(x_unlab)

        # reconst phase
        divisor = torch.tensor(x_unlab.size()) / x_unlab.size(0)
        divisor = divisor.prod() * x_unlab.size(0)
        reconst_loss = F.mse_loss(x_, x_unlab, reduction='mean') / divisor
        val_metrics['reconst'].append(reconst_loss.item())

        if not pretraining:
            # regularisation (discrim)
            z_sample, y_sample = (s.to(device)
                                  for s in model.get_samples(xs.size(0)))

            with torch.no_grad():
                p_real_z = model.discriminate(z_sample, 'z')
                p_fake_z = model.discriminate(z, 'z')
                p_real_y = model.discriminate(y_sample, 'y')
                p_fake_y = model.discriminate(y_, 'y')

            d_loss_z = torch.mean(-torch.log(eps + p_real_z) -
                                  torch.log(eps + 1 - p_fake_z))
            d_loss_y = torch.mean(-torch.log(eps + p_real_y) -
                                  torch.log(eps + 1 - p_fake_y))
            val_metrics['disc_z'].append(d_loss_z.item())
            val_metrics['disc_y'].append(d_loss_y.item())
            val_metrics['z_d_acc'].append(
                find_acc(p_real_z, torch.ones_like(p_real_z), p_fake_z,
                         torch.zeros_like(p_fake_z)))
            val_metrics['y_d_acc'].append(
                find_acc(p_real_y, torch.ones_like(p_real_y), p_fake_y,
                         torch.zeros_like(p_fake_y)))

            # regularisation (generator/encoder)
            g_loss = torch.mean(-torch.log(eps + p_fake_z) -
                                torch.log(eps + p_fake_y))
            val_metrics['enc'].append(g_loss.item())

        # semi-supervised
        with torch.no_grad():
            _, y_ = model.get_latents(xs)
        y_loss = F.cross_entropy(y_, ys) * (9 * pretraining + 1)
        val_metrics['y_'].append(y_loss.item())
        val_metrics['y_acc'].append(find_acc(y_, ys, threshold=0.5))

    return val_metrics


def train(model,
          train_dl,
          num_epochs,
          device,
          optim_sched,
          verbose_freq=50,
          save=True,
          cache_dir=None,
          val_dl=None,
          start_epoch=1,
          pretraining=False,
          semisup=True,
          model_name='AAE'):
    val_flag = (val_dl != None)
    model.to(device)

    batch_history = {
        'reconst': [],
        'disc_y': [],
        'disc_z': [],
        'enc': [],
        'y_': [],
        'z_d_acc': [],
        'y_d_acc': [],
        'y_acc': []
    }
    epoch_history = {
        'train': {k: []
                  for k in batch_history},
        'val': {k: []
                for k in batch_history}
    }

    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_time = time.time()

        model.train()

        for step, (x_unlab, xs, ys) in enumerate(train_dl):
            model.zero_grad()
            x_unlab = x_unlab.to(device)
            xs = xs.to(device)
            ys = ys.to(device)

            # reconst phase
            x_ = model.get_reconst(x_unlab)
            divisor = torch.tensor(x_unlab.size()) / x_unlab.size(0)
            divisor = divisor.prod() * x_unlab.size(0)
            reconst_loss = F.mse_loss(x_, x_unlab, reduction='mean') / divisor
            batch_history['reconst'].append(reconst_loss.item())
            make_step(reconst_loss, *optim_sched['reconst'],
                      model.parameters())

            if not pretraining:
                # regularisation (discrim)
                model.enc.eval(
                )  # since information shouldnt be restricted through encoder at this stage
                z_sample, y_sample = (s.to(device)
                                      for s in model.get_samples(xs.size(0)))
                z, y_ = model.get_latents(x_unlab)

                p_real_z = model.discriminate(z_sample, 'z')
                p_fake_z = model.discriminate(z, 'z')
                p_real_y = model.discriminate(y_sample,
                                              'y')  # TODO: try with actual y
                p_fake_y = model.discriminate(y_, 'y')

                d_loss_z = torch.mean(-torch.log(eps + p_real_z) -
                                      torch.log(eps + 1 - p_fake_z))
                d_loss_y = torch.mean(-torch.log(eps + p_real_y) -
                                      torch.log(eps + 1 - p_fake_y))
                batch_history['disc_z'].append(d_loss_z.item())
                batch_history['disc_y'].append(d_loss_y.item())
                make_step(d_loss_z, *optim_sched['disc_z'], model.parameters())
                make_step(d_loss_y, *optim_sched['disc_y'], model.parameters())

                batch_history['z_d_acc'].append(
                    find_acc(p_real_z, torch.ones_like(p_real_z), p_fake_z,
                             torch.zeros_like(p_fake_z)))
                batch_history['y_d_acc'].append(
                    find_acc(p_real_y, torch.ones_like(p_real_y), p_fake_y,
                             torch.zeros_like(p_fake_y)))

                # regularisation (generator/encoder)
                model.enc.train()
                z, y_ = model.get_latents(
                    x_unlab)  # TODO: try without generating new latents
                p_fake_z = model.discriminate(z, 'z')
                p_fake_y = model.discriminate(y_, 'y')
                g_loss = torch.mean(-torch.log(eps + p_fake_z) -
                                    torch.log(eps + p_fake_y)) * 1.5
                batch_history['enc'].append(g_loss.item())
                make_step(g_loss, *optim_sched['enc'], model.parameters())

            # semi-supervised
            if semisup:
                _, y_ = model.get_latents(xs)
                y_loss = F.cross_entropy(y_, ys) * (9 * pretraining + 1)
                batch_history['y_'].append(y_loss.item())
                make_step(y_loss, *optim_sched['y_'], model.parameters())

                batch_history['y_acc'].append(find_acc(y_, ys, threshold=0.5))

            if step > 0 and step % verbose_freq == 0:
                print(f"Batch {step}")
                for metric, record in batch_history.items():
                    batch_mean = np.mean(record[-verbose_freq:], axis=0)
                    try:
                        print(f"{metric:^15} : {batch_mean:.6f}")
                    except:
                        print(f"{metric:^15} : {batch_mean}")
                print('-' * 100)

        print(f"Epoch {epoch} Train:")
        for metric, record in batch_history.items():
            epoch_mean = np.mean(record[-len(train_dl):], axis=0)
            epoch_history['train'][metric].append(epoch_mean)
            try:
                print(f"|{metric:^15}| : {epoch_mean:.6f}")
            except:
                print(f"|{metric:^15}| : {epoch_mean}")

        val_metrics = None
        if val_flag:
            print(f"Epoch {epoch} Val:")
            val_metrics = evaluate(model,
                                   val_dl,
                                   device,
                                   pretraining=pretraining)
            for metric, record in val_metrics.items():
                val_mean = np.mean(record, axis=0)
                epoch_history['val'][metric].append(val_mean)
                try:
                    print(f"|{metric:^15}| : {val_mean:.6f}")
                except:
                    print(f"|{metric:^15}| : {val_mean}")

        if save:
            save_state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optim_state_dicts': {
                    name: optim.state_dict()
                    for name, (optim, sched) in optim_sched.items()
                }
            }
            fname = f'{model_name}_pretrain_{epoch}' if pretraining else f'{model_name}_{epoch}'
            torch.save(save_state, cache_dir + f'\\savestates\\{fname}.pkl')

        print(f"Epoch {epoch} - time taken: {time.time() - epoch_time}")
    if save:
        with open(cache_dir + f'\\records\\{fname}.pkl', 'wb') as stat_f:
            pickle.dump((epoch_history, batch_history), stat_f)
            stat_f.close()

    total_time = time.time() - start_time
    print(f'Total time taken: {total_time:.2f}')
    return epoch_history, batch_history
