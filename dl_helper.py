import torch
# import torch.nn as nn
import copy

from tqdm import tqdm
from time import time

from utils import seg_res_img, t_n  # , show
import torchvision

EPOCHS = 10
DEVICE = "cuda"


def load_weights(path, model, device):
    weight = torch.load(path, map_location=device)
    model.to(device)
    md_param = dict(model.state_dict())
    for i in list(weight.keys()):
        if weight[i].shape != md_param[i].shape:
            del weight[i]
    model.load_state_dict(weight, strict=False)
    return model


def load_model(title, model, name='model', device=DEVICE):
    return load_weights(f'./results/{title}/{name}.pt', model, device)


# save best model
def save_best_model(model, loss, title='dl'):
    if len(loss) <= 2 or (len(loss) > 2 and loss[-1] <= min(loss[:-1])):
        print('saving model')
        with open(f'./results/{title}/model.txt', 'w') as f:
            print(len(loss), file=f)
        torch.save(model.state_dict(), f'./results/{title}/model.pt')


def predict_batch_step(X, y, model, crop, device):
    if crop:
        X = X.flatten(0, 1)
        y = y.flatten(0, 1)
    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    pred = t_n(pred).contiguous().flatten(0, -2)
    if len(y.shape) >= 2:
        y = t_n(y).contiguous().flatten(0, -2)
    return pred, y


def predict_batch_acc_step(Xs, ys, model, crop, device):
    pred = []
    for i in range(len(Xs)):
        X = Xs[i]
        X = X.view(1, *X.shape).to(device)
        if crop:
            X = X.flatten(0, 1)
        pred.append(model(X))
    # t_n should be with b argument
    # but we don't care here
    pred = torch.cat([t_n(i).contiguous().flatten(0, -2) for i in pred])
    if len(ys.shape) >= 2:
        ys = torch.cat([t_n(i.to(device)).contiguous().flatten(0, -2)
                        for i in ys])
    return pred, ys


# do one batch
def compute_pred(X, y, model, grad_acc, crop, device):
    if grad_acc:
        y_pred, y_true = predict_batch_acc_step(X, y, model, crop, device)
    else:
        y_pred, y_true = predict_batch_step(X, y, model, crop, device)
    return y_pred, y_true


# one epoch
def iterate_step(dataloader, model, grad_acc, crop, compute_loss=False,
                 train=False, loss_fn=None, optimizer=None, store=False,
                 device=DEVICE, log=0):
    n_batch = len(dataloader)
    loss_total = 0
    y_pred_total = []
    y_true_total = []
    for batch, (X, y, *_) in tqdm(enumerate(dataloader), total=n_batch,
                                  disable=log == 0):
        y_pred, y_true = compute_pred(X, y, model, grad_acc, crop, device)
        if compute_loss:
            loss = loss_fn(y_pred, y_true)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_total += float(loss.item())
        if store:
            y_pred_total.append(y_pred)
            y_true_total.append(y_true)
        torch.cuda.empty_cache()
    if store:
        y_pred_total = torch.cat(y_pred_total)
        y_true_total = torch.cat(y_true_total)
    loss_total = loss_total / n_batch if n_batch != 0 else 0
    torch.cuda.empty_cache()
    return loss_total, y_pred_total, y_true_total


# do one epoch
def train_batches(dataloader, model, loss_fn, optimizer, grad_acc,
                  log=0, device=DEVICE, my_ds=True):
    model.train()
    model.to(device)
    if my_ds:
        dataloader.dataset.train()
    loss, _, _ = iterate_step(dataloader, model, grad_acc, crop=False,
                              compute_loss=True, train=True,
                              loss_fn=loss_fn, optimizer=optimizer,
                              device=device, log=log)
    if log == 1:
        print(f"training loss: {loss:>7f}")
    return loss


# run the validate step
def validate(dataloader, model, loss_fn, grad_acc, crop, log=0, device=DEVICE,
             my_ds=True, ep=-1, save=False):
    model.eval()
    model.to(device)
    if my_ds:
        dataloader.dataset.validate()
    with torch.no_grad():
        loss, y1, y2 = iterate_step(dataloader, model, grad_acc, crop,
                                    compute_loss=True, train=False,
                                    loss_fn=loss_fn, store=True,
                                    device=device, log=log)
        if save:
            ep += 1
            if ep in [0, 1, 3, 5, 10, 50, 100, 500]:
                y_pred = torch.argmax(y1, dim=1)
                s = 304
                y_pred = y_pred[:304*304*4].reshape(304, 304, 4).permute(2, 0, 1)
                true = y2[:304*304*4].reshape(304, 304, 4).permute(2, 0, 1)
                i = 0
                img = t_n(seg_res_img(y_pred[i], true[i]).cpu()).numpy()
                tmp = torchvision.transforms.functional.to_pil_image(img.astype('uint8'))
                tmp.save(f'loss_img_3m{ep}.png')
                # show(t_n(seg_res_img(y_pred[i], true[i]).cpu()))
    if log == 1:
        print(f"validate loss: {loss:>7f}")
    return loss


# run the testing step
def test(dataloader, model, grad_acc=False, crop=False, device=DEVICE, log=0,
         my_ds=True):
    model.eval()
    model.to(device)
    if my_ds:
        dataloader.dataset.test()
    with torch.no_grad():
        _, y_pred, y_true = iterate_step(dataloader, model, grad_acc, crop,
                                         compute_loss=False, train=False,
                                         device=device, store=True, log=log)
    return y_pred, y_true


# run through epoch
def train_epochs(tr_dl, v_dl, model, loss_fn, optimizer, grad_acc=False,
                 crop=False, log=0, save='best',
                 title='dl', epochs=EPOCHS, device=DEVICE, my_ds=True,
                 early_stop=400):
    loss_val = validate(v_dl, model, loss_fn, grad_acc, crop=crop, log=log,
                        device=device, my_ds=my_ds)
    loss_train_time = []
    loss_val_time = [loss_val]
    i = 0
    best_val_loss = (loss_val, i)
    while i < epochs and early_stop > (i - best_val_loss[1]):
        t = time()
        if log >= 1:
            print(f"Epoch {i+1}\n----------------------------")
        loss = train_batches(tr_dl, model, loss_fn, optimizer, grad_acc,
                             log=log, device=device, my_ds=my_ds)
        loss_train_time.append(loss)
        loss_val = validate(v_dl, model, loss_fn, grad_acc, crop=crop, log=log,
                            device=device, my_ds=my_ds, ep=i)
        loss_val_time.append(loss_val)
        el = time() - t
        if log >= 1:
            print(f"Trained epoch in {el:.1f} sec")
        if save == 'best':
            save_best_model(model, loss_val_time, title=title)
        if loss_val < best_val_loss[0]:
            best_val_loss = (loss_val, i)
        i += 1
    print('DONE!')
    if save == 'last':
        save_best_model(model, loss_val_time, title=title)
    # torch.save(model.state_dict(), f'./results/{title}/model_last.pt')
    return model, loss_train_time, loss_val_time


def train_cross(tr_dls, v_dls, model, loss_fn, optimizer_fn, grad_acc=False,
                crop=False, log=0, save='best',
                title='dl', epochs=EPOCHS, device=DEVICE, my_ds=True):
    models = {}
    loss_train_time_n = {}
    loss_val_time_n = {}
    for k in tr_dls:
        if log >= 1:
            print(f"FOLD {k+1}\n----------------------------")
        model = load_model(title, model, name='weights')
        optimizer = optimizer_fn(model.parameters())
        m, ltt, lvt = train_epochs(tr_dls[k], v_dls[k], model, loss_fn,
                                   optimizer, grad_acc, crop, log, save,
                                   f'{title}/{k}', epochs, device, my_ds=my_ds)
        models[k] = copy.deepcopy(m.cpu())
        loss_train_time_n[k] = ltt
        loss_val_time_n[k] = lvt
    return models, loss_train_time_n, loss_val_time_n


def train(cross, *args, **kwargs):
    if cross:
        return train_cross(*args, **kwargs)
    else:
        return train_epochs(*args, **kwargs)
