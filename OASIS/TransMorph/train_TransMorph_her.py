from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from losses_her import HyperelasticLoss


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def _resolve_oasis_paths():
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    oasis_root = os.path.join(repo_root, "OASIS", "data")
    train_dir = os.path.join(oasis_root, "All")
    val_dir = os.path.join(oasis_root, "Test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing OASIS train dir: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Missing OASIS val dir: {val_dir}")
    return train_dir, val_dir


def main():
    batch_size = 2
    train_dir, val_dir = _resolve_oasis_paths()
    weights = [1.0, 1.0, 1.0]  # [NCC, Grad3d, HER]
    save_dir = "TransMorph_OASIS_HER_ncc_{}_grad_{}_her_{}_a0_b0.02_g20/".format(
        weights[0], weights[1], weights[2]
    )
    if not os.path.exists("experiments/" + save_dir):
        os.makedirs("experiments/" + save_dir)
    if not os.path.exists("logs/" + save_dir):
        os.makedirs("logs/" + save_dir)
    sys.stdout = Logger("logs/" + save_dir)
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 500  # max training epoch
    cont_training = False  # if continue training

    """
    Initialize model
    """
    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    model.cuda()

    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(config.img_size, "nearest")
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, "bilinear")
    reg_model_bilin.cuda()

    """
    If continue from previous training
    """
    if cont_training:
        epoch_start = 201
        model_dir = "experiments/" + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])["state_dict"]
        print("Model: {} loaded!".format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    """
    Initialize training
    """
    train_composed = transforms.Compose(
        [trans.RandomFlip(0), trans.NumpyType((np.float32, np.int16))]
    )
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_paths = natsorted(glob.glob(os.path.join(train_dir, "*.pkl")))
    val_paths = natsorted(glob.glob(os.path.join(val_dir, "*.pkl")))
    if len(train_paths) == 0:
        raise FileNotFoundError(f"No training .pkl found in {train_dir}")
    if len(val_paths) == 0:
        raise FileNotFoundError(f"No validation .pkl found in {val_dir}")
    print(f"OASIS TransMorph-HER: train={len(train_paths)} val={len(val_paths)}")

    train_set = datasets.OASISBrainDataset(train_paths, transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(val_paths, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty="l2")
    criterion_her = HyperelasticLoss(
        alpha_length=0.0,
        beta_volume=0.02,
        gamma_fold=20.0,
    )
    best_dsc = 0
    writer = SummaryWriter(log_dir="logs/" + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print("Training Starts")
        """
        Training
        """
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            x_in = torch.cat((x, y), dim=1)
            output, flow = model(x_in)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow, y) * weights[1]
            loss_her = criterion_her(flow) * weights[2]
            loss = loss_ncc + loss_reg + loss_her
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in, output, flow

            y_in = torch.cat((y, x), dim=1)
            output, flow = model(y_in)
            loss_ncc_b = criterion_ncc(output, x) * weights[0]
            loss_reg_b = criterion_reg(flow, x) * weights[1]
            loss_her_b = criterion_her(flow) * weights[2]
            loss_b = loss_ncc_b + loss_reg_b + loss_her_b
            loss_all.update(loss_b.item(), x.numel())
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()

            print(
                "Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, HER: {:.6f}".format(
                    idx,
                    len(train_loader),
                    loss_b.item(),
                    (loss_ncc + loss_ncc_b).item() / 2,
                    (loss_reg + loss_reg_b).item() / 2,
                    (loss_her + loss_her_b).item() / 2,
                )
            )

        writer.add_scalar("Loss/train", loss_all.avg, epoch)
        print("Epoch {} loss {:.4f}".format(epoch, loss_all.avg))
        """
        Validation
        """
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_dsc": best_dsc,
                "optimizer": optimizer.state_dict(),
            },
            save_dir="experiments/" + save_dir,
            filename="dsc{:.4f}.pth.tar".format(eval_dsc.avg),
        )
        writer.add_scalar("DSC/validate", eval_dsc.avg, epoch)
        plt.switch_backend("agg")
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure("Grid", grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure("input", x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure("ground truth", tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure("prediction", pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        del def_out, def_grid, grid_img, output
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")
        plt.imshow(img[i, :, :], cmap="gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group["lr"] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir="models", filename="checkpoint.pth.tar", max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + "*"))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + "*"))


if __name__ == "__main__":
    """
    GPU configuration
    """
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))
    torch.manual_seed(0)
    main()
