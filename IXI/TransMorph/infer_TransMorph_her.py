"""IXI 测试集（Test/，115 例）推理：与 Results 中 Baseline 行数一致，供 analysis_trans 按 115 列计算。默认 dsc0.743.pth.tar。"""
import glob
import shutil
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn as nn

CKPT_NAME = "dsc0.743.pth.tar"


def _resolve_ixi_atlas(ixi_root):
    for name in ("atlas.pkl", "altas.pkl"):
        p = os.path.join(ixi_root, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No atlas.pkl (or altas.pkl) in {ixi_root}."
    )


def main():
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    ixi_root = os.path.join(repo_root, "IXI_data")
    atlas_dir = _resolve_ixi_atlas(ixi_root)
    # 与 IXI/Results 内 Baseline CSV 一致：Test 集 115 例
    test_subdir = "Test"
    test_dir = os.path.join(ixi_root, test_subdir)
    test_glob = os.path.join(ixi_root, test_subdir, "")

    model_folder = "TransMorph_IXI_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20/"
    model_dir = "experiments/" + model_folder
    ckpt_path = os.path.join(model_dir, CKPT_NAME)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    pkl_list = natsorted(glob.glob(test_glob + "*.pkl"))
    print("IXI inference: {} cases from {}".format(len(pkl_list), test_dir))
    if len(pkl_list) != 115:
        print("Warning: expected 115 Test .pkl for analysis_trans, got {}.".format(len(pkl_list)))
    csv_name = model_folder[:-1] + "_Test"

    d = utils.process_label()
    if not os.path.exists("Quantitative_Results/"):
        os.makedirs("Quantitative_Results/")
    out_base = "Quantitative_Results/" + csv_name
    if os.path.exists(out_base + ".csv"):
        os.remove(out_base + ".csv")
    csv_writter(model_folder[:-1], out_base)
    line = ""
    for i in range(46):
        line = line + "," + d[i]
    csv_writter(line + ",non_jec", out_base)

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    best_model = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    print("Loaded: {}".format(ckpt_path))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, "bilinear")
    reg_model.cuda()
    test_composed = transforms.Compose(
        [trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))]
    )
    test_set = datasets.IXIBrainInferDataset(
        pkl_list, atlas_dir, transforms=test_composed
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True
    )
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                def_seg = reg_model([x_seg_oh[:, i : i + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :]
            )
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line + "," + str(np.sum(jac_det <= 0) / np.prod(tar.shape))
            csv_writter(line, out_base)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print("det < 0: {}".format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print("Trans dsc: {:.4f}, Raw dsc: {:.4f}".format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print(
            "Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}".format(
                eval_dsc_def.avg, eval_dsc_def.std, eval_dsc_raw.avg, eval_dsc_raw.std
            )
        )
        print("deformed det: {}, std: {}".format(eval_det.avg, eval_det.std))
    out_csv = out_base + ".csv"
    # 供 IXI/analysis_trans.py 与 Baseline 同一 Results/ 目录对比
    results_dir = os.path.join(repo_root, "IXI", "Results")
    os.makedirs(results_dir, exist_ok=True)
    her_alias = os.path.join(results_dir, "TransMorph_HER_IXI.csv")
    shutil.copyfile(out_csv, her_alias)
    print("Also saved for analysis: {}".format(her_alias))
    return out_csv


def csv_writter(line, name):
    with open(name + ".csv", "a", encoding="utf-8", newline="") as file:
        file.write(line)
        file.write("\n")


if __name__ == "__main__":
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        print("     GPU #" + str(GPU_idx) + ": " + torch.cuda.get_device_name(GPU_idx))
    torch.cuda.set_device(GPU_iden)
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("CUDA: " + str(torch.cuda.is_available()))
    main()
