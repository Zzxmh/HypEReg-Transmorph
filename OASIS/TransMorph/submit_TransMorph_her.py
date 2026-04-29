import glob
import os
import utils
import numpy as np
import torch
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from scipy.ndimage.interpolation import zoom


def main():
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    test_dir = os.path.join(repo_root, "OASIS", "data", "Test")
    save_dir = os.path.join(repo_root, "OASIS", "data", "Submit", "submission", "task_03")
    os.makedirs(save_dir, exist_ok=True)

    model_idx = -1
    model_folder = "TransMorph_OASIS_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20/"
    model_dir = "experiments/" + model_folder
    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])["state_dict"]
    print("Best model: {}".format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    file_names = natsorted(glob.glob(os.path.join(test_dir, "*.pkl")))
    with torch.no_grad():
        for data in file_names:
            x, y, _, _ = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = os.path.basename(data).split(".")[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y), dim=1)
            _, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            np.savez(os.path.join(save_dir, "disp_{}.npz".format(file_name)), flow)


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
    main()
