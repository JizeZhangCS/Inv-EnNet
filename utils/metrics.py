import torch
import torch.nn.functional as F
try:
    import matlab
    import matlab.engine
except ModuleNotFoundError:
    print("Warning: no matlab installation found. Please do not involve NIQE in the metrics.")

# Printing the status of the models each epoch
def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction='mean')
    return -10 * torch.log10(mse + eps)

def niqe(base_dir):
    base_dir = "./" + str(base_dir) + "/"
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath("./niqe_release"))
    niqe_value = eng.comp_niqe(base_dir)
    eng.exit()
    return niqe_value
