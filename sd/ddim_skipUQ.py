import argparse, os
import torch
import math
from itertools import islice
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from custom_ld import CustomLD
from dataset import laion_dataset
import torchvision.utils as tvu
from ldm.util import instantiate_from_config
from utils import NoiseScheduleVP, get_model_input_time
from ddimUQ_utils import compute_alpha, singlestep_ddim_sample, var_iteration, exp_iteration, \
    sample_from_gaussion

def load_model_from_config(config, ckpt, verbose=False): 
    print(f"Loading model from {ckpt}")  #theo doix tiến trình, in checkpoint
    pl_sd = torch.load(ckpt, map_location="cpu") #load ckpt vào CPU, đọc data model đã train
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] #chứa trọng số mạng
    model = instantiate_from_config(config.model) #dựng model Stable Diffusion
    m, u = model.load_state_dict(sd, strict=False) #oad trọng số vào kiến thức, khôi phục model từ ckpt
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval() #inference, k cần gradient (tắt dropout,batchnorm cố định)
    return model

def conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None): #cập nhật muy có điều kiện cờ bool
    if pre_wuq == True:
        return exp_iteration(model, exp_xt, seq, timestep, mc_eps_exp_t)
    else:
        return exp_iteration(model, exp_xt, seq, timestep, acc_eps_t)
    
def conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

    if pre_wuq == True:
        return var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep)
    else:
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
        at = compute_alpha(model.betas, t.long())
        at_next = compute_alpha(model.betas, next_t.long())
        var_xt_next = (at_next/at) * var_xt

        return var_xt_next

def get_scaled_var_eps(scale, var_eps_c, var_eps_uc):
    return pow(1-scale, 2)* var_eps_uc + pow(scale, 2)* var_eps_c
def get_scaled_exp_eps(scale, exp_eps_c, exp_eps_uc):   
    return (1-scale)* exp_eps_uc + scale* exp_eps_c

def chunk(it, size): #chia thành các tuple size phần tử
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
 
def main():
    parser = argparse.ArgumentParser() #đọc tham số từ command line

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a cheetah drinking an espresso",
        help="the prompt to render"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--laion_art_path",
        type=str,)
    parser.add_argument(
        "--local_image_path",
        type=str,)
    parser.add_argument("--mc_size", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--train_la_batch_size", type=int, default=4)
    parser.add_argument("--train_la_data_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default= 50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--total_n_samples', type=int, default=80)
    opt = parser.parse_args() #chứa tham số đã parse
    print(opt)
    seed_everything(opt.seed) #sinh kq ngẫu nhiên nhưng lặp được

    config = OmegaConf.load(f"{opt.config}") #đọc config yaml
    model = load_model_from_config(config, f"{opt.ckpt}") #load mô hình dfs từ ckpt
    # print(model.model.diffusion_model.out[2])
    # Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    #setup device và dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    train_dataset= laion_dataset(model, opt)
    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_la_batch_size, shuffle=False)
    custom_ld = CustomLD(model, train_dataloader)

    fixed_xT = torch.randn([opt.total_n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device) #sinh tensor Gaussian ngẫu nhiên x_T
##########   get t sequence (note that t is different from timestep)  ########## 

    skip = model.num_timesteps // opt.timesteps
    seq = range(0, model.num_timesteps, skip) #ds timestep thực sự dùng trong sampling

#########   get skip UQ rules  ##########  
# if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (opt.timesteps)
    for i in range(opt.timesteps-1, 0, -5):
        uq_array[i] = True
    
#########   get prompt  ##########  
    if opt.from_file: #nếu có file prompt
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f: #load nhiều dòng
            data = f.read().splitlines()
    else: #dùng 1 prompt duy nhất
        c = model.get_learned_conditioning(opt.prompt) #embedding của prompt
        c = torch.concat(opt.sample_batch_size * [c], dim=0)
        uc = model.get_learned_conditioning(opt.sample_batch_size * [""]) #embedding rỗng
        exp_dir = f'./ddim_exp/skipUQ/cfg{opt.scale}_{opt.prompt}_train{opt.train_la_data_size}_step{opt.timesteps}_S{opt.mc_size}/'
        os.makedirs(exp_dir, exist_ok=True)

#########   start sample  ########## 
    total_n_samples = opt.total_n_samples #tổng số ảnh muốn tạo
    if total_n_samples % opt.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by opt.sample_batch_size, but got {} and {}".format(total_n_samples, opt.sample_batch_size))
    n_rounds = total_n_samples // opt.sample_batch_size #chia thành nhiều round
    var_sum = torch.zeros((opt.sample_batch_size, n_rounds)).to(device) #lưu var từng sample
    img_id = 1000000
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope(): #mô hình ổn định khi sinh ảnh
                for prompts in tqdm(data):
                    sample_x = []
                    c = model.get_learned_conditioning(prompts)
                    c = torch.concat(opt.sample_batch_size * [c], dim=0)
                    uc = model.get_learned_conditioning(opt.sample_batch_size * [""])
                    exp_dir = f'./ddim_exp/skipUQ/cfg{opt.scale}_{prompts}_train{opt.train_la_data_size}_step{opt.timesteps}_S{opt.mc_size}/'
                    os.makedirs(exp_dir, exist_ok=True)
                    
                    for loop in tqdm(
                        range(n_rounds), desc="Generating image samples for FID evaluation."
                    ):
                        
                        xT, timestep, mc_sample_size  = fixed_xT[loop*opt.sample_batch_size:(loop+1)*opt.sample_batch_size, :, :, :], opt.timesteps-1, opt.mc_size
                        T = seq[timestep]
                        if uq_array[timestep] == True: #chế độ UQ
                            #lấy mean + var từ custom_ld
                            #tạo nhiều mẫu MC (nhiều xt_next_i) để tính cov
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xT, (torch.ones(opt.sample_batch_size) * T).to(xT.device), c=c) 
                            eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xT, (torch.ones(opt.sample_batch_size) * T).to(xT.device), c=uc) 
                            eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                            cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                            list_eps_mu_t_next_i = torch.unsqueeze(eps_mu_t_next, dim=0)
                        else:
                            # chỉ lấy E (accurate_forward)
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            eps_mu_t_next_c = custom_ld.accurate_forward(xT, (torch.ones(opt.sample_batch_size) * T).to(xT.device), c=c)
                            eps_mu_t_next_uc = custom_ld.accurate_forward(xT, (torch.ones(opt.sample_batch_size) * T).to(xT.device), c=uc)
                            eps_mu_t_next = get_scaled_exp_eps(opt.scale, eps_mu_t_next_c, eps_mu_t_next_uc)

                        for timestep in range(opt.timesteps-1, 0, -1):

                            if uq_array[timestep] == True:
                                xt = xt_next
                                exp_xt, var_xt = exp_xt_next, var_xt_next
                                eps_mu_t, eps_var_t, cov_xt_epst = eps_mu_t_next, eps_var_t_next, cov_xt_next_epst_next
                                mc_eps_exp_t = torch.mean(list_eps_mu_t_next_i, dim=0)
                            else: 
                                xt = xt_next
                                exp_xt, var_xt = exp_xt_next, var_xt_next
                                eps_mu_t = eps_mu_t_next

                            if uq_array[timestep] == True:
                                # dùng sample_from_gaussion để tính noise từ mean/var
                                eps_t= sample_from_gaussion(eps_mu_t, eps_var_t)
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq= uq_array[timestep])
                                if uq_array[timestep-1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i=[], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                            
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            else:
                                # dùng giá trị deterministic.
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_mu_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t = eps_mu_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst= None, var_epst=None, seq= seq, timestep=timestep, pre_wuq= uq_array[timestep])
                                if uq_array[timestep-1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i=[], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                            
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(opt.sample_batch_size) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                        
                        ####### Save variance and sample image  ######         
                        var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))
                        x_samples = model.decode_first_stage(xt_next)
                        x = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        # os.makedirs(os.path.join(exp_dir, 'sam/'), exist_ok=True)
                        # for i in range(x.shape[0]):
                        #     path = os.path.join(exp_dir, 'sam/', f"{img_id}.png")
                        #     tvu.save_image(x.cpu()[i].float(), path)
                        #     img_id += 1
                        sample_x.append(x)

                    # xếp ảnh theo var(cao thì đứng đầu)
                    sample_x = torch.concat(sample_x, dim=0)
                    var = []
                    for j in range(n_rounds):
                        var.append(var_sum[:, j])
                    var = torch.concat(var, dim=0)
                    sorted_var, sorted_indices = torch.sort(var, descending=True)
                    reordered_sample_x = torch.index_select(sample_x, dim=0, index=sorted_indices.int())
                    grid_sample_x = tvu.make_grid(reordered_sample_x, nrow=8, padding=2)
                    tvu.save_image(grid_sample_x.cpu().float(), os.path.join(exp_dir, "sorted_sample.png"))

                    print(f'Sampling {total_n_samples} images in {exp_dir}')
                    torch.save(var_sum.cpu(), os.path.join(exp_dir, 'var_sum.pt'))

if __name__ == "__main__":
    main()
