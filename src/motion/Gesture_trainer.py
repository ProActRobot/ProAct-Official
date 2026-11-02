import os
import random
import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import joblib as jl

# from ddpm.diffusion import Gesture_Diffusion
from dataset.GesDiff_datamodule_double_with_state_mfcc import GesDiffDataModule_Interaction_with_state_mfcc
from dataset.GesDiff_datamodule_text import GesDiffDataModule_BVHText
from dataset.TripletsDiff_datamodule import GesDiffDataModule_Unified
from utils.ldm_lr_scheduler import LambdaLinearScheduler
from data_process.utils.utils import inv_standardize
from data_process.pymo.writers import BVHWriter


def setup(rank, world_size):

    if dist.is_initialized():
        return
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def to_bvh(anim_clip, save_path, data_pipe, framerate):
    shape = anim_clip.shape
    if len(shape) == 3:
        combine_anim_clip = anim_clip.reshape((shape[0] * shape[1], shape[2]))
    else:
        combine_anim_clip = anim_clip

    combine_anim_clip = np.expand_dims(combine_anim_clip, axis=0)
    inv_data = data_pipe.inverse_transform(combine_anim_clip)

    writer = BVHWriter()
    with open(save_path, 'w') as f:
        writer.write(inv_data[0], f, framerate=framerate)


def smoothstep(x):
    return x**2 * (3 - 2 * x)


class Gesture_trainer():
    def __init__(
        self,
        yaml_file_path: str = "./configs/training/backbone.yaml",
        inference_save_dir: str = ''
    ):

        self.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        self.local_rank = int(os.environ['LOCAL_RANK']) if "LOCAL_RANK" in os.environ else 0
        self.world_size = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
        setup(self.local_rank, self.world_size)

        self.config = OmegaConf.load(yaml_file_path)
        self.task_mode = self.config.task_mode
        self.config_trainer = self.config.trainer

        if self.config_trainer.method == 'flow_matching':
            print("Using flow matching...")
            # from ddpm.flow_matching import Gesture_Diffusion_double
            from ddpm.flow_matching_new import Gesture_Diffusion_double
        elif self.config_trainer.method == 'ddim':
            print("Using DDIM...")
            from ddpm.diffusion_double import Gesture_Diffusion_double
        elif self.config_trainer.method == 'diffusers':
            print("Using Diffusers...")
            from ddpm.diffusion_double_diffuser import Gesture_Diffusion_double
        else:
            raise ValueError("Invalid training method")

        if self.config_trainer.new_audio_encoder:
            print("Using new audio encoder...")
            from model.model_for_socialrobot_cfg_mfcc import Gesture_DiT_interaction_7_cfg_mfcc
        else:
            print("Using original audio encoder...")
            from model.model_for_interaction_7_cfg_mfcc import Gesture_DiT_interaction_7_cfg_mfcc

        self.config_data = self.config.data
        self.config_model = self.config.model
        self.config_ddpm = self.config.ddpm
        self.config_scheduler = self.config.scheduler
        self.config_log = self.config.logger

        self.use_scheduler = self.config_trainer.use_scheduler
        self.learning_rate = self.config_trainer.learning_rate
        self.mode = self.config_trainer.mode
        self.epochs = self.config_trainer.epochs
        self.valid_interval = self.config_trainer.valid_interval
        self.save_interval = self.config_trainer.save_interval
        self.start_save_point = self.config_trainer.start_save_point
        self.checkpoint_path = self.config_trainer.checkpoint_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_trainer.matmul_precision == 'high':
            torch.set_float32_matmul_precision('high')
        elif self.config_trainer.matmul_precision == 'medium':
            torch.set_float32_matmul_precision('medium')
        else:
            raise ValueError("Invalid matmul precision setting")


        if self.mode == "train":

            self.train_save_dir = None
            self.current_time = None
            time_tensor = torch.zeros(100, dtype=torch.uint8).to(device)
            if dist.get_rank() == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                encoded_time = cur_time.encode('utf-8')
                time_tensor[:len(encoded_time)] = torch.tensor(list(encoded_time), dtype=torch.uint8).to(device)

            dist.broadcast(time_tensor, src=0)
            self.current_time = bytes(time_tensor.tolist()).decode('utf-8').rstrip('\x00')
            self.train_save_dir = os.path.join(self.config_trainer.save_dir, self.current_time)
            os.makedirs(self.train_save_dir, exist_ok=True)
        elif self.mode == "inference":
            assert inference_save_dir is not None
            self.inference_save_dir = inference_save_dir or os.getenv("MOTION_OUTPUT_DIR", os.path.join("runtime", "motion"))
            os.makedirs(self.inference_save_dir, exist_ok=True)

        # print(self.rank, self.local_rank, self.world_size, self.train_save_dir)
        # wait for all processes to be ready
        # dist.barrier()

        self.set_seed(self.config_trainer.seed)

        self.output_scaler = jl.load(os.path.join(self.config.data.data_root, 'output_scaler.sav'))
        self.data_pipe = jl.load(os.path.join(self.config.data.data_root, 'data_pipe_' + str(self.config.data.framerate) + 'fps.sav'))


        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True
            if self.config_trainer.matmul_precision == 'high':
                torch.set_float32_matmul_precision('high')
            elif self.config_trainer.matmul_precision == 'medium':
                torch.set_float32_matmul_precision('medium')
            else:
                raise ValueError("Invalid matmul precision setting")


            torch.backends.cudnn.benchmark = True


            try:
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_mem_efficient=True,
                    enable_math=False,
                )
            except Exception:
                pass

        if self.task_mode == "interaction_7_cfg_mfcc":
            self.datamodule = GesDiffDataModule_Interaction_with_state_mfcc(**self.config_data, local_rank=self.local_rank, world_size=self.world_size)
            model = Gesture_DiT_interaction_7_cfg_mfcc(**self.config_model).to(device)
        else:
            raise ValueError("Invalid task_mode")

        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

        self.model = model.to(self.local_rank)
        if self.task_mode in ["interaction_3", "interaction_4", "interaction_4_cfg", "interaction_4_cfg_2", "interaction_5_cfg", "interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc", "interaction_7_cfg_single"]:
            self.diffusion = Gesture_Diffusion_double(model=self.model, **self.config_ddpm).to(self.local_rank)
        else:
            raise NotImplementedError("Only double diffusion is implemented.")
            # self.diffusion = Gesture_Diffusion(model=self.model, **self.config_ddpm).to(self.local_rank)
        self.diffusion = DDP(self.diffusion, device_ids=[self.local_rank], find_unused_parameters=True)

        if self.use_scheduler:
            self.opt, self.scheduler = self.configure_optimizers()
        else:
            self.opt = self.configure_optimizers()


        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)

            # self.diffusion.load_state_dict(checkpoint['model_state_dict'])

            def fix_model_weights(state_dict):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if '_orig_mod.' in key:
                        key = key.replace('_orig_mod.', '')

                    new_state_dict[key] = value

                return new_state_dict

            fixed_weights = fix_model_weights(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            self.diffusion.load_state_dict(fixed_weights, strict=False)
            # self.diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if self.mode == "train":
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.use_scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        elif self.task_mode == "transfer_learning":
            self.transfer_learning_checkpoint_path = self.config_trainer.transfer_learning_checkpoint_path
            self.need_frozen_step = self.config_trainer.need_frozen_step
            if self.mode == "train":
                print(f"Loading transfer learning checkpoint from {self.transfer_learning_checkpoint_path}")
                checkpoint = torch.load(self.transfer_learning_checkpoint_path, map_location=device, weights_only=False)
                self.keys_to_remove = [
                    'module.model.motion_encoder.net.1.weight',
                    'module.model.motion_encoder.net.1.bias',
                    'module.model.motion_decoder.net.1.weight',
                    'module.model.motion_decoder.net.1.bias',
                    'module.model.final_layer.output.net.1.weight',
                    'module.model.final_layer.output.net.1.bias'
                ]

                for name, param in checkpoint['model_state_dict'].items():
                    if 'model' not in name:
                        self.keys_to_remove.append(name)
                for key in self.keys_to_remove:
                    if key in checkpoint['model_state_dict']:
                        del checkpoint['model_state_dict'][key]

                # print(checkpoint['model_state_dict'].keys())

                self.diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)


                for name, param in self.diffusion.named_parameters():
                    if name not in self.keys_to_remove:
                        param.requires_grad = False



        if self.mode == "train":
            self.train_sampler, self.train_data, self.valid_data, self.test_data = self.datamodule.prepare_dataloaders(self.mode)
            if dist.get_rank() == 0:
                wandb.init(project=self.config_log.wandb_pj_name, name=self.config_log.exp_name)
                config_save_path = os.path.join(self.train_save_dir, "train_config.yaml")
                OmegaConf.save(config=self.config, f=config_save_path)
                wdir = os.path.join(self.train_save_dir, "weights")
                os.makedirs(wdir, exist_ok=True)
        elif self.mode == "inference":
            config_save_path = os.path.join(self.inference_save_dir, "inference_config.yaml")
            OmegaConf.save(config=self.config, f=config_save_path)
            pass
        else:
            raise ValueError("Invalid mode")

        # if dist.get_rank() == 0:
        #     pass


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.diffusion.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            scheduler = LambdaLinearScheduler(**self.config_scheduler)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=scheduler.schedule)
            return opt, scheduler
        return opt


    def eval(self):
        self.diffusion.eval()


    def train(self):
        self.diffusion.train()


    def train_loop(self):
        self.train()

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            # train
            self.train()
            self.train_sampler.set_epoch(epoch)
            epoch_train_loss_dict = {}
            epoch_train_loss, epoch_train_loss_vlb = 0, 0

            if self.task_mode == "transfer_learning":
                if self.need_frozen_step >= 0 and global_step >= self.need_frozen_step:
                    print(f"Unfreezing model parameters at step {global_step}...")
                    for name, param in self.diffusion.named_parameters():
                        if name not in self.keys_to_remove:
                            param.requires_grad = True

            for step, batch in enumerate(tqdm(self.train_data) if dist.get_rank() == 0 else self.train_data):
                motion_data = batch['motion_data'].to(self.local_rank)
                cond = batch['cond'].to(self.local_rank)
                state = None
                if self.task_mode in ["interaction_5_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc", "interaction_7_cfg_single"]:
                    state = batch['state'].to(self.local_rank)

                loss, loss_dict = self.diffusion(motion_data, cond, state)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=0.5)
                self.opt.step()
                if self.use_scheduler:
                    self.scheduler.step()
                    lr = self.opt.param_groups[0]['lr']
                    loss_dict['train/lr'] = lr

                global_step += 1

                # ema update and train loss update only on main
                # loss_value, vlb_loss_value = loss_dict['train/loss'], loss_dict['train/loss_vlb']
                # dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)
                # dist.all_reduce(vlb_loss_value, op=dist.ReduceOp.AVG)
                # loss_dict['train/loss'] = loss_value
                # loss_dict['train/loss_vlb'] = vlb_loss_value
                if dist.get_rank() == 0:
                    epoch_train_loss += loss_dict['train/loss'].item()
                    epoch_train_loss_vlb += loss_dict['train/loss_vlb'].item()
                    wandb.log(loss_dict)

                # break

            if dist.get_rank() == 0:
                epoch_train_loss_dict['train_epoch/loss'] = epoch_train_loss / len(self.train_data)
                epoch_train_loss_dict['train_epoch/loss_vlb'] = epoch_train_loss_vlb / len(self.train_data)
                epoch_train_loss_dict['train/epoch'] = epoch
                wandb.log(epoch_train_loss_dict)

            # validation

            torch.cuda.empty_cache()
            if epoch % self.valid_interval == 0:
                self.eval()
                with torch.no_grad():
                    epoch_val_loss_dict = {}
                    epoch_val_loss, epoch_val_loss_vlb = 0, 0

                    for step, batch in enumerate(tqdm(self.valid_data) if dist.get_rank() == 0 else self.valid_data):
                        motion_data = batch['motion_data'].to(self.local_rank)
                        cond = batch['cond'].to(self.local_rank)
                        state = None
                        if self.task_mode in ["interaction_5_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc", "interaction_7_cfg_single"]:
                            state = batch['state'].to(self.local_rank)
                        loss, loss_dict = self.diffusion(motion_data, cond, state)
                        if torch.isnan(loss).any():
                            raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                        if dist.get_rank() == 0:
                            epoch_val_loss += loss_dict['val/loss'].item()
                            epoch_val_loss_vlb += loss_dict['val/loss_vlb'].item()

                    if dist.get_rank() == 0:
                        epoch_val_loss_dict['val_epoch/loss'] = epoch_val_loss / len(self.valid_data)
                        epoch_val_loss_dict['val_epoch/loss_vlb'] = epoch_val_loss_vlb / len(self.valid_data)
                        epoch_val_loss_dict['val/epoch'] = epoch
                        wandb.log(epoch_val_loss_dict)

                if epoch_val_loss < best_val_loss:
                    if dist.get_rank() == 0:
                        save_ckpt_path = os.path.join(self.train_save_dir, "weights", "model_best.pt")
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.diffusion.state_dict(),
                            'optimizer_state_dict': self.opt.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                        }
                        # torch.save(self.model.state_dict(), save_ckpt_path)
                        torch.save(checkpoint, save_ckpt_path)
                        print(f"Model saved at {save_ckpt_path}")

            # save_model
            if epoch % self.save_interval == 0 and epoch >= self.start_save_point:

                if self.config_trainer.method == 'diffusers' or self.config_trainer.method == 'ddim':
                    for step, batch in enumerate(tqdm(self.valid_data) if dist.get_rank() == 0 else self.valid_data):
                        cond = batch['cond'].to(self.local_rank)
                        object_motion = None
                        if self.task_mode not in ["interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                            object_motion = batch['object_motion'].to(self.local_rank)
                        state = None
                        if self.task_mode in ["interaction_5_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                            state = batch['state'].to(self.local_rank)
                        bs, _, _ = cond.shape
                        shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)
                        motion = self.diffusion.module.p_sample_loop(
                            cond, object_motion, shape, return_intermediates=False, state=state
                        )
                        motion = motion.cpu().numpy()
                        motion = inv_standardize(motion, self.output_scaler)
                        bs_m, len_m, dim_m = motion.shape
                        assert len_m % 2 == 0
                        half_len_m = len_m // 2
                        if self.task_mode == "single_no_cond":
                            combine_motion = motion
                        else:
                            combine_motion = np.zeros((bs_m + 1, half_len_m, dim_m))
                            combine_motion[:-1] = motion[:, :half_len_m]
                            combine_motion[-1] = motion[-1, half_len_m:]
                        combine_motion = combine_motion.reshape(-1, dim_m)
                        bvh_save_path = os.path.join(self.train_save_dir, "val_p_sample_epoch_{}".format(epoch) + ".bvh")
                        print(f"combine_motion.shape: {combine_motion.shape}", f"bvh_save_path: {bvh_save_path}")
                        to_bvh(combine_motion, bvh_save_path, self.data_pipe, self.config_data.framerate)

                    self.render_sample_double_g1(
                        inference_sovler='DDIM_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_DDIM_eta1_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=1.
                    )

                    self.render_sample_double_g1(
                        inference_sovler='DDIM_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_DDIM_eta0_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=0.
                    )

                if self.config_trainer.method == 'diffusers':
                    self.render_sample_double_g1(
                        inference_sovler='UniPC_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_UniPC_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=20,
                        guidance_weight=3,
                        eta=1.
                    )

                if self.config_trainer.method == 'flow_matching':
                    self.render_sample_double_g1(
                        inference_sovler='flow_matching_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_fm_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=1.
                    )

                if dist.get_rank() == 0:
                    save_ckpt_path = os.path.join(self.train_save_dir, "weights", f"model_{epoch}.pt")
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.diffusion.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }
                    # torch.save(self.model.state_dict(), save_ckpt_path)
                    torch.save(checkpoint, save_ckpt_path)
                    print(f"Model saved at {save_ckpt_path}")

        if dist.get_rank() == 0:
            wandb.run.finish()

        cleanup()

    def render_sample_double_g1(self,
                      inference_sovler,
                      test_loaders,
                      test_file_save_names,
                      output_scaler,
                      data_pipe,
                      fps,
                      sample_steps=50,
                      guidance_weight=3,
                      eta=0.):
        self.eval()
        assert self.task_mode in ["interaction_3", "interaction_4", "interaction_4_cfg", "interaction_4_cfg_2", "interaction_5_cfg", "interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        # print("is_cfg:", is_cfg)
        with torch.no_grad():
            for i in range(len(test_loaders)):
                test_loader = test_loaders[i]
                test_audio_name = test_file_save_names[i]
                print(f"Rendering {test_audio_name}...")
                for step, batch in enumerate(tqdm(test_loader) if dist.get_rank() == 0 else test_loader):
                    cond = batch['cond'].to(self.local_rank)

                    object_motion = None
                    if self.task_mode not in ["interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                        object_motion = batch['object_motion'].to(self.local_rank)

                    state = None
                    if self.task_mode in ["interaction_5_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                        state = batch['state'].to(self.local_rank)
                    print(cond.shape)

                    bs, _, _ = cond.shape
                    shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

                    if inference_sovler == 'DDPM':
                        motion = self.diffusion.module.p_sample_loop(cond, object_motion, state, shape, return_intermediates=False)
                    elif inference_sovler == 'DDIM_edge':
                        motion = self.diffusion.module.ddim_sample_loop(cond, object_motion, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
                    elif inference_sovler == 'UniPC_edge':
                        motion = self.diffusion.module.unipc_sample_loop(cond, object_motion, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
                    elif inference_sovler == 'flow_matching_edge':
                        motion = self.diffusion.module.fm_sample_loop(cond, object_motion, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
                    motion = motion.cpu().numpy()
                    motion = inv_standardize(motion, output_scaler)

                    bs_m, len_m, dim_m = motion.shape
                    assert len_m % 2 == 0
                    half_len_m = len_m // 2

                    if self.task_mode == "single_no_cond":
                        combine_motion = motion
                    else:
                        combine_motion = np.zeros((bs_m + 1, half_len_m, dim_m))
                        combine_motion[:-1] = motion[:, :half_len_m]
                        combine_motion[-1] = motion[-1, half_len_m:]

                    combine_motion = combine_motion.reshape(-1, dim_m)

                    if self.mode == "train":
                        bvh_save_path = os.path.join(self.train_save_dir, test_audio_name[0] + ".bvh")
                    else:
                        bvh_save_path = os.path.join(self.inference_save_dir, test_audio_name[0] + ".bvh")
                    print(f"combine_motion.shape: {combine_motion.shape}", f"bvh_save_path: {bvh_save_path}")
                    to_bvh(combine_motion, bvh_save_path, data_pipe, fps)

    def render_sample_streaming_g1(self,
                      inference_sovler,
                      cond,
                      previous_motion_last_half=None,
                      cache_length=0,
                      sample_steps=50,
                      guidance_weight=2,
                      eta=1.):
        self.eval()
        assert self.task_mode in ["interaction_7_cfg_mfcc"]
        # is_cfg = False
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        # print("is_cfg:", is_cfg)
        with torch.no_grad():
            bs, _, _ = cond.shape
            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            if inference_sovler == 'DDIM_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'DDIM_edge_w_prior':
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge_w_prior':
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.fm_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge_w_prior':
                motion, cache = self.diffusion.module.fm_sample_loop_streaming(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)

            motion = motion[:, cache_length:, :]  # remove the padding frames
            cache = cache[:, cache_length:, :]
            return motion, cache

class Controlnet_trainer():
    def __init__(
        self,
        yaml_file_path: str = "./configs/training/controlnet.yaml",
        inference_save_dir: str = ''
    ):

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        setup(self.local_rank, self.world_size)

        self.config = OmegaConf.load(yaml_file_path)
        self.task_mode = self.config.task_mode
        self.config_trainer = self.config.trainer

        if self.config_trainer.method == 'flow_matching':
            print("Using flow matching for ControlNet training...")
            # from ddpm.flow_matching import Gesture_Diffusion_double
            from ddpm.flow_matching_new import Gesture_Diffusion_double
        elif self.config_trainer.method == 'ddim':
            print("Using DDIM for ControlNet training...")
            from ddpm.diffusion_double import Gesture_Diffusion_double
        elif self.config_trainer.method == 'diffusers':
            print("Using Diffusers for ControlNet training...")
            from ddpm.diffusion_double_diffuser import Gesture_Diffusion_double
        else:
            raise ValueError("Invalid training method")

        if self.config_trainer.new_audio_encoder:
            print("Using new audio encoder for ControlNet model...")
            from model.model_for_socialrobot_cfg_mfcc import Gesture_DiT_interaction_7_cfg_mfcc, ControlNet_DiT
        else:
            print("Using original audio encoder for ControlNet model...")
            from model.model_for_interaction_7_cfg_mfcc import Gesture_DiT_interaction_7_cfg_mfcc, ControlNet_DiT

        self.config_data = self.config.data
        self.config_model = self.config.model
        self.config_controlnet = self.config.controlnet
        self.config_ddpm = self.config.ddpm
        self.config_scheduler = self.config.scheduler
        self.config_log = self.config.logger

        self.use_scheduler = self.config_trainer.use_scheduler
        self.learning_rate = self.config_trainer.learning_rate
        self.mode = self.config_trainer.mode
        self.epochs = self.config_trainer.epochs
        self.valid_interval = self.config_trainer.valid_interval
        self.save_interval = self.config_trainer.save_interval
        self.start_save_point = self.config_trainer.start_save_point
        self.checkpoint_path = self.config_trainer.checkpoint_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_trainer.matmul_precision == 'high':
                torch.set_float32_matmul_precision('high')
        elif self.config_trainer.matmul_precision == 'medium':
            torch.set_float32_matmul_precision('medium')
        else:
            raise ValueError("Invalid matmul precision setting")


        if self.mode == "train":

            self.train_save_dir = None
            self.current_time = None
            time_tensor = torch.zeros(100, dtype=torch.uint8).to(self.device)
            if dist.get_rank() == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                encoded_time = cur_time.encode('utf-8')
                time_tensor[:len(encoded_time)] = torch.tensor(list(encoded_time), dtype=torch.uint8).to(self.device)

            dist.broadcast(time_tensor, src=0)
            self.current_time = bytes(time_tensor.tolist()).decode('utf-8').rstrip('\x00')
            self.train_save_dir = os.path.join(self.config_trainer.save_dir, self.current_time)
            os.makedirs(self.train_save_dir, exist_ok=True)
        elif self.mode == "inference":
            assert inference_save_dir is not None
            self.inference_save_dir = inference_save_dir or os.getenv("MOTION_OUTPUT_DIR", os.path.join("runtime", "motion"))
            os.makedirs(self.inference_save_dir, exist_ok=True)

        # print(self.rank, self.local_rank, self.world_size, self.train_save_dir)
        # wait for all processes to be ready
        # dist.barrier()

        self.set_seed(self.config_trainer.seed)

        self.output_scaler = jl.load(os.path.join(self.config.data.data_root, 'output_scaler.sav'))
        self.data_pipe = jl.load(os.path.join(self.config.data.data_root, 'data_pipe_' + str(self.config.data.framerate) + 'fps.sav'))

        if self.task_mode == "controlnet_interaction_7_cfg_mfcc":
            self.datamodule = GesDiffDataModule_BVHText(**self.config_data, local_rank=self.local_rank, world_size=self.world_size)
            diffusion_model = Gesture_DiT_interaction_7_cfg_mfcc(**self.config_model).to(self.device)
            controlnet = ControlNet_DiT(diffusion_model, **self.config_controlnet)
        else:
            raise ValueError("Invalid task_mode")

        print("Model has {} parameters".format(sum(y.numel() for y in controlnet.parameters())))

        self.model_diffusion = diffusion_model.to(self.local_rank)
        self.model = controlnet.to(self.local_rank)
        if self.task_mode in ["controlnet_interaction_7_cfg_mfcc"]:
            self.diffusion = Gesture_Diffusion_double(model=self.model, **self.config_ddpm).to(self.local_rank)
        else:
            raise NotImplementedError("Only double diffusion is implemented for ControlNet.")
        self.diffusion = DDP(self.diffusion, device_ids=[self.local_rank], find_unused_parameters=True)

        if self.use_scheduler:
            self.opt, self.scheduler = self.configure_optimizers()
        else:
            self.opt = self.configure_optimizers()


        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])
            if self.mode == "train":
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.use_scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        elif self.task_mode == "transfer_learning":
            self.transfer_learning_checkpoint_path = self.config_trainer.transfer_learning_checkpoint_path
            self.need_frozen_step = self.config_trainer.need_frozen_step
            if self.mode == "train":
                print(f"Loading transfer learning checkpoint from {self.transfer_learning_checkpoint_path}")
                checkpoint = torch.load(self.transfer_learning_checkpoint_path, map_location=self.device, weights_only=False)
                self.keys_to_remove = [
                    'module.model.motion_encoder.net.1.weight',
                    'module.model.motion_encoder.net.1.bias',
                    'module.model.motion_decoder.net.1.weight',
                    'module.model.motion_decoder.net.1.bias',
                    'module.model.final_layer.output.net.1.weight',
                    'module.model.final_layer.output.net.1.bias'
                ]

                for name, param in checkpoint['model_state_dict'].items():
                    if 'model' not in name:
                        self.keys_to_remove.append(name)
                for key in self.keys_to_remove:
                    if key in checkpoint['model_state_dict']:
                        del checkpoint['model_state_dict'][key]

                # print(checkpoint['model_state_dict'].keys())

                self.diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)


                for name, param in self.diffusion.named_parameters():
                    if name not in self.keys_to_remove:
                        param.requires_grad = False



        if self.mode == "train":
            self.train_sampler, self.train_data, self.valid_data, self.test_data = self.datamodule.prepare_dataloaders(self.mode)
            if dist.get_rank() == 0:
                wandb.init(project=self.config_log.wandb_pj_name, name=self.config_log.exp_name)
                config_save_path = os.path.join(self.train_save_dir, "train_config.yaml")
                OmegaConf.save(config=self.config, f=config_save_path)
                wdir = os.path.join(self.train_save_dir, "weights")
                os.makedirs(wdir, exist_ok=True)
        elif self.mode == "inference":
            config_save_path = os.path.join(self.inference_save_dir, "inference_config.yaml")
            OmegaConf.save(config=self.config, f=config_save_path)
            pass
        else:
            raise ValueError("Invalid mode")


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.diffusion.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            scheduler = LambdaLinearScheduler(**self.config_scheduler)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=scheduler.schedule)
            return opt, scheduler
        return opt


    def eval(self):
        self.diffusion.eval()


    def train(self):
        self.diffusion.train()


    def train_loop(self):
        self.train()

        global_step = 0
        audio_feature_dim = self.config_model.mfcc_dim * 2

        for epoch in range(1, self.epochs + 1):
            # train
            self.train()
            self.train_sampler.set_epoch(epoch)
            epoch_train_loss_dict = {}
            epoch_train_loss, epoch_train_loss_vlb = 0, 0

            if self.task_mode == "transfer_learning":
                if self.need_frozen_step >= 0 and global_step >= self.need_frozen_step:
                    print(f"Unfreezing model parameters at step {global_step}...")
                    for name, param in self.diffusion.named_parameters():
                        if name not in self.keys_to_remove:
                            param.requires_grad = True

            for step, batch in enumerate(tqdm(self.train_data) if dist.get_rank() == 0 else self.train_data):
                motion_data = batch['motion_data'].to(self.local_rank)
                bsz, seq_len = motion_data.shape[0], motion_data.shape[1]
                empty_noise = torch.randn((bsz, seq_len, audio_feature_dim), device=self.device)
                cond_text = batch['cond_text'].to(self.local_rank)
                state = torch.ones((bsz, seq_len)).to(self.local_rank)

                loss, loss_dict = self.diffusion(motion_data, empty_noise, state, cond_text)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=0.5)
                self.opt.step()
                if self.use_scheduler:
                    self.scheduler.step()
                    lr = self.opt.param_groups[0]['lr']
                    loss_dict['train/lr'] = lr

                global_step += 1

                # ema update and train loss update only on main
                # loss_value, vlb_loss_value = loss_dict['train/loss'], loss_dict['train/loss_vlb']
                # dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)
                # dist.all_reduce(vlb_loss_value, op=dist.ReduceOp.AVG)
                # loss_dict['train/loss'] = loss_value
                # loss_dict['train/loss_vlb'] = vlb_loss_value
                if dist.get_rank() == 0:
                    epoch_train_loss += loss_dict['train/loss'].item()
                    epoch_train_loss_vlb += loss_dict['train/loss_vlb'].item()
                    wandb.log(loss_dict)

                # break

            if dist.get_rank() == 0:
                epoch_train_loss_dict['train_epoch/loss'] = epoch_train_loss / len(self.train_data)
                epoch_train_loss_dict['train_epoch/loss_vlb'] = epoch_train_loss_vlb / len(self.train_data)
                epoch_train_loss_dict['train/epoch'] = epoch
                wandb.log(epoch_train_loss_dict)

            # validation

            torch.cuda.empty_cache()
            if epoch % self.valid_interval == 0:
                self.eval()
                with torch.no_grad():
                    epoch_val_loss_dict = {}
                    epoch_val_loss, epoch_val_loss_vlb = 0, 0

                    for step, batch in enumerate(tqdm(self.valid_data) if dist.get_rank() == 0 else self.valid_data):
                        motion_data = batch['motion_data'].to(self.local_rank)
                        bsz, seq_len = motion_data.shape[0], motion_data.shape[1]
                        empty_noise_val = torch.randn((bsz, seq_len, audio_feature_dim), device=self.device)
                        cond_text = batch['cond_text'].to(self.local_rank)
                        state = torch.ones((bsz, seq_len)).to(self.local_rank)
                        loss, loss_dict = self.diffusion(motion_data, empty_noise_val, state, cond_text)
                        if torch.isnan(loss).any():
                            raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                        if dist.get_rank() == 0:
                            epoch_val_loss += loss_dict['val/loss'].item()
                            epoch_val_loss_vlb += loss_dict['val/loss_vlb'].item()

                    if dist.get_rank() == 0:
                        epoch_val_loss_dict['val_epoch/loss'] = epoch_val_loss / len(self.valid_data)
                        epoch_val_loss_dict['val_epoch/loss_vlb'] = epoch_val_loss_vlb / len(self.valid_data)
                        epoch_val_loss_dict['val/epoch'] = epoch
                        wandb.log(epoch_val_loss_dict)

            # save_model
            if epoch % self.save_interval == 0 and epoch >= self.start_save_point:

                if self.config_trainer.method == 'ddim' or self.config_trainer.method == 'diffusers':
                    self.render_sample_double_g1_text(
                        inference_sovler='DDIM_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_DDIM_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=1.
                    )
                elif self.config_trainer.method == 'diffusers':
                    self.render_sample_double_g1_text(
                        inference_sovler='UniPC_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_UniPC_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=20,
                        guidance_weight=3,
                        eta=1.
                    )
                elif self.config_trainer.method == 'flow_matching':
                    self.render_sample_double_g1_text(
                        inference_sovler='flow_matching_edge',
                        test_loaders=[self.valid_data],
                        test_file_save_names=[["val_sample_fm_epoch_{}".format(epoch)]],
                        output_scaler=self.output_scaler,
                        data_pipe=self.data_pipe,
                        fps=self.config_data.framerate,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=1.
                    )
                else:
                    raise ValueError("Invalid training method for ControlNet.")

                if dist.get_rank() == 0:
                    save_ckpt_path = os.path.join(self.train_save_dir, "weights", f"model_{epoch}.pt")
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.diffusion.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }
                    # torch.save(self.model.state_dict(), save_ckpt_path)
                    torch.save(checkpoint, save_ckpt_path)
                    print(f"Model saved at {save_ckpt_path}")

        if dist.get_rank() == 0:
            wandb.run.finish()

        cleanup()

    def render_sample_double_g1(self,
                      inference_sovler,
                      test_loaders,
                      test_file_save_names,
                      output_scaler,
                      data_pipe,
                      fps,
                      sample_steps=50,
                      guidance_weight=3,
                      eta=0.):
        self.eval()
        assert self.task_mode in ["interaction_3", "interaction_4", "interaction_4_cfg", "interaction_4_cfg_2", "interaction_5_cfg", "interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        print("is_cfg:", is_cfg)
        with torch.no_grad():
            for i in range(len(test_loaders)):
                test_loader = test_loaders[i]
                test_audio_name = test_file_save_names[i]
                print(f"Rendering {test_audio_name}...")
                for step, batch in enumerate(tqdm(test_loader) if dist.get_rank() == 0 else test_loader):
                    cond = batch['cond_1'].to(self.local_rank)

                    object_motion = None
                    if self.task_mode not in ["interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                        object_motion = batch['object_motion'].to(self.local_rank)

                    state = None
                    if self.task_mode in ["interaction_5_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]:
                        state = batch['state_1'].to(self.local_rank)
                    print(cond.shape)

                    bs, _, _ = cond.shape
                    shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

                    if inference_sovler == 'DDPM':
                        motion = self.diffusion.module.p_sample_loop(cond, object_motion, state, shape, return_intermediates=False)
                    elif inference_sovler == 'DDIM_edge':
                        motion = self.diffusion.module.ddim_sample_loop(cond, object_motion, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)

                    motion = motion.cpu().numpy()
                    motion = inv_standardize(motion, output_scaler)

                    bs_m, len_m, dim_m = motion.shape
                    assert len_m % 2 == 0
                    half_len_m = len_m // 2

                    if self.task_mode == "single_no_cond":
                        combine_motion = motion
                    else:
                        combine_motion = np.zeros((bs_m + 1, half_len_m, dim_m))
                        combine_motion[:-1] = motion[:, :half_len_m]
                        combine_motion[-1] = motion[-1, half_len_m:]

                    combine_motion = combine_motion.reshape(-1, dim_m)

                    if self.mode == "train":
                        bvh_save_path = os.path.join(self.train_save_dir, test_audio_name[0] + ".bvh")
                    else:
                        bvh_save_path = os.path.join(self.inference_save_dir, test_audio_name[0] + ".bvh")
                    print(f"combine_motion.shape: {combine_motion.shape}", f"bvh_save_path: {bvh_save_path}")
                    to_bvh(combine_motion, bvh_save_path, data_pipe, fps)

    def render_sample_double_g1_text(self,
                      inference_sovler,
                      test_loaders,
                      test_file_save_names,
                      output_scaler,
                      data_pipe,
                      fps,
                      sample_steps=50,
                      guidance_weight=3,
                      eta=0.):
        self.eval()
        # assert self.task_mode in ["interaction_3", "interaction_4", "interaction_4_cfg", "interaction_4_cfg_2", "interaction_5_cfg", "interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc"]
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        print("is_cfg:", is_cfg)
        with torch.no_grad():
            for i in range(len(test_loaders)):
                test_loader = test_loaders[i]
                test_audio_name = test_file_save_names[i]
                print(f"Rendering {test_audio_name}...")
                for step, batch in enumerate(tqdm(test_loader) if dist.get_rank() == 0 else test_loader):
                    bsz, seq_len = batch['motion_data'].shape[0], batch['motion_data'].shape[1]
                    empty_noise = torch.randn((bsz, seq_len, self.config_model.mfcc_dim * 2), device=self.device)
                    cond_text = batch['cond_text'].to(self.local_rank)
                    state = torch.ones((bsz, seq_len)).to(self.local_rank)

                    shape = (bsz, self.config_model.window_size_max_len, self.config_model.motion_dim)

                    if inference_sovler == 'DDIM_edge':
                        motion = self.diffusion.module.ddim_sample_loop_text(empty_noise, cond_text, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
                    elif inference_sovler == 'flow_matching_edge':
                        motion = self.diffusion.module.fm_sample_loop_text(empty_noise, cond_text, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
                    else:
                        raise NotImplementedError("Only DDIM and flow matching are implemented for text-conditioned generation.")

                    motion = motion.cpu().numpy()
                    motion = inv_standardize(motion, output_scaler)

                    bs_m, len_m, dim_m = motion.shape

                    combine_motion = motion

                    combine_motion = combine_motion.reshape(-1, dim_m)

                    bvh_save_path = os.path.join(self.train_save_dir, test_audio_name[0] + ".bvh")
                    print(f"combine_motion.shape: {combine_motion.shape}", f"bvh_save_path: {bvh_save_path}")
                    to_bvh(combine_motion, bvh_save_path, data_pipe, fps)

    def render_sample_streaming_g1(self,
                      inference_sovler,
                      cond,
                      previous_motion_last_half=None,
                      cache_length=0,
                      sample_steps=50,
                      guidance_weight=2,
                      eta=1.):
        self.eval()
        assert self.task_mode in ["controlnet_interaction_7_cfg_mfcc"]
        # is_cfg = False
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        print("is_cfg:", is_cfg)
        with torch.no_grad():
            bs, _, _ = cond.shape
            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            if inference_sovler == 'DDIM_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'DDIM_edge_w_prior':
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge_w_prior':
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.fm_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge_w_prior':
                motion, cache = self.diffusion.module.fm_sample_loop_streaming_controlnet(cond, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)

            motion = motion[:, cache_length:, :]  # remove the padding frames
            cache = cache[:, cache_length:, :]
            return motion, cache

    def render_sample_streaming_g1_text(self,
                      inference_sovler,
                      cond,
                      cond_text,
                      previous_motion_last_half=None,
                      cache_length=0,
                      sample_steps=50,
                      guidance_weight=2,
                      eta=1.):
        self.eval()
        assert self.task_mode in ["controlnet_interaction_7_cfg_mfcc"]
        # is_cfg = False
        is_cfg = False
        # if 'cfg' in self.task_mode:
            # is_cfg = False
        print("is_cfg:", is_cfg)
        with torch.no_grad():
            bs, _, _ = cond.shape
            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            if inference_sovler == 'DDIM_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'DDIM_edge_w_prior':
                motion, cache = self.diffusion.module.ddim_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'UniPC_edge_w_prior':
                motion, cache = self.diffusion.module.unipc_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge':
                cond_padding = torch.zeros((bs, cache_length, cond.shape[2]), device=cond.device)
                cond = torch.cat((cond_padding, cond), dim=1)
                motion, cache = self.diffusion.module.fm_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)
            elif inference_sovler == 'flow_matching_edge_w_prior':
                motion, cache = self.diffusion.module.fm_sample_loop_streaming_text(cond, cond_text, shape, previous_motion_last_half, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=True)

            motion = motion[:, cache_length:, :]  # remove the padding frames
            cache = cache[:, cache_length:, :]
            return motion, cache

    def render_sample_text(self,
                      inference_sovler,
                      input_text_feats,
                      sample_steps=50,
                      guidance_weight=3,
                      eta=0.):
        self.eval()
        is_cfg = False
        with torch.no_grad():
            bsz, seq_len = input_text_feats.shape[0], self.config_model.window_size_max_len
            empty_noise = torch.randn((bsz, seq_len, self.config_model.mfcc_dim * 2), device=self.device)
            state = torch.ones((bsz, seq_len)).to(self.local_rank)
            shape = (bsz, seq_len, self.config_model.motion_dim)

            if inference_sovler == 'DDIM_edge':
                motion = self.diffusion.module.ddim_sample_loop_text(empty_noise, input_text_feats, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
            elif inference_sovler == 'flow_matching_edge':
                motion = self.diffusion.module.fm_sample_loop_text(empty_noise, input_text_feats, state, shape, sample_steps, guidance_weight, eta, is_cfg, return_intermediates=False)
            else:
                raise NotImplementedError("Only DDIM and flow matching are implemented for text-conditioned generation.")

            return motion


class MixTask_trainer():
    def __init__(
        self,
        yaml_file_path: str = "./configs/legacy/triplet.yaml",
        inference_save_dir: str = ''
    ):

        self.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        self.local_rank = int(os.environ['LOCAL_RANK']) if "LOCAL_RANK" in os.environ else 0
        self.world_size = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
        setup(self.local_rank, self.world_size)

        self.config = OmegaConf.load(yaml_file_path)
        self.task_mode = self.config.task_mode
        self.config_trainer = self.config.trainer

        if self.config_trainer.method == 'flow_matching':
            print("Using flow matching...")
            from ddpm.flow_matching_triplet import Gesture_Diffusion_double
        elif self.config_trainer.method == 'ddim':
            print("Using DDIM...")
            from ddpm.diffusion_triplet import Gesture_Diffusion_double
        elif self.config_trainer.method == 'diffusers':
            print("Using Diffusers...")
            raise NotImplementedError("Diffusers is not implemented yet.")
        else:
            raise ValueError("Invalid training method")

        if self.config_trainer.new_audio_encoder:
            print("Using new audio encoder...")
            raise NotImplementedError("New audio encoder is not implemented yet.")
        else:
            print("Using original audio encoder...")
            from model.model_for_interaction_7_cfg_mfcc import Mix_DiT_interaction_7_cfg_mfcc

        self.config_data = self.config.data
        self.config_model = self.config.model
        self.config_ddpm = self.config.ddpm
        self.config_scheduler = self.config.scheduler
        self.config_log = self.config.logger

        self.use_scheduler = self.config_trainer.use_scheduler
        self.learning_rate = self.config_trainer.learning_rate
        self.mode = self.config_trainer.mode
        self.epochs = self.config_trainer.epochs
        self.valid_interval = self.config_trainer.valid_interval
        self.save_interval = self.config_trainer.save_interval
        self.start_save_point = self.config_trainer.start_save_point
        self.checkpoint_path = self.config_trainer.checkpoint_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_trainer.matmul_precision == 'high':
            torch.set_float32_matmul_precision('high')
        elif self.config_trainer.matmul_precision == 'medium':
            torch.set_float32_matmul_precision('medium')
        else:
            raise ValueError("Invalid matmul precision setting")


        if self.mode == "train":

            self.train_save_dir = None
            self.current_time = None
            time_tensor = torch.zeros(100, dtype=torch.uint8).to(self.device)
            if dist.get_rank() == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                encoded_time = cur_time.encode('utf-8')
                time_tensor[:len(encoded_time)] = torch.tensor(list(encoded_time), dtype=torch.uint8).to(self.device)

            dist.broadcast(time_tensor, src=0)
            self.current_time = bytes(time_tensor.tolist()).decode('utf-8').rstrip('\x00')
            self.train_save_dir = os.path.join(self.config_trainer.save_dir, self.current_time)
            os.makedirs(self.train_save_dir, exist_ok=True)
        elif self.mode == "inference":
            assert inference_save_dir is not None
            self.inference_save_dir = inference_save_dir or os.getenv("MOTION_OUTPUT_DIR", os.path.join("runtime", "motion"))
            os.makedirs(self.inference_save_dir, exist_ok=True)

        self.set_seed(self.config_trainer.seed)

        self.output_scaler = jl.load(os.path.join(self.config.data.data_root, 'output_scaler.sav'))
        self.data_pipe = jl.load(os.path.join(self.config.data.data_root, 'data_pipe_' + str(self.config.data.framerate) + 'fps.sav'))


        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True
            if self.config_trainer.matmul_precision == 'high':
                torch.set_float32_matmul_precision('high')
            elif self.config_trainer.matmul_precision == 'medium':
                torch.set_float32_matmul_precision('medium')
            else:
                raise ValueError("Invalid matmul precision setting")


            torch.backends.cudnn.benchmark = True


            try:
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_mem_efficient=True,
                    enable_math=False,
                )
            except Exception:
                pass

        if self.task_mode == "interaction_7_cfg_mfcc":
            self.datamodule = GesDiffDataModule_Unified(**self.config_data, local_rank=self.local_rank, world_size=self.world_size)
            model = Mix_DiT_interaction_7_cfg_mfcc(**self.config_model).to(self.device)
        else:
            raise ValueError("Invalid task_mode")

        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

        self.model = model.to(self.local_rank)
        if self.task_mode in ["interaction_3", "interaction_4", "interaction_4_cfg", "interaction_4_cfg_2", "interaction_5_cfg", "interaction_6_cfg", "interaction_7_cfg", "interaction_7_cfg_mfcc", "interaction_7_cfg_single"]:
            self.diffusion = Gesture_Diffusion_double(model=self.model, **self.config_ddpm).to(self.local_rank)
        else:
            raise NotImplementedError("Only double diffusion is implemented.")
            # self.diffusion = Gesture_Diffusion(model=self.model, **self.config_ddpm).to(self.local_rank)
        self.diffusion = DDP(self.diffusion, device_ids=[self.local_rank], find_unused_parameters=True)

        if self.use_scheduler:
            self.opt, self.scheduler = self.configure_optimizers()
        else:
            self.opt = self.configure_optimizers()


        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            # self.diffusion.load_state_dict(checkpoint['model_state_dict'])

            def fix_model_weights(state_dict):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if '_orig_mod.' in key:
                        key = key.replace('_orig_mod.', '')

                    new_state_dict[key] = value

                return new_state_dict

            fixed_weights = fix_model_weights(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            self.diffusion.load_state_dict(fixed_weights, strict=False)
            # self.diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if self.mode == "train":
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.use_scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        elif self.task_mode == "transfer_learning":
            self.transfer_learning_checkpoint_path = self.config_trainer.transfer_learning_checkpoint_path
            self.need_frozen_step = self.config_trainer.need_frozen_step
            if self.mode == "train":
                print(f"Loading transfer learning checkpoint from {self.transfer_learning_checkpoint_path}")
                checkpoint = torch.load(self.transfer_learning_checkpoint_path, map_location=self.device, weights_only=False)
                self.keys_to_remove = [
                    'module.model.motion_encoder.net.1.weight',
                    'module.model.motion_encoder.net.1.bias',
                    'module.model.motion_decoder.net.1.weight',
                    'module.model.motion_decoder.net.1.bias',
                    'module.model.final_layer.output.net.1.weight',
                    'module.model.final_layer.output.net.1.bias'
                ]

                for name, param in checkpoint['model_state_dict'].items():
                    if 'model' not in name:
                        self.keys_to_remove.append(name)
                for key in self.keys_to_remove:
                    if key in checkpoint['model_state_dict']:
                        del checkpoint['model_state_dict'][key]

                # print(checkpoint['model_state_dict'].keys())

                self.diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)


                for name, param in self.diffusion.named_parameters():
                    if name not in self.keys_to_remove:
                        param.requires_grad = False



        if self.mode == "train":
            self.train_sampler, self.train_data, self.valid_data = self.datamodule.prepare_dataloaders(self.mode)
            if dist.get_rank() == 0:
                wandb.init(project=self.config_log.wandb_pj_name, name=self.config_log.exp_name)
                config_save_path = os.path.join(self.train_save_dir, "train_config.yaml")
                OmegaConf.save(config=self.config, f=config_save_path)
                wdir = os.path.join(self.train_save_dir, "weights")
                os.makedirs(wdir, exist_ok=True)
        elif self.mode == "inference":
            config_save_path = os.path.join(self.inference_save_dir, "inference_config.yaml")
            OmegaConf.save(config=self.config, f=config_save_path)
            pass
        else:
            raise ValueError("Invalid mode")

        # if dist.get_rank() == 0:
        #     pass


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.diffusion.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            scheduler = LambdaLinearScheduler(**self.config_scheduler)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=scheduler.schedule)
            return opt, scheduler
        return opt


    def eval(self):
        self.diffusion.eval()


    def train(self):
        self.diffusion.train()


    def train_loop(self):
        self.train()

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            # train
            self.train()
            self.train_sampler.set_epoch(epoch)
            epoch_train_loss_dict = {}
            epoch_train_loss = 0

            for step, batch in enumerate(tqdm(self.train_data) if dist.get_rank() == 0 else self.train_data):
                motion_data = batch['motion'].to(self.local_rank) # [B, T, D]
                cond_audio = batch['audio'].to(self.local_rank)   # [B, T, D_a]
                cond_text = batch['text'].to(self.local_rank)     # [B, 77, D_t]
                mask = batch['mask'].to(self.local_rank)          # [B, 3]
                # print(f"motion_data.shape: {motion_data.shape}, cond_audio.shape: {cond_audio.shape}, cond_text.shape: {cond_text.shape}, mask.shape: {mask.shape}")

                loss, loss_dict = self.diffusion(motion_data, cond_audio, cond_text, mask)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=0.5)
                self.opt.step()
                if self.use_scheduler:
                    self.scheduler.step()
                    lr = self.opt.param_groups[0]['lr']
                    loss_dict['train/lr'] = lr

                global_step += 1

                if dist.get_rank() == 0:
                    epoch_train_loss += loss.item()
                    wandb.log(loss_dict)

                # break

            # validation

            torch.cuda.empty_cache()
            if epoch % self.valid_interval == 0:
                self.eval()
                with torch.no_grad():
                    epoch_val_loss_dict = {}
                    epoch_val_loss = 0

                    for step, batch in enumerate(self.valid_data):
                        motion_data = batch['motion'].to(self.local_rank)
                        cond_audio = batch['audio'].to(self.local_rank)
                        cond_text = batch['text'].to(self.local_rank)
                        mask = batch['mask'].to(self.local_rank)

                        loss, loss_dict = self.diffusion(motion_data, cond_audio, cond_text, mask)
                        epoch_val_loss += loss.item()

                    if dist.get_rank() == 0:
                         wandb.log({'val/loss': epoch_val_loss / len(self.valid_data), 'epoch': epoch})

            # save_model
            if epoch % self.save_interval == 0 and epoch >= self.start_save_point:

                if dist.get_rank() == 0:
                    # Save Checkpoint
                    save_ckpt_path = os.path.join(self.train_save_dir, "weights", f"model_{epoch}.pt")
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.diffusion.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                    }
                    torch.save(checkpoint, save_ckpt_path)

                # --- Render Samples (Mixed Modality Test) ---
                have_render_audio = False
                have_render_text = False
                for step, batch in enumerate(self.valid_data):
                    motion_data = batch['motion'].to(self.local_rank)
                    cond_audio = batch['audio'].to(self.local_rank)
                    cond_text = batch['text'].to(self.local_rank)
                    mask = batch['mask'].to(self.local_rank)
                    if mask[0, 2] == 0 and not have_render_audio:
                        self.render_sample_mix(batch, modality='audio', name_prefix=f'val_epoch_{epoch}_audio')
                        have_render_audio = True
                    if mask[0, 1] == 0 and not have_render_text:
                        self.render_sample_mix(batch, modality='text', name_prefix=f'val_epoch_{epoch}_text')
                        have_render_text = True
                    if have_render_audio and have_render_text:
                        break

        if dist.get_rank() == 0:
            wandb.run.finish()

        cleanup()

    def render_sample_mix(self, batch, modality, name_prefix):
        """
        modality: 'audio' or 'text'
        """
        self.eval()
        with torch.no_grad():
            # Prepare conditions
            cond_audio = batch['audio'].to(self.local_rank)
            cond_text = batch['text'].to(self.local_rank)
            bs = cond_audio.shape[0]

            # Manually construct mask for inference testing
            # If testing Audio: mask = [1, 1, 0]
            # If testing Text:  mask = [1, 0, 1]
            if modality == 'audio':
                mask = torch.tensor([1, 1, 0], device=self.device, dtype=torch.float32).repeat(bs, 1)
                # Ensure text is zeroed out just in case
                cond_text = torch.zeros_like(cond_text)
            elif modality == 'text':
                mask = torch.tensor([1, 0, 1], device=self.device, dtype=torch.float32).repeat(bs, 1)
                # Ensure audio is zeroed out
                cond_audio = torch.zeros_like(cond_audio)

            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            motion = self.diffusion.module.ddim_sample_loop_mix(
                cond_audio, cond_text, mask, shape, sample_steps=50
            )

            motion = motion.cpu().numpy()
            motion = inv_standardize(motion, self.output_scaler)

            # Save first sample
            save_path = os.path.join(self.train_save_dir, f"{name_prefix}_sample0.bvh")
            to_bvh(motion[0], save_path, self.data_pipe, self.config_data.framerate)


    def render_sample_streaming_g1_mix(self,
                      inference_sovler,
                      cond_audio,
                      cond_text,
                      modality,
                      previous_motion_last_half=None,
                      cache_length=0,
                      sample_steps=50,
                      guidance_weight=2,
                      eta=1.):
        """
        Streaming Inference for Mix Model
        modality: 'audio' or 'text'
        """
        self.eval()
        # is_cfg is typically True for inference with guidance
        is_cfg = False

        with torch.no_grad():
            bs = cond_audio.shape[0]

            # 1. Construct Mask & Zero-out unused condition
            if modality == 'audio':
                mask = torch.tensor([1, 1, 0], device=self.device, dtype=torch.float32).repeat(bs, 1)
                cond_text = torch.zeros_like(cond_text)
            elif modality == 'text':
                mask = torch.tensor([1, 0, 1], device=self.device, dtype=torch.float32).repeat(bs, 1)
                cond_audio = torch.zeros_like(cond_audio)
            else:
                 raise ValueError(f"Unknown modality: {modality}")

            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            # 2. Handle Padding for Streaming Solvers

            if inference_sovler == 'DDIM_edge':
                # Pad Audio [B, T, D] -> [B, Cache+T, D]
                cond_audio_padding = torch.zeros((bs, cache_length, cond_audio.shape[2]), device=self.device)
                cond_audio = torch.cat((cond_audio_padding, cond_audio), dim=1)

                motion, cache = self.diffusion.module.ddim_sample_loop_streaming_mix(
                    cond_audio=cond_audio,
                    cond_text=cond_text,
                    mask=mask,
                    shape=shape,
                    previous_motion_last_half=previous_motion_last_half,
                    sample_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    eta=eta,
                    is_cfg=is_cfg,
                    return_intermediates=True
                )
            # Add other solvers if needed (e.g., flow_matching_edge)
            elif inference_sovler == 'DDIM_edge_w_prior':
                 motion, cache = self.diffusion.module.ddim_sample_loop_streaming_mix(
                    cond_audio=cond_audio,
                    cond_text=cond_text,
                    mask=mask,
                    shape=shape,
                    previous_motion_last_half=previous_motion_last_half,
                    sample_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    eta=eta,
                    is_cfg=is_cfg,
                    return_intermediates=True
                )
            elif inference_sovler == 'flow_matching_edge':
                # Pad Audio [B, T, D] -> [B, Cache+T, D]
                cond_audio_padding = torch.zeros((bs, cache_length, cond_audio.shape[2]), device=self.device)
                cond_audio = torch.cat((cond_audio_padding, cond_audio), dim=1)

                motion, cache = self.diffusion.module.fm_sample_loop_streaming_mix(
                    cond_audio=cond_audio,
                    cond_text=cond_text,
                    mask=mask,
                    shape=shape,
                    previous_motion_last_half=previous_motion_last_half,
                    sample_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    eta=eta,
                    is_cfg=is_cfg,
                    return_intermediates=True
                )
            elif inference_sovler == 'flow_matching_edge_w_prior':
                motion, cache = self.diffusion.module.fm_sample_loop_streaming_mix(
                    cond_audio=cond_audio,
                    cond_text=cond_text,
                    mask=mask,
                    shape=shape,
                    previous_motion_last_half=previous_motion_last_half,
                    sample_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    eta=eta,
                    is_cfg=is_cfg,
                    return_intermediates=True
                )
            else:
                raise NotImplementedError(f"Solver {inference_sovler} not supported for mix streaming.")

            # 3. Slice output to remove padding frames
            motion = motion[:, cache_length:, :]
            cache = cache[:, cache_length:, :]

            return motion, cache

    def render_sample_text(self, solver_name, input_text_feats, sample_steps, guidance_weight, eta):
        """
        modality: 'audio' or 'text'
        """
        self.eval()
        with torch.no_grad():
            cond_text = input_text_feats
            bs = input_text_feats.shape[0]

            # mask = torch.tensor([1, 0, 1], device=self.device, dtype=torch.float32).repeat(bs, 1)
            mask = torch.tensor([1, 1, 1], device=self.device, dtype=torch.float32).repeat(bs, 1)
            # Ensure audio is zeroed out
            # cond_audio = torch.zeros((bs, self.config_model.window_size_max_len, self.config_model.mfcc_dim * 2), device=self.device)
            cond_audio = torch.randn((bs, self.config_model.window_size_max_len, self.config_model.mfcc_dim * 2), device=self.device)

            shape = (bs, self.config_model.window_size_max_len, self.config_model.motion_dim)

            motion = self.diffusion.module.ddim_sample_loop_mix(
                cond_audio, cond_text, mask, shape, sample_steps=sample_steps, guidance_weight=guidance_weight, eta=eta
            )

            return motion
