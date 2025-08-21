from copy import deepcopy
from typing import Union, Tuple, List
import types
import torch
from einops import rearrange
import os 
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_warn_always(True)
import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
import torch, types
from monai.data.meta_tensor import MetaTensor
import timm.layers.pos_embed_sincos as pes
import timm.models.eva as timm_eva
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)    
    torch.backends.cuda.enable_mem_efficient_sdp(True)  
    torch.backends.cuda.enable_math_sdp(False)  
from einops import rearrange
import timm.layers.pos_embed_sincos as pes

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

#from torch import autocast
from torch.amp import autocast     # <-- instead of "from torch import autocast"

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.voco_architecture import VoCoArchitecture
from nnssl.training.loss.contrastive_loss import NTXentLoss
from nnssl.utilities.helpers import dummy_context

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnssl.ssl_data.dataloading.simclr_transform import SimCLRTransform
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA

torch.autograd.set_detect_anomaly(True)

from monai.transforms import (
    Compose,
    RandSpatialCrop,
    RandFlip,
    RandRotate,
    RandShiftIntensity,
    RandAdjustContrast,
    Resize,
    EnsureChannelFirst
)

import torch.distributed as dist
from monai.data.meta_tensor import MetaTensor


import types
from einops import rearrange
import torch

def primus_forward_tokens(self,x):
    x = self.down_projection(x)                          # (B, C, W, H, D)
    x = rearrange(x, "b c w h d -> b (w h d) c").contiguous()       # (B, N, C)
    x, _ = self.eva(x)                                   # no masking → keep_indices is None
    return x.contiguous()


def as_plain(x):
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    return x.to(memory_format=torch.contiguous_format).contiguous()

monai_transform = Compose([
            RandSpatialCrop(roi_size=(60, 60, 60), random_center=True, random_size=True),
            Resize(spatial_size=(160,160,160), mode='trilinear'),
            RandFlip( prob=0.5, spatial_axis=[2]),
            RandRotate(range_x=0.785, prob=0.5, mode='trilinear'),  # ~45 degrees
            RandShiftIntensity(offsets=0.5, prob=0.8),
            RandAdjustContrast(gamma=(0.5, 1.5), prob=0.8),
            ])

def batch_monai_transform(batch):
    return [monai_transform(img) for img in batch]

def monai_transform_adapter(**kwargs):
    data = kwargs["data"]
    aug1 = batch_monai_transform(data) # put it back
    aug2 = batch_monai_transform(data)
    return {
        "aug1": torch.stack(aug1),  # convert to tensor
        "aug2": torch.stack(aug2),
        **{k: v for k, v in kwargs.items() if k != "data"},  # keep other keys
    }

class SimCLRTrainer(AbstractBaseTrainer):
    """
    TODO:
    - implement data aug path for simclr [x]
        - check which standard transforms to keep [x] - went with default nnUNet transforms fow now
    - re-use VoCoArchitecture (seems like no change necessary here, double-check) [x]
    - implement train/val steps (loss returns loss, accuracy) -> maybe track acc. similar to pseudo dice in nnUNet [x] - not tracking yet
    - re-implement similar to VoCoTransform (need more sub-crops, and random crops in general) [x]
    - maybe force partial overlaps between crops [x]
    - clean up, test runs [x]

    Memory consumption & batch/s on 4090:
    - batch_size 4, num_crops_per_image 3, crop_size (64, 64, 64): 9.55 GB & 5.3 batches/s
    - batch_size 8, num_crops_per_image 3, crop_size (64, 64, 64): 15.4 GB & 2.9 batches/s
    - batch_size 16, num_crops_per_image 3, crop_size (64, 64, 64): 23.5 GB & 1.08 batches/s
    - batch_size 32, num_crops_per_image 2, crop_size (64, 64, 64): >24.5 GB (OoM)
    - batch_size 32, num_crops_per_image 1, crop_size (64, 64, 64): 19.3 GB & 2.2 batches/s
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (160, 160, 160),
        crop_size: tuple = (160, 160, 160),
        num_crops_per_image: int = 1,
        min_crop_overlap: float = 0.5,
    ):
        plan.configurations[configuration_name].patch_size = patch_size
        self.crop_size = crop_size

        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_crops_per_image = num_crops_per_image
        self.min_crop_overlap = min_crop_overlap

    def build_loss(self) -> nn.Module:
        """Implements the standard contrastive loss."""
        return NTXentLoss(
            batch_size=self.total_batch_size, #self.batch_size * self.num_crops_per_image,
            temperature=0.5,
            similarity_function="cosine",
            device=self.device,
        )

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:

        return monai_transform_adapter
        # tr_transforms = []

        # if do_dummy_2d_data_aug:
        #     raise NotImplementedError("We don't do dummy 2d aug here anymore. Data should be isotropic!")

        # # --------------------------- SimCLR Transformation --------------------------- #
        # # All train augmentations are moved to the SimCLR Transform class.

        # tr_transforms.append(
        #     SimCLRTransform(
        #         crop_size=self.crop_size,
        #         aug="train",
        #         crop_count_per_image=self.num_crops_per_image,
        #         min_overlap_ratio=self.min_crop_overlap,
        #         data_key="data",
        #     )
        # )
        # # From here on out we are working with reference and overlapping crops!

        # tr_transforms.append(NumpyToTensor(["all_crops"], "float"))
        # tr_transforms = Compose(tr_transforms)
        # return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []

        # --------------------------- VoCo Transformation --------------------------- #
        val_transforms.append(
            SimCLRTransform(
                crop_size=self.crop_size,
                aug="none",
                crop_count_per_image=self.num_crops_per_image,
                min_overlap_ratio=self.min_crop_overlap,
                data_key="data",
            )
        )

        val_transforms.append(NumpyToTensor(["all_crops"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        # We don't do non-90 degree rotations for the VoCo Trainer.
        dl_tr, dl_val = self.get_plain_dataloaders(patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val



    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
       # print(num_input_channels)
      #  print(num_output_channels)
        encoder = get_network_by_name(
            config_plan,
            "PrimusM",
            num_input_channels,
            num_output_channels,
            encoder_only=False,
        )
      #  print(encoder)
        encoder.forward = types.MethodType(primus_forward_tokens, encoder)

        # Turns out VoCoArchitecture can be used for SimCLR purpose here.
        architecture = VoCoArchitecture(encoder, [encoder.down_projection.proj.out_channels], vit=True)

        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.crop_size

        # adapt_plan = AdaptationPlan(
        #     architecture_plans=ArchitecturePlans("PrimusM"),
        #     pretrain_plan=plan,
        #     recommended_downstream_patchsize=self.recommended_downstream_patchsize,
        #     pretrain_num_input_channels=1,
        #     key_to_encoder="encoder.stages",
        #     key_to_stem="encoder.stem",
        #     keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        # )

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.eva",
            key_to_stem="encoder.down_projection",
            keys_to_in_proj=("projection.proj",),
            key_to_lpe="encoder.eva.pos_embed",
        )
        return architecture, adapt_plan

    def gather_embeddings(self, z):
        """Gather embeddings across GPUs while preserving gradients for the local rank."""
        if not self.is_ddp:
            return z

        world_size = dist.get_world_size()
        z_list = [torch.zeros_like(z) for _ in range(world_size)]
        dist.all_gather(z_list, z.detach())  # gather copies (detached)

        # Replace the local rank's entry with the original tensor (with grad)
        z_list[dist.get_rank()] = z
        return torch.cat(z_list, dim=0)


    def train_step(self, batch: Tuple[dict, dict]) -> dict:

      #  all_crops = batch["all_crops"]
      #  NREF = batch["reference_crop_index"]
        aug1 = as_plain(batch["aug1"]).to(self.device, non_blocking=True) #batch["aug1"].to(self.device, non_blocking=True)
        aug2 = as_plain(batch["aug2"]).to(self.device, non_blocking=True)
      #  print(aug1.shape, 'aug1')

        # all_crops = all_crops.to(self.device, non_blocking=True)

        # if torch.isnan(all_crops).any():
        #     print("NaN values found in input data!")
        # if torch.isinf(all_crops).any():
        #     print("Infinity values found in input data!")

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
       # with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
        with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            # all_crop_embeddings = self.network(all_crops)
            # if torch.isnan(all_crop_embeddings).any():
            #     print("NaN values found in embeddings!")
            all_crop_embeddings1 = self.network(aug1)
           # print(all_crop_embeddings1.shape, 'crop embeds 1')
            all_crop_embeddings2 = self.network(aug2)
            if torch.isnan(all_crop_embeddings1).any() or torch.isnan(all_crop_embeddings2).any():
                print("NaN values found in embeddings!")

            # This rearrange below isn't necessary, but would come in handy when doing more involved contrastive tasks.
            # z_i_embeddings = rearrange(
            #     all_crop_embeddings[:NREF], "(b NREF) c -> b NREF c", b=self.batch_size
            # )
            # z_j_embeddings = rearrange(
            #     all_crop_embeddings[NREF:], "(b NREF) c -> b NREF c", b=self.batch_size
            # )

            # Normalize prior to contrastive loss
          #  z_i_embeddings = nn.functional.normalize(all_crop_embeddings[:NREF], dim=1)
            #z_j_embeddings = nn.functional.normalize(all_crop_embeddings[NREF:], dim=1)
            z_i_embeddings = nn.functional.normalize(all_crop_embeddings1, dim=1)
            z_j_embeddings = nn.functional.normalize(all_crop_embeddings2, dim=1)

           # print("z_i before gather:", z_i_embeddings.shape)
           # print("z_j before gather:", z_j_embeddings.shape)
            z_i_embeddings = self.gather_embeddings(z_i_embeddings)
            z_j_embeddings = self.gather_embeddings(z_j_embeddings)
           # print("z_i after gather :", z_i_embeddings.shape)
           # print("z_j after gather :", z_j_embeddings.shape)

            #print(z_i_embeddings.shape)

            # del data
            l, acc = self.loss(z_i_embeddings, z_j_embeddings)

        if self.grad_scaler is not None:
            if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
                with torch.autograd.set_detect_anomaly(True):
                    self.grad_scaler.scale(l).backward()
            else:
                self.grad_scaler.scale(l).backward()
                    #self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)


            def report_no_grad_params(model):
                missing = []
                total = 0
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        total += 1
                        if p.grad is None:
                            missing.append(n)
                print(f"[grad-check] missing grads: {len(missing)} / {total}")
                for n in missing[:40]:  # don't spam
                    print("  -", n)

            #if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
            #    report_no_grad_params(self.network)

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.optimizer.step()

        # print(f"Train loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NREF = batch["reference_crop_index"]

        all_crops = all_crops.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                all_crop_embeddings = self.network(all_crops)
                # This rearrange below isn't necessary, but would come in handy when doing more involved contrastive tasks.
                # z_i_embeddings = rearrange(
                #     all_crop_embeddings[:NREF],
                #     "(b NREF) c -> b NREF c",
                #     b=self.batch_size,
                # )
                # z_j_embeddings = rearrange(
                #     all_crop_embeddings[NREF:],
                #     "(b NREF) c -> b NREF c",
                #     b=self.batch_size,
                # )

                # Normalize prior to contrastive loss
                z_i_embeddings = nn.functional.normalize(all_crop_embeddings[:NREF], dim=1)
                z_j_embeddings = nn.functional.normalize(all_crop_embeddings[NREF:], dim=1)

                # del data
                l, acc = self.loss(z_i_embeddings, z_j_embeddings)
                # print(f"Val loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class SimCLRTrainer_BS6(SimCLRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 6


class SimCLRTrainer_BS32(SimCLRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 32
