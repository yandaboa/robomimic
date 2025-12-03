"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

# import GPT2 Transformer based policy networks
import transformers
from transformers import GPT2Config, GPT2Model


@register_algo_factory_func("dit_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    assert algo_config.transformer.enabled, "Enable transformer in algo config for DIT Policy"
    return DiTPolicyUNet, {}


class DiTPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
            return_dict=False
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        # create projection layer
        encoded_dim = obs_encoder.output_shape()[0]
        projection_layer = nn.Linear(
            in_features=encoded_dim,
            out_features=self.algo_config.transformer.embed_dim,
        )
        self.context_length = self.algo_config.transformer.context_length
        # create GPT2 transformer
        transformer_config = GPT2Config(
            vocab_size=1,  # we don't use vocab embeddings
            n_positions=self.context_length,
            n_embd=self.algo_config.transformer.embed_dim,
            n_layer=self.algo_config.transformer.num_layers,
            n_head=self.algo_config.transformer.num_heads,
            resid_pdrop=self.algo_config.transformer.block_output_dropout,
            attn_pdrop=self.algo_config.transformer.attn_dropout,
            embd_pdrop=self.algo_config.transformer.emb_dropout,
        )
        transformer = GPT2Model(transformer_config)
        
        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.algo_config.transformer.embed_dim,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net,
                "transformer_encoder": transformer,
                "projection_layer": projection_layer,
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def chunk_actions(self, actions, attention_mask, dones):
        """
        Convert actions from shape [B, T, Da] to [B, T, Tp, Da]
        where Tp is the prediction horizon.

        Args:
            actions (torch.Tensor): actions of shape [B, T, Da]
            attention_mask (torch.Tensor): attention mask of shape [B, T]
            dones (torch.Tensor): dones of shape [B, T]
        Returns:
            chunked_actions (torch.Tensor): actions of shape [B, T, Tp, Da]
        """
        Tp = self.algo_config.horizon.prediction_horizon
        B, T, Da = actions.shape
        base  = torch.arange(Tp, device=actions.device).view(1,1,Tp)   # (1,1,Tp)
        start = torch.arange(T,  device=actions.device).view(1,T,1)    # (1,T,1)
        idxs   = (start + base).expand(B, -1, -1) # (B,T,Tp)
        lengths = attention_mask.sum(dim=1).long()  # [B]

        # Compute per-timestep episode boundaries
        # done_positions[b, t] = index of NEXT done at or after t (else last step)
        done_mask = dones.bool()                                      # (B,T)
        done_idx = torch.arange(T, device=actions.device).view(1,T)   # (1,T)
        done_idx = done_idx.repeat(B,1).unsqueeze(-1) * done_mask                   # (B,T)
        # Reverse so we propagate backwards
        rev_done = torch.flip(done_idx, dims=(1,))
        # cumulative max spreads the last nonzero "done index" forward
        rev_cum, _ = torch.cummax(rev_done, dim=1)
        next_done = torch.flip(rev_cum, dims=(1,))

        # For sequences with no done at all: fix by setting end=T-1
        # no_done_mask = next_done.eq(0).all(dim=1, keepdim=True)       # (B,1)  We need to get dones in here properly...
        next_done = next_done.masked_fill(next_done.eq(0), T-1)

        # Zero-out actions that cross the done boundary
        boundary = next_done.unsqueeze(-1)                            # (B,T,1)
        mask = idxs > boundary                                        # (B,T,Tp)

        inp = actions.unsqueeze(2).expand(B, T, Tp, Da)
        chunked_actions = inp.gather(dim=1, index=idxs.unsqueeze(-1).expand(B, T, Tp, Da))
        chunked_actions = chunked_actions.masked_fill(mask.unsqueeze(-1), 0)
        return chunked_actions

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        # To = self.algo_config.horizon.observation_horizon
        # Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        # input batch should contain observations, actions, attention masks


        input_batch = dict()
        input_batch["obs"] = batch["obs"] # B x T x (xyz)
        input_batch["actions"] = self.chunk_actions(
            batch["actions"], batch["attention_mask"], batch["dones"])
        input_batch["attention_mask"] = batch["attention_mask"]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B,T,Tp1,_ = batch["actions"].shape
        assert Tp1 == Tp
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiTPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            if torch.cuda.device_count() > 1:
                obs_features = TensorUtils.time_distributed(
                    inputs, self.nets.module["policy"]["obs_encoder"], inputs_as_kwargs=True)
            else:
                obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            # pass through projection layer
            if torch.cuda.device_count() > 1:
                obs_features = self.nets["policy"].module.projection_layer(obs_features)
            else:
                obs_features = self.nets["policy"]["projection_layer"](obs_features)

            # pass through transformer
            attention_mask = batch["attention_mask"]
            if torch.cuda.device_count() > 1:
                transformer_outputs = self.nets.module["policy"]["transformer_encoder"](
                    inputs_embeds=obs_features,
                    attention_mask=attention_mask
                )
            else:
                transformer_outputs = self.nets["policy"]["transformer_encoder"](
                    inputs_embeds=obs_features,
                    attention_mask=attention_mask
                )
            transformer_features = transformer_outputs.last_hidden_state  # [B, T, D]
            assert transformer_features.ndim == 3  # [B, T, D]

            transformer_features = transformer_features.reshape(B*T, -1)  # [B*T, D]
            obs_cond = transformer_features  # global conditioning for diffusion
            actions = actions.reshape(B*T, Tp, action_dim)  # [B*T, Tp, Da]

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B*T,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps) # [B*T, Tp, Da]
            
            # predict the noise residual
            if torch.cuda.device_count() > 1:
                noise_pred = self.nets.module["policy"]["noise_pred_net"](
                    noisy_actions, timesteps, global_cond=obs_cond) # [B*T, Tp, Da]
            else:
                noise_pred = self.nets["policy"]["noise_pred_net"](
                    noisy_actions, timesteps, global_cond=obs_cond) # [B*T, Tp, Da]
            
            if "loss_mask" in batch:
                loss_mask = batch["loss_mask"]
            else:
                loss_mask = attention_mask
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise, reduction="none") # [B*T, Tp, Da]
            loss = loss.mean(dim=-1)  # [B*T, Tp]
            loss = loss.mean(dim=-1)  # [B*T]
            loss = loss * loss_mask.reshape(B*T)  # [B*T]
            loss = loss.sum() / loss_mask.sum()  # scalar
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiTPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=self.context_length)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue.append(obs_dict)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory()
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
    
    def get_isaac_formatted_action(self, num_actions, obs_dict, env_t):
        # obs: dict, {key: torch.Tensor of shape [B, T, Do]}
        # num_actions: int
        # env_t: torch.Tensor of shape [B], the time step of each environment
        assert num_actions <= self.algo_config.horizon.prediction_horizon, "num_actions must be less than or equal to the prediction horizon"
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        obs_features = TensorUtils.time_distributed(obs_dict, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # pass through projection layer
        obs_features = nets["policy"]["projection_layer"](obs_features)
        assert obs_features.ndim == 3  # [B, T, D]

        # pass through transformer
        attention_mask = torch.arange(obs_features.shape[1], device=obs_features.device).unsqueeze(0) <= env_t.unsqueeze(1) # [B, T]
        transformer_outputs = nets["policy"]["transformer_encoder"](
            inputs_embeds=obs_features,
            attention_mask=attention_mask
        )
        transformer_features = transformer_outputs.last_hidden_state  # [B, T, D]
        assert transformer_features.ndim == 3  # [B, T, D]

        obs_cond = transformer_features[torch.arange(B, device=transformer_features.device), env_t, :]  # [B, D]
        assert obs_cond.ndim == 2  # [B, D]

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        action = naction[:,:num_actions] # [B, num_actions, Da]
        return action

    def _get_action_trajectory(self):
        assert not self.nets.training
        # To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # convert obs queue to batch
        obs_list = list(self.obs_queue)
        obs_dict = dict()
        for k in obs_list[0]:
            obs_dict[k] = torch.cat([ obs_list[i][k].unsqueeze(1) for i in range(len(obs_list)) ], dim=1)  # [1,T,Do]

        # encode obs
        inputs = {
            "obs": obs_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # pass through projection layer
        obs_features = nets["policy"]["projection_layer"](obs_features)

        # pass through transformer
        attention_mask = torch.ones((B, obs_features.shape[1]), device=obs_features.device)
        transformer_outputs = nets["policy"]["transformer_encoder"](
            inputs_embeds=obs_features,
            attention_mask=attention_mask
        )
        transformer_features = transformer_outputs.last_hidden_state  # [B, T, D]
        assert transformer_features.ndim == 3  # [B, T, D]

        obs_cond = transformer_features[:, -1, :].reshape(B, -1)  # [B, D], only use last feature for conditioning

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        action = naction[:,:Ta]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
