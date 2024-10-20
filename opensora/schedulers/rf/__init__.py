import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(
            prompts
        )  # a police car is driving down the street at night aesthetic score: 6.5.
        y_null = text_encoder.null(n)
        model_y = [model_args["y"], y_null]
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        # TODO maybe you miss this?
        # if self.use_timestep_transform:
        # timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        progress_wrap = tqdm if progress else (lambda x: x)
        # pred_cache = {}
        total_t = 1.0
        v_est_t_minus_1 = None
        v_est_t = None
        z_est_t_minus_1 = None
        z_est_t = None
        est_t_minus_1 = None
        est_t = None
        skip_steps = 0

        dt = timesteps[0] - timesteps[1]
        dt = dt / self.num_timesteps
        prediction_steps = 0
        for i, t in progress_wrap(enumerate(timesteps)):
            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            if "y" in model_args:
                del model_args["y"]

            if skip_steps > 0:
                skip_steps -= 1
            else:
                prediction_steps += 1
                print("Prediction Step", t, prediction_steps)
                pred_cond = model(z_in[0:1], t[0:1], y=model_y[0], **model_args).chunk(2, dim=1)[0]
                pred_uncond = model(z_in[1:2], t[1:2], y=model_y[1], **model_args).chunk(2, dim=1)[0]
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                if z_est_t is None:
                    z_est_t_minus_1 = None
                    z_est_t = z.clone()
                else:
                    # z at the previous step
                    z_est_t_minus_1 = z_est_t
                    # z at the current step
                    z_est_t = z.clone()

                # Try to find out the estimate of x0
                total_t -= dt.item()
                # For the first step
                if v_est_t is None:
                    v_est_t_minus_1 = None
                    v_est_t = v_pred.clone()
                    est_t_minus_1 = None
                    est_t = t
                else:
                    est_t_minus_1 = est_t
                    est_t = t

                    v_est_t_minus_1 = v_est_t
                    v_est_t = v_pred.clone()
                    forward_n_extra_steps = self._get_n_forward_steps(
                        z_est_t_minus_1,
                        z_est_t,
                        v_est_t_minus_1,
                        v_est_t,
                        est_t_minus_1,
                        est_t,
                        dt,
                        alpha=0.01,
                        max_lookup_steps=len(timesteps) - i,
                    )
                    skip_steps = forward_n_extra_steps

            z = z + v_pred * dt[:, None, None, None, None]

        return z

    def _get_n_forward_steps(
        self,
        z_est_t_minus_1,
        z_est_t,
        v_est_t_minus_1,
        v_est_t,
        est_t_minus_1,
        est_t,
        dt,
        alpha=0.01,
        max_lookup_steps=50,
    ):
        anchor_distance = self._get_sample_distance(
            z_est_t_minus_1, z_est_t, v_est_t_minus_1, v_est_t, est_t_minus_1, est_t, dt, look_ahead_n=1.0
        )
        # print("Anchor distance: ", anchor_distance)
        distance_array = []
        for i in range(1, max_lookup_steps):
            distance = self._get_sample_distance(
                z_est_t_minus_1, z_est_t, v_est_t_minus_1, v_est_t, est_t_minus_1, est_t, dt, look_ahead_n=1.0 + i
            )
            distance_array.append(distance)
            # print("Look ahead n: ", i + 1, "Distance: ", distance, anchor_distance, "max_lookup_steps", max_lookup_steps)
        for i in range(len(distance_array)):
            if abs(distance_array[i] - anchor_distance) / anchor_distance > alpha:
                print("Found next: ", i, distance_array[i], anchor_distance)
                return i
        return 0

    def _get_sample_distance(
        self, z_est_t_minus_1, z_est_t, v_est_t_minus_1, v_est_t, est_t_minus_1, est_t, dt, look_ahead_n=1.0
    ):
        anchor_t = z_est_t + v_est_t * self.expand(dt) * look_ahead_n
        dt_two_anchor = est_t_minus_1 - est_t
        anchor_t_minus_1 = z_est_t_minus_1 + self.expand(dt * (dt_two_anchor + look_ahead_n)) * v_est_t_minus_1
        return torch.mean(torch.abs(anchor_t_minus_1 - anchor_t))

    def expand(self, x):
        return x[:, None, None, None, None]

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)
