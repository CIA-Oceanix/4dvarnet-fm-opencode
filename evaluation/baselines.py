import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from models.dynamics import DynamicsBase


def _gaspari_cohn(z):
    z = float(z)
    if z >= 2.0:
        return 0.0
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z
    z5 = z4 * z
    if z >= 1.0:
        return (1.0/12.0)*z5 - 0.5*z4 + (5.0/8.0)*z3 + (5.0/3.0)*z2 - 5.0*z + 4.0 - 2.0/(3.0*z)
    return -0.25*z5 + 0.5*z4 + (5.0/8.0)*z3 - (5.0/3.0)*z2 + 1.0


def _build_loc_matrices(state_dim, obs_operator, NO, J, loc_radius, device):
    if obs_operator.indices is not None:
        obs_indices = obs_operator.indices.cpu().numpy()
    else:
        obs_indices = np.arange(state_dim)
    obs_dim = len(obs_indices)

    def pos(i):
        return float(i) if i < NO else float((i - NO) // J)

    state_pos = torch.tensor([pos(i) for i in range(state_dim)], device=device)
    obs_pos = torch.tensor([pos(i) for i in obs_indices], device=device)

    L_x = torch.zeros((state_dim, obs_dim), device=device)
    L_y = torch.zeros((obs_dim, obs_dim), device=device)

    for si in range(state_dim):
        for oj in range(obs_dim):
            d = abs(float(state_pos[si] - obs_pos[oj]))
            d = min(d, NO - d)
            L_x[si, oj] = _gaspari_cohn(d / loc_radius)

    for oi in range(obs_dim):
        for oj in range(obs_dim):
            d = abs(float(obs_pos[oi] - obs_pos[oj]))
            d = min(d, NO - d)
            L_y[oi, oj] = _gaspari_cohn(d / loc_radius)

    return L_x, L_y


class ObsOperator:
    def __init__(self, state_dim: int, obs_indices=None):
        if obs_indices is not None:
            self.indices = torch.tensor(obs_indices, dtype=torch.long)
            self._obs_dim = len(obs_indices)
        else:
            self.indices = None
            self._obs_dim = state_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.indices is None:
            return x
        return x[..., self.indices]

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to(self, device):
        if self.indices is not None:
            self.indices = self.indices.to(device)
        return self

    def expand_to_state(self, obs_vec: torch.Tensor, state_dim: int) -> torch.Tensor:
        if self.indices is None:
            return obs_vec
        full = torch.zeros(obs_vec.shape[:-1] + (state_dim,), device=obs_vec.device, dtype=obs_vec.dtype)
        full[..., self.indices] = obs_vec
        return full


def _expand_obs_to_state(interp_obs, obs_operator, state_dim):
    sd = state_dim
    if obs_operator.indices is None:
        return interp_obs
    *batch_dims, T = interp_obs.shape[:-1]
    last_dim = interp_obs.shape[-1]
    full = torch.zeros(interp_obs.shape[:-1] + (sd,), device=interp_obs.device, dtype=interp_obs.dtype)
    full[..., obs_operator.indices] = interp_obs
    return full


def _init_bg_from_obs(interp_obs, obs_operator, state_dim, noise_std, device):
    state = _expand_obs_to_state(interp_obs, obs_operator, state_dim)
    if obs_operator.indices is not None:
        noise = torch.randn_like(state) * noise_std
        noise[..., obs_operator.indices] = 0.0
        state = state + noise
    else:
        state = state + torch.randn_like(state) * noise_std
    return state


def _safe_ref(ref, analysis, obs_operator):
    if analysis.shape[-1] != ref.shape[-1]:
        if obs_operator is not None and obs_operator.indices is not None:
            ref = ref[..., obs_operator.indices.cpu().numpy()]
        else:
            ref = ref[..., :analysis.shape[-1]]
    return ref


def _interp_observations(observations, obs_mask):
    B, T, D = observations.shape
    obs_np = observations.cpu().numpy()
    mask_np = obs_mask.cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    interp = np.zeros_like(obs_np)
    t = np.arange(T)
    for b in range(B):
        for d in range(D):
            idx = np.where(mask_np[b])[0]
            if len(idx) == 0:
                interp[b, :, d] = 0.0
            elif len(idx) == 1:
                interp[b, :, d] = obs_np[b, idx[0], d]
            else:
                interp[b, :, d] = np.interp(t, idx, obs_np[b, idx, d],
                                            left=obs_np[b, idx[0], d],
                                            right=obs_np[b, idx[-1], d])
    return torch.from_numpy(interp).to(device=observations.device, dtype=observations.dtype)


@dataclass
class BaselineResult:
    trajectory: np.ndarray
    rmse: np.ndarray
    ensemble: np.ndarray = None
    ensemble_variance: np.ndarray = None
    params: np.ndarray = None


class Weak4DVar:
    def __init__(
        self,
        da_window_steps: int = 300,
        B_var: float = 2.0,
        R_var: float = 0.5,
        Q_var: float = 0.05,
        lr: float = 0.02,
        opt_steps: int = 150,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
        obs_operator: ObsOperator = None,
    ):
        self.da_window_steps = da_window_steps
        self.B_var = B_var
        self.R_var = R_var
        self.Q_var = Q_var
        self.lr = lr
        self.opt_steps = opt_steps
        self.dt = dt
        self.device = device
        self.coupling_exponent = coupling_exponent
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim if dynamics else 3
        self.obs_operator = obs_operator or ObsOperator(self.state_dim)

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        sd = self.state_dim
        num_steps = observations.shape[0]
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, sd))

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        current_bg = _init_bg_from_obs(interp_obs[0], self.obs_operator, sd, 1.5, self.device)

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[start:end]
            win_mask = obs_mask[start:end]
            win_force = forcing[start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(self.da_window_steps, sd, device=self.device, requires_grad=True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl], lr=self.lr)

            H = self.obs_operator
            for _ in range(self.opt_steps):
                opt.zero_grad()
                traj = self._forward_weak(x0_ctrl, q_ctrl, self.da_window_steps, start, win_force, **params)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        diff = H(traj[t]) - win_obs[t]
                        J_o += torch.sum(diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q
                J_total.backward()
                opt.step()

            final_traj = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            next_forecast = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            current_bg = next_forecast[-1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse)

    def _forward_weak(self, x0, q, steps, start_idx, forcing, clip_range=50.0, **kwargs):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            W = forcing[t - 1]
            next_s = self.dynamics.step(s, W, **kwargs) + q[t]
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj)

    def _forward_weak_batch(self, x0, q, steps, start_idx, forcing, clip_range=50.0, **kwargs):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            W = forcing[:, t - 1]
            next_s = self.dynamics.step(s, W, **kwargs) + q[:, t]
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj, dim=1)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        sd = self.state_dim
        B, num_steps, _ = observations.shape
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, sd))

        interp_obs = _interp_observations(observations, obs_mask)
        current_bg = _init_bg_from_obs(interp_obs[:, 0], self.obs_operator, sd, 1.5, self.device)

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[:, start:end]
            win_mask = obs_mask[:, start:end]
            win_force = forcing[:, start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(B, self.da_window_steps, sd, device=self.device, requires_grad=True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl], lr=self.lr)

            H = self.obs_operator
            for _ in range(self.opt_steps):
                opt.zero_grad()
                traj = self._forward_weak_batch(x0_ctrl, q_ctrl, self.da_window_steps, start, win_force, **params)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                win_obs_clean = torch.nan_to_num(win_obs, nan=0.0)
                diff = H(traj) - win_obs_clean
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q
                J_total.backward()
                opt.step()

            final_traj = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            next_forecast = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            current_bg = next_forecast[:, -1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(trajectory=analysis[b], rmse=rmse_b))
        return results


class Strong4DVar:
    def __init__(
        self,
        da_window_steps: int = 300,
        B_var: float = 2.0,
        R_var: float = 0.5,
        max_iter: int = 40,
        lr: float = 0.1,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
        obs_operator: ObsOperator = None,
    ):
        self.da_window_steps = da_window_steps
        self.B_var = B_var
        self.R_var = R_var
        self.max_iter = max_iter
        self.lr = lr
        self.dt = dt
        self.device = device
        self.coupling_exponent = coupling_exponent
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim if dynamics else 3
        self.obs_operator = obs_operator or ObsOperator(self.state_dim)

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, sd))

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        current_bg = _init_bg_from_obs(interp_obs[0], self.obs_operator, sd, 1.5, self.device)
        H = self.obs_operator

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[start:end]
            win_mask = obs_mask[start:end]
            win_force = forcing[start:end]

            x_ctrl = current_bg.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.LBFGS([x_ctrl], max_iter=self.max_iter, lr=self.lr)

            def closure():
                opt.zero_grad()
                traj = self._forward_strong(x_ctrl, self.da_window_steps, start, win_force, **params)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        diff = H(traj[t]) - win_obs[t]
                        J_o += torch.sum(diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o
                J_total.backward()
                return J_total

            for _ in range(4):
                opt.step(closure)

            final_traj = self._forward_strong(
                x_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            current_bg = final_traj[-1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse)

    def _forward_strong(self, x0, steps, start_idx, forcing, clip_range=50.0, **kwargs):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            W = forcing[t - 1]
            next_s = self.dynamics.step(s, W, **kwargs)
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj)

    def _forward_strong_batch(self, x0, steps, start_idx, forcing, clip_range=50.0, **kwargs):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            W = forcing[:, t - 1]
            next_s = self.dynamics.step(s, W, **kwargs)
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj, dim=1)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, sd))

        interp_obs = _interp_observations(observations, obs_mask)
        current_bg = _init_bg_from_obs(interp_obs[:, 0], self.obs_operator, sd, 1.5, self.device)
        H = self.obs_operator

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[:, start:end]
            win_mask = obs_mask[:, start:end]
            win_force = forcing[:, start:end]

            x_ctrl = current_bg.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x_ctrl], lr=self.lr)

            for _ in range(self.max_iter * 4 if hasattr(self, 'max_iter') else 160):
                opt.zero_grad()
                traj = self._forward_strong_batch(x_ctrl, self.da_window_steps, start, win_force, **params)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                win_obs_clean = torch.nan_to_num(win_obs, nan=0.0)
                diff = H(traj) - win_obs_clean
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o
                J_total.backward()
                opt.step()

            final_traj = self._forward_strong_batch(
                x_ctrl.detach(), self.da_window_steps, start, win_force, **params
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            current_bg = final_traj[:, -1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(trajectory=analysis[b], rmse=rmse_b))
        return results


class ETKF:
    def __init__(
        self,
        N_ensemble: int = 30,
        R_var: float = 0.5,
        inflation: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
        obs_operator: ObsOperator = None,
    ):
        self.N_ensemble = N_ensemble
        self.R_var = R_var
        self.inflation = inflation
        self.dt = dt
        self.device = device
        self.coupling_exponent = coupling_exponent
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim if dynamics else 3
        self.obs_operator = obs_operator or ObsOperator(self.state_dim)

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        sd = self.state_dim
        N = self.N_ensemble
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)
        H = self.obs_operator

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        ensemble = _init_bg_from_obs(interp_obs[0], self.obs_operator, sd, 1.5, self.device).unsqueeze(0).repeat(N, 1)
        noise = torch.randn_like(ensemble) * 1.5
        if self.obs_operator.indices is not None:
            noise_obs = torch.randn((N, H.obs_dim), device=self.device) * np.sqrt(self.R_var)
            noise[..., self.obs_operator.indices] = noise_obs
        ensemble += noise

        analysis = np.zeros((num_steps, sd))
        ens_var = np.zeros((num_steps, sd))
        analysis[0] = torch.mean(ensemble, dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble, dim=0).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[t - 1]
            ensemble = self.dynamics.step(ensemble, W, **params)


            if obs_mask[t]:
                y_t = observations[t]
                mu = torch.mean(ensemble, dim=0)
                A = ensemble - mu
                mu_obs = H(mu)
                HA = H(ensemble) - mu_obs.unsqueeze(0)
                dy = y_t - mu_obs

                HA_w = HA * R_sym_sqrt_inv

                U, s, Vt = torch.linalg.svd(HA_w, full_matrices=False)
                s2 = s ** 2
                d = s2 + N1

                Pw = U @ torch.diag(1.0 / d) @ U.T
                Tmat = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T

                R_inv = 1.0 / self.R_var
                w = (dy * R_inv) @ HA.T @ Pw

                ensemble = mu + w @ A + Tmat @ A

                mu = torch.mean(ensemble, dim=0)
                ensemble = mu + self.inflation * (ensemble - mu)

            analysis[t] = torch.mean(ensemble, dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble, dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, ensemble=np.zeros((N, num_steps, self.state_dim)), ensemble_variance=ens_var)
    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        N = self.N_ensemble
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)
        H = self.obs_operator

        interp_obs = _interp_observations(observations, obs_mask)
        ensemble = _init_bg_from_obs(interp_obs[:, 0], self.obs_operator, self.state_dim, 1.5, self.device).unsqueeze(1).repeat(1, N, 1)
        noise = torch.randn_like(ensemble) * 1.5
        if self.obs_operator.indices is not None:
            noise_obs = torch.randn((B, N, H.obs_dim), device=self.device) * np.sqrt(self.R_var)
            noise[..., self.obs_operator.indices] = noise_obs
        ensemble += noise

        analysis = np.zeros((B, num_steps, self.state_dim))
        ens_var = np.zeros((B, num_steps, self.state_dim))
        analysis[:, 0] = torch.mean(ensemble, dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble, dim=1).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[:, t - 1]
            B0, N, D = ensemble.shape
            step_params = {k: (v.unsqueeze(1).expand(B0, N).reshape(B0 * N) if isinstance(v, torch.Tensor) and v.dim() == 1 else v) for k, v in params.items()}
            ensemble = self.dynamics.step(
                ensemble.reshape(B0 * N, D),
                W.unsqueeze(1).expand(B0, N).reshape(B0 * N),
                **step_params,
            ).reshape(B0, N, D)

            if obs_mask[:, t].any():
                for b in range(B):
                    if not obs_mask[b, t]:
                        continue
                    ens_b = ensemble[b]
                    y_t = observations[b, t]
                    mu = torch.mean(ens_b, dim=0)
                    A = ens_b - mu
                    mu_obs = H(mu)
                    HA = H(ens_b) - mu_obs.unsqueeze(0)
                    dy = y_t - mu_obs

                    HA_w = HA * R_sym_sqrt_inv

                    U, s, Vt = torch.linalg.svd(HA_w, full_matrices=False)
                    s2 = s ** 2
                    d = s2 + N1

                    Pw = U @ torch.diag(1.0 / d) @ U.T
                    Tmat = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T

                    R_inv = 1.0 / self.R_var
                    w = (dy * R_inv) @ HA.T @ Pw

                    ens_b = mu + w @ A + Tmat @ A
                    mu = torch.mean(ens_b, dim=0)
                    ensemble[b] = mu + self.inflation * (ens_b - mu)

            analysis[:, t] = torch.mean(ensemble, dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble, dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b,
                ensemble=np.zeros((N, num_steps, self.state_dim)),
                ensemble_variance=ens_var[b],
            ))
        return results


class EnKF:
    def __init__(
        self,
        N_ensemble: int = 30,
        R_var: float = 0.5,
        inflation: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
        obs_operator: ObsOperator = None,
        loc_radius: float = None,
        NO: int = 8,
        J: int = 4,
    ):
        self.N_ensemble = N_ensemble
        self.R_var = R_var
        self.inflation = inflation
        self.dt = dt
        self.device = device
        self.coupling_exponent = coupling_exponent
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim if dynamics else 3
        self.obs_operator = obs_operator or ObsOperator(self.state_dim)
        self.loc_radius = loc_radius
        if loc_radius is not None:
            self.loc_Lx, self.loc_Ly = _build_loc_matrices(
                self.state_dim, self.obs_operator, NO, J, loc_radius, device)

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        H = self.obs_operator
        od = H.obs_dim
        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        ensemble = _init_bg_from_obs(interp_obs[0], self.obs_operator, self.state_dim, 1.5, self.device).unsqueeze(0).repeat(self.N_ensemble, 1)
        noise = torch.randn_like(ensemble) * 1.5
        if self.obs_operator.indices is not None:
            noise_obs = torch.randn((self.N_ensemble, H.obs_dim), device=self.device) * np.sqrt(self.R_var)
            noise[..., self.obs_operator.indices] = noise_obs
        ensemble += noise

        analysis = np.zeros((num_steps, self.state_dim))
        ens_var = np.zeros((num_steps, self.state_dim))
        analysis[0] = torch.mean(ensemble, dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble, dim=0).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[t - 1]
            ensemble = self.dynamics.step(ensemble, W, **params)

            if obs_mask[t]:
                y_t = observations[t]
                mean_e = torch.mean(ensemble, dim=0)
                A = ensemble - mean_e
                H_ens = H(ensemble)
                H_mean_e = torch.mean(H_ens, dim=0)
                HA = H_ens - H_mean_e.unsqueeze(0)
                P_obs = (HA.T @ HA) / (self.N_ensemble - 1)
                cross_cov = (A.T @ HA) / (self.N_ensemble - 1)
                if self.loc_radius is not None:
                    P_obs = self.loc_Ly * P_obs
                    cross_cov = self.loc_Lx * cross_cov
                R_obs = torch.eye(od, device=self.device) * self.R_var
                K = cross_cov @ torch.inverse(P_obs + R_obs)
                for n in range(self.N_ensemble):
                    perturbed = y_t + torch.randn(od, device=self.device) * np.sqrt(self.R_var)
                    ensemble[n] += K @ (perturbed - H(ensemble[n]))

                mean_e = torch.mean(ensemble, dim=0)
                ensemble = mean_e + self.inflation * (ensemble - mean_e)

            analysis[t] = torch.mean(ensemble, dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble, dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, ensemble=np.zeros((self.N_ensemble, num_steps, self.state_dim)), ensemble_variance=ens_var)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        H = self.obs_operator
        od = H.obs_dim
        interp_obs = _interp_observations(observations, obs_mask)
        ensemble = _init_bg_from_obs(interp_obs[:, 0], self.obs_operator, self.state_dim, 1.5, self.device).unsqueeze(1).repeat(1, self.N_ensemble, 1)
        noise = torch.randn_like(ensemble) * 1.5
        if self.obs_operator.indices is not None:
            noise_obs = torch.randn((B, self.N_ensemble, H.obs_dim), device=self.device) * np.sqrt(self.R_var)
            noise[..., self.obs_operator.indices] = noise_obs
        ensemble += noise

        analysis = np.zeros((B, num_steps, self.state_dim))
        ens_var = np.zeros((B, num_steps, self.state_dim))
        analysis[:, 0] = torch.mean(ensemble, dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble, dim=1).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[:, t - 1]
            B0, N, D = ensemble.shape
            step_params = {k: (v.unsqueeze(1).expand(B0, N).reshape(B0 * N) if isinstance(v, torch.Tensor) and v.dim() == 1 else v) for k, v in params.items()}
            ensemble = self.dynamics.step(
                ensemble.reshape(B0 * N, D),
                W.unsqueeze(1).expand(B0, N).reshape(B0 * N),
                **step_params,
            ).reshape(B0, N, D)

            if obs_mask[:, t].any():
                y_t = observations[:, t]
                mean_e = torch.mean(ensemble, dim=1)
                A = ensemble - mean_e.unsqueeze(1)
                H_ens = H(ensemble)
                H_mean_e = torch.mean(H_ens, dim=1)
                HA = H_ens - H_mean_e.unsqueeze(1)
                P_obs = (HA.transpose(1, 2) @ HA) / (self.N_ensemble - 1)
                cross_cov = (A.transpose(1, 2) @ HA) / (self.N_ensemble - 1)
                if self.loc_radius is not None:
                    P_obs = self.loc_Ly.unsqueeze(0) * P_obs
                    cross_cov = self.loc_Lx.unsqueeze(0) * cross_cov
                R_obs = torch.eye(od, device=self.device).unsqueeze(0) * self.R_var
                K = cross_cov @ torch.inverse(P_obs + R_obs)
                for n in range(self.N_ensemble):
                    perturbed = y_t + torch.randn((B, od), device=self.device) * np.sqrt(self.R_var)
                    ensemble[:, n] += (K @ (perturbed - H(ensemble[:, n])).unsqueeze(-1)).squeeze(-1)

                mean_e = torch.mean(ensemble, dim=1)
                ensemble = mean_e.unsqueeze(1) + self.inflation * (ensemble - mean_e.unsqueeze(1))

            analysis[:, t] = torch.mean(ensemble, dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble, dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b,
                ensemble=np.zeros((self.N_ensemble, num_steps, self.state_dim)),
                ensemble_variance=ens_var[b],
            ))
        return results


class JointWeak4DVar(Weak4DVar):
    def __init__(
        self,
        da_window_steps: int = 300,
        B_var: float = 2.0,
        R_var: float = 0.5,
        Q_var: float = 0.05,
        P_var: float = 1.0,
        lr: float = 0.02,
        opt_steps: int = 150,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
    ):
        super().__init__(
            da_window_steps=da_window_steps,
            B_var=B_var, R_var=R_var, Q_var=Q_var,
            lr=lr, opt_steps=opt_steps, dt=dt,
            device=device, coupling_exponent=coupling_exponent,
            dynamics=dynamics,
        )
        self.P_var = P_var

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, sd))
        param_arr = np.zeros((num_steps, 4))

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        current_bg = interp_obs[0].clone() + torch.randn(sd, device=self.device) * 1.5
        log_s = torch.tensor(np.log(max(sigma, 1e-6)), device=self.device)
        log_r = torch.tensor(np.log(max(rho, 1e-6)), device=self.device)
        log_b = torch.tensor(np.log(max(beta, 1e-6)), device=self.device)
        log_c = torch.tensor(np.log(max(c1, 1e-6)), device=self.device)
        s_prior, r_prior, b_prior, c_prior = log_s, log_r, log_b, log_c

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[start:end]
            win_mask = obs_mask[start:end]
            win_force = forcing[start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(self.da_window_steps, sd, device=self.device, requires_grad=True)
            ls = log_s.clone().detach().requires_grad_(True)
            lr_ = log_r.clone().detach().requires_grad_(True)
            lb = log_b.clone().detach().requires_grad_(True)
            lc = log_c.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl, ls, lr_, lb, lc], lr=self.lr)

            for _ in range(self.opt_steps):
                opt.zero_grad()
                s_val, r_val, b_val = torch.exp(ls), torch.exp(lr_), torch.exp(lb)
                c_val = torch.exp(lc)
                traj = self._forward_weak(x0_ctrl, q_ctrl, self.da_window_steps,
                                          start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                J_p = ((ls - s_prior) ** 2 + (lr_ - r_prior) ** 2 +
                       (lb - b_prior) ** 2 + (lc - c_prior) ** 2) / self.P_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        J_o += torch.sum((traj[t] - win_obs[t]) ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q + 0.1 * J_p
                J_total.backward()
                opt.step()

            s_val, r_val, b_val = torch.exp(ls.detach()), torch.exp(lr_.detach()), torch.exp(lb.detach())
            c_val = torch.exp(lc.detach())
            final_traj = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            param_arr[start:end] = np.tile(
                [float(s_val), float(r_val), float(b_val), float(c_val)], (self.da_window_steps, 1))
            next_forecast = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            current_bg = next_forecast[-1].detach()
            log_s, log_r, log_b, log_c = ls.detach(), lr_.detach(), lb.detach(), lc.detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, params=param_arr)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, sd))
        param_arr = np.zeros((B, num_steps, 4))

        if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
            sigma_b, rho_b, beta_b, c1_b = sigma, rho, beta, c1
        else:
            sigma_b = torch.full((B,), sigma, device=self.device)
            rho_b = torch.full((B,), rho, device=self.device)
            beta_b = torch.full((B,), beta, device=self.device)
            c1_b = torch.full((B,), c1, device=self.device)

        interp_obs = _interp_observations(observations, obs_mask)
        current_bg = interp_obs[:, 0].clone() + torch.randn(B, sd, device=self.device) * 1.5
        log_s = torch.log(sigma_b.clamp(min=1e-6))
        log_r = torch.log(rho_b.clamp(min=1e-6))
        log_b = torch.log(beta_b.clamp(min=1e-6))
        log_c = torch.log(c1_b.clamp(min=1e-6))
        s_prior, r_prior, b_prior, c_prior = log_s.clone(), log_r.clone(), log_b.clone(), log_c.clone()

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[:, start:end]
            win_mask = obs_mask[:, start:end]
            win_force = forcing[:, start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(B, self.da_window_steps, sd, device=self.device, requires_grad=True)
            ls = log_s.clone().detach().requires_grad_(True)
            lr_ = log_r.clone().detach().requires_grad_(True)
            lb = log_b.clone().detach().requires_grad_(True)
            lc = log_c.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl, ls, lr_, lb, lc], lr=self.lr)

            for _ in range(self.opt_steps):
                opt.zero_grad()
                s_val, r_val, b_val = torch.exp(ls), torch.exp(lr_), torch.exp(lb)
                c_val = torch.exp(lc)
                traj = self._forward_weak_batch(x0_ctrl, q_ctrl, self.da_window_steps,
                                                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                J_p = torch.sum((ls - s_prior) ** 2 + (lr_ - r_prior) ** 2 +
                                (lb - b_prior) ** 2 + (lc - c_prior) ** 2) / self.P_var
                diff = traj - win_obs
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q + 0.1 * J_p
                J_total.backward()
                opt.step()

            s_val, r_val, b_val = torch.exp(ls.detach()), torch.exp(lr_.detach()), torch.exp(lb.detach())
            c_val = torch.exp(lc.detach())
            final_traj = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            param_arr[:, start:end] = torch.stack([s_val, r_val, b_val, c_val], dim=1).unsqueeze(1).expand(
                B, self.da_window_steps, 4).detach().cpu().numpy()
            next_forecast = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            current_bg = next_forecast[:, -1].detach()
            log_s, log_r, log_b, log_c = ls.detach(), lr_.detach(), lb.detach(), lc.detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(trajectory=analysis[b], rmse=rmse_b, params=param_arr[b]))
        return results


class JointStrong4DVar(Strong4DVar):
    def __init__(
        self,
        da_window_steps: int = 300,
        B_var: float = 2.0,
        R_var: float = 0.5,
        P_var: float = 1.0,
        max_iter: int = 40,
        lr: float = 0.1,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
    ):
        super().__init__(
            da_window_steps=da_window_steps,
            B_var=B_var, R_var=R_var,
            max_iter=max_iter, lr=lr, dt=dt,
            device=device, coupling_exponent=coupling_exponent,
            dynamics=dynamics,
        )
        self.P_var = P_var

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, sd))
        param_arr = np.zeros((num_steps, 4))

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        current_bg = interp_obs[0].clone() + torch.randn(sd, device=self.device) * 1.5
        log_s = torch.tensor(np.log(max(sigma, 1e-6)), device=self.device)
        log_r = torch.tensor(np.log(max(rho, 1e-6)), device=self.device)
        log_b = torch.tensor(np.log(max(beta, 1e-6)), device=self.device)
        log_c = torch.tensor(np.log(max(c1, 1e-6)), device=self.device)
        s_prior, r_prior, b_prior, c_prior = log_s, log_r, log_b, log_c

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[start:end]
            win_mask = obs_mask[start:end]
            win_force = forcing[start:end]

            x_ctrl = current_bg.clone().detach().requires_grad_(True)
            ls = log_s.clone().detach().requires_grad_(True)
            lr_ = log_r.clone().detach().requires_grad_(True)
            lb = log_b.clone().detach().requires_grad_(True)
            lc = log_c.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x_ctrl, ls, lr_, lb, lc], lr=self.lr)

            for _ in range(self.max_iter * 4):
                opt.zero_grad()
                s_val, r_val, b_val = torch.exp(ls), torch.exp(lr_), torch.exp(lb)
                c_val = torch.exp(lc)
                traj = self._forward_strong(x_ctrl, self.da_window_steps,
                                            start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                J_p = ((ls - s_prior) ** 2 + (lr_ - r_prior) ** 2 +
                       (lb - b_prior) ** 2 + (lc - c_prior) ** 2) / self.P_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        J_o += torch.sum((traj[t] - win_obs[t]) ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.1 * J_p
                J_total.backward()
                opt.step()

            s_val, r_val, b_val = torch.exp(ls.detach()), torch.exp(lr_.detach()), torch.exp(lb.detach())
            c_val = torch.exp(lc.detach())
            final_traj = self._forward_strong(
                x_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            param_arr[start:end] = np.tile(
                [float(s_val), float(r_val), float(b_val), float(c_val)], (self.da_window_steps, 1))
            current_bg = final_traj[-1].detach()
            log_s, log_r, log_b, log_c = ls.detach(), lr_.detach(), lb.detach(), lc.detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, params=param_arr)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        sd = self.state_dim
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, sd))
        param_arr = np.zeros((B, num_steps, 4))

        if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
            sigma_b, rho_b, beta_b, c1_b = sigma, rho, beta, c1
        else:
            sigma_b = torch.full((B,), sigma, device=self.device)
            rho_b = torch.full((B,), rho, device=self.device)
            beta_b = torch.full((B,), beta, device=self.device)
            c1_b = torch.full((B,), c1, device=self.device)

        interp_obs = _interp_observations(observations, obs_mask)
        current_bg = interp_obs[:, 0].clone() + torch.randn(B, sd, device=self.device) * 1.5
        log_s = torch.log(sigma_b.clamp(min=1e-6))
        log_r = torch.log(rho_b.clamp(min=1e-6))
        log_b = torch.log(beta_b.clamp(min=1e-6))
        log_c = torch.log(c1_b.clamp(min=1e-6))
        s_prior, r_prior, b_prior, c_prior = log_s.clone(), log_r.clone(), log_b.clone(), log_c.clone()

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[:, start:end]
            win_mask = obs_mask[:, start:end]
            win_force = forcing[:, start:end]

            x_ctrl = current_bg.clone().detach().requires_grad_(True)
            ls = log_s.clone().detach().requires_grad_(True)
            lr_ = log_r.clone().detach().requires_grad_(True)
            lb = log_b.clone().detach().requires_grad_(True)
            lc = log_c.clone().detach().requires_grad_(True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x_ctrl, ls, lr_, lb, lc], lr=self.lr)

            for _ in range(self.max_iter * 4):
                opt.zero_grad()
                s_val, r_val, b_val = torch.exp(ls), torch.exp(lr_), torch.exp(lb)
                c_val = torch.exp(lc)
                traj = self._forward_strong_batch(x_ctrl, self.da_window_steps,
                                                  start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                J_p = torch.sum((ls - s_prior) ** 2 + (lr_ - r_prior) ** 2 +
                                (lb - b_prior) ** 2 + (lc - c_prior) ** 2) / self.P_var
                diff = traj - win_obs
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.1 * J_p
                J_total.backward()
                opt.step()

            s_val, r_val, b_val = torch.exp(ls.detach()), torch.exp(lr_.detach()), torch.exp(lb.detach())
            c_val = torch.exp(lc.detach())
            final_traj = self._forward_strong_batch(
                x_ctrl.detach(), self.da_window_steps,
                start, win_force, sigma=s_val, rho=r_val, beta=b_val, c1=c_val
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            param_arr[:, start:end] = torch.stack([s_val, r_val, b_val, c_val], dim=1).unsqueeze(1).expand(
                B, self.da_window_steps, 4).detach().cpu().numpy()
            current_bg = final_traj[:, -1].detach()
            log_s, log_r, log_b, log_c = ls.detach(), lr_.detach(), lb.detach(), lc.detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(trajectory=analysis[b], rmse=rmse_b, params=param_arr[b]))
        return results


class JointEnKF(EnKF):
    def __init__(
        self,
        N_ensemble: int = 30,
        R_var: float = 0.5,
        inflation: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
    ):
        super().__init__(
            N_ensemble=N_ensemble, R_var=R_var, inflation=inflation,
            dt=dt, device=device, coupling_exponent=coupling_exponent,
            dynamics=dynamics,
        )

    def _init_ensemble(self, obs0, sigma, rho, beta, c1):
        N = self.N_ensemble
        state = obs0.clone().unsqueeze(0).repeat(N, 1)
        state += torch.randn((N, self.state_dim), device=self.device) * 1.5
        sigmas = torch.full((N, 1), sigma, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        rhos = torch.full((N, 1), rho, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        betas = torch.full((N, 1), beta, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        c1s = torch.full((N, 1), c1, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        return torch.cat([state, sigmas, rhos, betas, c1s], dim=1)

    def _init_ensemble_batch(self, obs0, sigma, rho, beta, c1):
        B = obs0.shape[0]
        N = self.N_ensemble
        state = obs0.clone().unsqueeze(1).repeat(1, N, 1)
        state += torch.randn((B, N, self.state_dim), device=self.device) * 1.5
        if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
            sigmas = sigma.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            rhos = rho.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            betas = beta.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            c1s = c1.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
        else:
            sigmas = torch.full((B, N, 1), sigma, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            rhos = torch.full((B, N, 1), rho, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            betas = torch.full((B, N, 1), beta, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            c1s = torch.full((B, N, 1), c1, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
        return torch.cat([state, sigmas, rhos, betas, c1s], dim=-1)

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        N = self.N_ensemble
        N_dim = self.state_dim + 4
        N1 = N - 1

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        ensemble = self._init_ensemble(interp_obs[0], sigma, rho, beta, c1)

        sd = self.state_dim
        analysis = np.zeros((num_steps, sd))
        ens_var = np.zeros((num_steps, sd))
        param_arr = np.zeros((num_steps, 4))
        analysis[0] = torch.mean(ensemble[:, :sd], dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble[:, :sd], dim=0).cpu().numpy()
        param_arr[0] = torch.mean(ensemble[:, sd:], dim=0).detach().cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[t - 1]
            sig_e = ensemble[:, 3].clamp(min=1e-6)
            rho_e = ensemble[:, 4].clamp(min=1e-6)
            beta_e = ensemble[:, 5].clamp(min=1e-6)
            ensemble[:, :sd] = self.dynamics.step(ensemble[:, :sd], W.expand(N), sigma=sig_e, rho=rho_e, beta=beta_e)

            if obs_mask[t]:
                y_t = observations[t]
                mean_e = torch.mean(ensemble, dim=0)
                A = ensemble - mean_e
                P_b = (A.T @ A) / N1
                H = torch.zeros(sd, N_dim, device=self.device)
                for i in range(sd):
                    H[i, i] = 1.0
                K = P_b @ H.T @ torch.inverse(H @ P_b @ H.T + torch.eye(sd, device=self.device) * self.R_var)
                for n in range(N):
                    perturbed = y_t + torch.randn(sd, device=self.device) * np.sqrt(self.R_var)
                    ensemble[n] += K @ (perturbed - H @ ensemble[n])

                mean_e = torch.mean(ensemble, dim=0)
                ensemble = mean_e + self.inflation * (ensemble - mean_e)
                ensemble[:, sd:] = ensemble[:, sd:].clamp(min=1e-6)
                ensemble[:, 3] = ensemble[:, 3].clamp(max=30.0)
                ensemble[:, 4] = ensemble[:, 4].clamp(max=50.0)
                ensemble[:, 5] = ensemble[:, 5].clamp(max=10.0)
                ensemble[:, 6] = ensemble[:, 6].clamp(max=5.0)

            analysis[t] = torch.mean(ensemble[:, :sd], dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble[:, :sd], dim=0).detach().cpu().numpy()
            param_arr[t] = torch.mean(ensemble[:, sd:], dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, params=param_arr,
                              ensemble=np.zeros((N, num_steps, sd)),
                              ensemble_variance=ens_var)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        N = self.N_ensemble
        sd = self.state_dim
        N_dim = sd + 4
        N1 = N - 1

        interp_obs = _interp_observations(observations, obs_mask)
        ensemble = self._init_ensemble_batch(interp_obs[:, 0], sigma, rho, beta, c1)

        analysis = np.zeros((B, num_steps, sd))
        ens_var = np.zeros((B, num_steps, sd))
        param_arr = np.zeros((B, num_steps, 4))
        analysis[:, 0] = torch.mean(ensemble[:, :, :sd], dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble[:, :, :sd], dim=1).cpu().numpy()
        param_arr[:, 0] = torch.mean(ensemble[:, :, sd:], dim=1).detach().cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[:, t - 1, None]
            sig_e = ensemble[:, :, 3].clamp(min=1e-6)
            rho_e = ensemble[:, :, 4].clamp(min=1e-6)
            beta_e = ensemble[:, :, 5].clamp(min=1e-6)
            ensemble[:, :, :sd] = self.dynamics.step(ensemble[:, :, :sd], W.expand(B, -1), sigma=sig_e, rho=rho_e, beta=beta_e)

            if obs_mask[:, t].any():
                for b in range(B):
                    if not obs_mask[b, t]:
                        continue
                    ens_b = ensemble[b]
                    y_t = observations[b, t]
                    mean_e = torch.mean(ens_b, dim=0)
                    A = ens_b - mean_e
                    P_b = (A.T @ A) / N1
                    H = torch.zeros(sd, N_dim, device=self.device)
                    for i in range(sd):
                        H[i, i] = 1.0
                    K = P_b @ H.T @ torch.inverse(H @ P_b @ H.T + torch.eye(sd, device=self.device) * self.R_var)
                    for n in range(N):
                        perturbed = y_t + torch.randn(sd, device=self.device) * np.sqrt(self.R_var)
                        ens_b[n] += K @ (perturbed - H @ ens_b[n])
                    mean_e = torch.mean(ens_b, dim=0)
                    ensemble[b] = mean_e + self.inflation * (ens_b - mean_e)
                    ensemble[b, :, sd:] = ensemble[b, :, sd:].clamp(min=1e-6)
                    ensemble[b, :, 3] = ensemble[b, :, 3].clamp(max=30.0)
                    ensemble[b, :, 4] = ensemble[b, :, 4].clamp(max=50.0)
                    ensemble[b, :, 5] = ensemble[b, :, 5].clamp(max=10.0)
                    ensemble[b, :, 6] = ensemble[b, :, 6].clamp(max=5.0)

            analysis[:, t] = torch.mean(ensemble[:, :, :sd], dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble[:, :, :sd], dim=1).detach().cpu().numpy()
            param_arr[:, t] = torch.mean(ensemble[:, :, sd:], dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b, params=param_arr[b],
                ensemble=np.zeros((N, num_steps, sd)),
                ensemble_variance=ens_var[b],
            ))
        return results


class JointETKF(ETKF):
    def __init__(
        self,
        N_ensemble: int = 30,
        R_var: float = 0.5,
        inflation: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device("cpu"),
        coupling_exponent: float = 1.0,
        dynamics: DynamicsBase = None,
    ):
        super().__init__(
            N_ensemble=N_ensemble, R_var=R_var, inflation=inflation,
            dt=dt, device=device, coupling_exponent=coupling_exponent,
            dynamics=dynamics,
        )

    def assimilate(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> BaselineResult:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        num_steps = observations.shape[0]
        N = self.N_ensemble
        sd = self.state_dim
        N_dim = sd + 4
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)

        interp_obs = _interp_observations(observations.unsqueeze(0), obs_mask.unsqueeze(0))[0]
        state = interp_obs[0].clone().unsqueeze(0).repeat(N, 1)
        state += torch.randn((N, sd), device=self.device) * 1.5
        sigmas = torch.full((N, 1), sigma, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        rhos = torch.full((N, 1), rho, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        betas = torch.full((N, 1), beta, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        c1s = torch.full((N, 1), c1, device=self.device) * (1 + torch.randn(N, 1, device=self.device) * 0.1)
        ensemble = torch.cat([state, sigmas, rhos, betas, c1s], dim=1)

        analysis = np.zeros((num_steps, sd))
        ens_var = np.zeros((num_steps, sd))
        param_arr = np.zeros((num_steps, 4))
        analysis[0] = torch.mean(ensemble[:, :sd], dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble[:, :sd], dim=0).cpu().numpy()
        param_arr[0] = torch.mean(ensemble[:, sd:], dim=0).detach().cpu().numpy()

        H_obs = torch.zeros(sd, N_dim, device=self.device)
        for i in range(sd):
            H_obs[i, i] = 1.0

        for t in range(1, num_steps):
            W = forcing[t - 1]
            sig_e = ensemble[:, 3].clamp(min=1e-6)
            rho_e = ensemble[:, 4].clamp(min=1e-6)
            beta_e = ensemble[:, 5].clamp(min=1e-6)
            ensemble[:, :sd] = self.dynamics.step(ensemble[:, :sd], W.unsqueeze(1).expand(N), sigma=sig_e, rho=rho_e, beta=beta_e)

            if obs_mask[t]:
                y_t = observations[t]
                mu = torch.mean(ensemble, dim=0)
                A = ensemble - mu
                HA = A @ H_obs.T
                Y = HA
                dy = y_t - mu[:sd]

                Y_w = Y * R_sym_sqrt_inv
                U, s, Vt = torch.linalg.svd(Y_w, full_matrices=False)
                s2 = s ** 2
                d = s2 + N1

                Pw = U @ torch.diag(1.0 / d) @ U.T
                T = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T

                R_inv = 1.0 / self.R_var
                w = (dy * R_inv) @ Y.T @ Pw

                ensemble = mu + w @ A + T @ A

                mu = torch.mean(ensemble, dim=0)
                ensemble = mu + self.inflation * (ensemble - mu)
                ensemble[:, sd:] = ensemble[:, sd:].clamp(min=1e-6)
                ensemble[:, 3] = ensemble[:, 3].clamp(max=30.0)
                ensemble[:, 4] = ensemble[:, 4].clamp(max=50.0)
                ensemble[:, 5] = ensemble[:, 5].clamp(max=10.0)
                ensemble[:, 6] = ensemble[:, 6].clamp(max=5.0)

            analysis[t] = torch.mean(ensemble[:, :sd], dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble[:, :sd], dim=0).detach().cpu().numpy()
            param_arr[t] = torch.mean(ensemble[:, sd:], dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, params=param_arr,
                              ensemble=np.zeros((N, num_steps, sd)),
                              ensemble_variance=ens_var)

    def assimilate_batch(
        self,
        observations: torch.Tensor,
        obs_mask: torch.Tensor,
        forcing: torch.Tensor,
        true_state: torch.Tensor = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        c1: float = 1.0,
            **kwargs,
    ) -> list:
        params = dict(sigma=sigma, rho=rho, beta=beta, c1=c1, **kwargs)

        B, num_steps, _ = observations.shape
        N = self.N_ensemble
        sd = self.state_dim
        N_dim = sd + 4
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)

        interp_obs = _interp_observations(observations, obs_mask)
        state = interp_obs[:, 0].clone().unsqueeze(1).repeat(1, N, 1)
        state += torch.randn((B, N, sd), device=self.device) * 1.5
        if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
            sigmas = sigma.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            rhos = rho.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            betas = beta.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            c1s = c1.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
        else:
            sigmas = torch.full((B, N, 1), sigma, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            rhos = torch.full((B, N, 1), rho, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            betas = torch.full((B, N, 1), beta, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
            c1s = torch.full((B, N, 1), c1, device=self.device) * (1 + torch.randn(B, N, 1, device=self.device) * 0.1)
        ensemble = torch.cat([state, sigmas, rhos, betas, c1s], dim=-1)

        analysis = np.zeros((B, num_steps, sd))
        ens_var = np.zeros((B, num_steps, sd))
        param_arr = np.zeros((B, num_steps, 4))
        analysis[:, 0] = torch.mean(ensemble[:, :, :sd], dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble[:, :, :sd], dim=1).cpu().numpy()
        param_arr[:, 0] = torch.mean(ensemble[:, :, sd:], dim=1).detach().cpu().numpy()

        H_obs = torch.zeros(1, sd, N_dim, device=self.device)
        for i in range(sd):
            H_obs[0, i, i] = 1.0

        for t in range(1, num_steps):
            W = forcing[:, t - 1, None]
            sig_e = ensemble[:, :, 3].clamp(min=1e-6)
            rho_e = ensemble[:, :, 4].clamp(min=1e-6)
            beta_e = ensemble[:, :, 5].clamp(min=1e-6)
            ensemble[:, :, :sd] = self.dynamics.step(ensemble[:, :, :sd], W.expand(B, -1), sigma=sig_e, rho=rho_e, beta=beta_e)

            if obs_mask[:, t].any():
                for b in range(B):
                    if not obs_mask[b, t]:
                        continue
                    ens_b = ensemble[b]
                    y_t = observations[b, t]
                    mu = torch.mean(ens_b, dim=0)
                    A = ens_b - mu
                    HA = A @ H_obs[0].T
                    dy = y_t - mu[:sd]

                    Y_w = HA * R_sym_sqrt_inv
                    U, s_, Vt = torch.linalg.svd(Y_w, full_matrices=False)
                    s2 = s_ ** 2
                    d = s2 + N1

                    Pw = U @ torch.diag(1.0 / d) @ U.T
                    T = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T

                    R_inv = 1.0 / self.R_var
                    w = (dy * R_inv) @ HA.T @ Pw

                    ens_b = mu + w @ A + T @ A
                    mu = torch.mean(ens_b, dim=0)
                    ensemble[b] = mu + self.inflation * (ens_b - mu)
                    ensemble[b, :, sd:] = ensemble[b, :, sd:].clamp(min=1e-6)
                    ensemble[b, :, 3] = ensemble[b, :, 3].clamp(max=30.0)
                    ensemble[b, :, 4] = ensemble[b, :, 4].clamp(max=50.0)
                    ensemble[b, :, 5] = ensemble[b, :, 5].clamp(max=10.0)
                    ensemble[b, :, 6] = ensemble[b, :, 6].clamp(max=5.0)

            analysis[:, t] = torch.mean(ensemble[:, :, :sd], dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble[:, :, :sd], dim=1).detach().cpu().numpy()
            param_arr[:, t] = torch.mean(ensemble[:, :, sd:], dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        ref = _safe_ref(ref, analysis, getattr(self, 'obs_operator', None))
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b, params=param_arr[b],
                ensemble=np.zeros((N, num_steps, sd)),
                ensemble_variance=ens_var[b],
            ))
        return results
