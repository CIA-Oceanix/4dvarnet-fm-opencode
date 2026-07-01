import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass


def _apply_coupling(W: torch.Tensor, c1: float, coupling_type: str) -> torch.Tensor:
    if coupling_type == "quartic":
        return c1 * torch.sign(W) * W ** 2
    return c1 * W


@dataclass
class BaselineResult:
    trajectory: np.ndarray
    rmse: np.ndarray
    ensemble: np.ndarray = None
    ensemble_variance: np.ndarray = None


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
        coupling_type: str = "linear",
    ):
        self.da_window_steps = da_window_steps
        self.B_var = B_var
        self.R_var = R_var
        self.Q_var = Q_var
        self.lr = lr
        self.opt_steps = opt_steps
        self.dt = dt
        self.device = device
        self.coupling_type = coupling_type

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
    ) -> BaselineResult:
        num_steps = observations.shape[0]
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, 3))

        current_bg = observations[0].clone() + torch.randn(3, device=self.device) * 1.5

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[start:end]
            win_mask = obs_mask[start:end]
            win_force = forcing[start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(self.da_window_steps, 3, device=self.device, requires_grad=True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl], lr=self.lr)

            for _ in range(self.opt_steps):
                opt.zero_grad()
                traj = self._forward_weak(x0_ctrl, q_ctrl, self.da_window_steps, start, win_force, sigma, rho, beta, c1)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        J_o += torch.sum((traj[t] - win_obs[t]) ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q
                J_total.backward()
                opt.step()

            final_traj = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            next_forecast = self._forward_weak(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            current_bg = next_forecast[-1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse)

    def _forward_weak(self, x0, q, steps, start_idx, forcing, sigma, rho, beta, c1, clip_range=50.0):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            X, Y, Z = s[0], s[1], s[2]
            W = forcing[t - 1]
            dX = sigma * (Y - X) + _apply_coupling(W, c1, self.coupling_type)
            dY = X * (rho - Z) - Y
            dZ = X * Y - beta * Z
            Xn = X + dX * self.dt + q[t, 0]
            Yn = Y + dY * self.dt + q[t, 1]
            Zn = Z + dZ * self.dt + q[t, 2]
            next_s = torch.stack([Xn, Yn, Zn])
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj)

    def _forward_weak_batch(self, x0, q, steps, start_idx, forcing, sigma, rho, beta, c1, clip_range=50.0):
        B = x0.shape[0]
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            X, Y, Z = s[:, 0], s[:, 1], s[:, 2]
            W = forcing[:, t - 1]
            dX = sigma * (Y - X) + _apply_coupling(W, c1, self.coupling_type)
            dY = X * (rho - Z) - Y
            dZ = X * Y - beta * Z
            Xn = X + dX * self.dt + q[:, t, 0]
            Yn = Y + dY * self.dt + q[:, t, 1]
            Zn = Z + dZ * self.dt + q[:, t, 2]
            next_s = torch.stack([Xn, Yn, Zn], dim=1)
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
    ) -> list:
        B, num_steps, _ = observations.shape
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, 3))

        current_bg = observations[:, 0].clone() + torch.randn(B, 3, device=self.device) * 1.5

        for w in range(num_windows):
            start = w * self.da_window_steps
            end = start + self.da_window_steps
            win_obs = observations[:, start:end]
            win_mask = obs_mask[:, start:end]
            win_force = forcing[:, start:end]

            x0_ctrl = current_bg.clone().detach().requires_grad_(True)
            q_ctrl = torch.zeros(B, self.da_window_steps, 3, device=self.device, requires_grad=True)
            x_bg_ref = current_bg.clone().detach()

            opt = optim.Adam([x0_ctrl, q_ctrl], lr=self.lr)

            for _ in range(self.opt_steps):
                opt.zero_grad()
                traj = self._forward_weak_batch(x0_ctrl, q_ctrl, self.da_window_steps, start, win_force, sigma, rho, beta, c1)
                J_b = torch.sum((x0_ctrl - x_bg_ref) ** 2) / self.B_var
                J_q = torch.sum(q_ctrl ** 2) / self.Q_var
                diff = traj - win_obs
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o + 0.5 * J_q
                J_total.backward()
                opt.step()

            final_traj = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            next_forecast = self._forward_weak_batch(
                x0_ctrl.detach(), q_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            current_bg = next_forecast[:, -1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
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
        coupling_type: str = "linear",
    ):
        self.da_window_steps = da_window_steps
        self.B_var = B_var
        self.R_var = R_var
        self.max_iter = max_iter
        self.lr = lr
        self.dt = dt
        self.device = device
        self.coupling_type = coupling_type

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
    ) -> BaselineResult:
        num_steps = observations.shape[0]
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((num_steps, 3))

        current_bg = observations[0].clone() + torch.randn(3, device=self.device) * 1.5

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
                traj = self._forward_strong(x_ctrl, self.da_window_steps, start, win_force, sigma, rho, beta, c1)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                J_o = torch.tensor(0.0, device=self.device)
                for t in range(self.da_window_steps):
                    if win_mask[t]:
                        J_o += torch.sum((traj[t] - win_obs[t]) ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o
                J_total.backward()
                return J_total

            for _ in range(4):
                opt.step(closure)

            final_traj = self._forward_strong(
                x_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            analysis[start:end] = final_traj.detach().cpu().numpy()
            current_bg = final_traj[-1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse)

    def _forward_strong(self, x0, steps, start_idx, forcing, sigma, rho, beta, c1, clip_range=50.0):
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            X, Y, Z = s[0], s[1], s[2]
            W = forcing[t - 1]
            dX = sigma * (Y - X) + _apply_coupling(W, c1, self.coupling_type)
            dY = X * (rho - Z) - Y
            dZ = X * Y - beta * Z
            next_s = torch.stack([X + dX * self.dt, Y + dY * self.dt, Z + dZ * self.dt])
            if clip_range is not None:
                next_s = torch.clamp(next_s, -clip_range, clip_range)
            traj.append(next_s)
        return torch.stack(traj)

    def _forward_strong_batch(self, x0, steps, start_idx, forcing, sigma, rho, beta, c1, clip_range=50.0):
        B = x0.shape[0]
        traj = [x0]
        for t in range(1, steps):
            s = traj[-1]
            X, Y, Z = s[:, 0], s[:, 1], s[:, 2]
            W = forcing[:, t - 1]
            dX = sigma * (Y - X) + _apply_coupling(W, c1, self.coupling_type)
            dY = X * (rho - Z) - Y
            dZ = X * Y - beta * Z
            next_s = torch.stack([X + dX * self.dt, Y + dY * self.dt, Z + dZ * self.dt], dim=1)
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
    ) -> list:
        B, num_steps, _ = observations.shape
        num_windows = num_steps // self.da_window_steps
        analysis = np.zeros((B, num_steps, 3))

        current_bg = observations[:, 0].clone() + torch.randn(B, 3, device=self.device) * 1.5

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
                traj = self._forward_strong_batch(x_ctrl, self.da_window_steps, start, win_force, sigma, rho, beta, c1)
                J_b = torch.sum((x_ctrl - x_bg_ref) ** 2) / self.B_var
                diff = traj - win_obs
                masked_diff = diff * win_mask.unsqueeze(-1)
                J_o = torch.sum(masked_diff ** 2) / self.R_var
                J_total = 0.5 * J_b + 0.5 * J_o
                J_total.backward()
                opt.step()

            final_traj = self._forward_strong_batch(
                x_ctrl.detach(), self.da_window_steps, start, win_force, sigma, rho, beta, c1
            )
            analysis[:, start:end] = final_traj.detach().cpu().numpy()
            current_bg = final_traj[:, -1].detach()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
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
        coupling_type: str = "linear",
    ):
        self.N_ensemble = N_ensemble
        self.R_var = R_var
        self.inflation = inflation
        self.dt = dt
        self.device = device
        self.coupling_type = coupling_type

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
    ) -> BaselineResult:
        num_steps = observations.shape[0]
        N = self.N_ensemble
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)

        ensemble = observations[0].clone().unsqueeze(0).repeat(N, 1)
        ensemble += torch.randn((N, 3), device=self.device) * 1.5

        analysis = np.zeros((num_steps, 3))
        ens_var = np.zeros((num_steps, 3))
        analysis[0] = torch.mean(ensemble, dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble, dim=0).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[t - 1]
            Xe, Ye, Ze = ensemble[:, 0], ensemble[:, 1], ensemble[:, 2]
            dX = sigma * (Ye - Xe) + _apply_coupling(W, c1, self.coupling_type)
            dY = Xe * (rho - Ze) - Ye
            dZ = Xe * Ye - beta * Ze
            ensemble[:, 0] += dX * self.dt
            ensemble[:, 1] += dY * self.dt
            ensemble[:, 2] += dZ * self.dt

            if obs_mask[t]:
                y_t = observations[t]
                mu = torch.mean(ensemble, dim=0)
                A = ensemble - mu
                Y = A
                dy = y_t - mu

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

            analysis[t] = torch.mean(ensemble, dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble, dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, ensemble=np.zeros((N, num_steps, 3)), ensemble_variance=ens_var)

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
    ) -> list:
        B, num_steps, _ = observations.shape
        N = self.N_ensemble
        N1 = N - 1
        R_sym_sqrt_inv = 1.0 / np.sqrt(self.R_var)

        ensemble = observations[:, 0].clone().unsqueeze(1).repeat(1, N, 1)
        ensemble += torch.randn((B, N, 3), device=self.device) * 1.5

        analysis = np.zeros((B, num_steps, 3))
        ens_var = np.zeros((B, num_steps, 3))
        analysis[:, 0] = torch.mean(ensemble, dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble, dim=1).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[:, t - 1, None]
            Xe, Ye, Ze = ensemble[:, :, 0], ensemble[:, :, 1], ensemble[:, :, 2]
            dX = sigma * (Ye - Xe) + _apply_coupling(W, c1, self.coupling_type)
            dY = Xe * (rho - Ze) - Ye
            dZ = Xe * Ye - beta * Ze
            ensemble[:, :, 0] += dX * self.dt
            ensemble[:, :, 1] += dY * self.dt
            ensemble[:, :, 2] += dZ * self.dt

            if obs_mask[:, t].any():
                for b in range(B):
                    if not obs_mask[b, t]:
                        continue
                    ens_b = ensemble[b]
                    y_t = observations[b, t]
                    mu = torch.mean(ens_b, dim=0)
                    A = ens_b - mu
                    Y = A
                    dy = y_t - mu

                    Y_w = Y * R_sym_sqrt_inv

                    U, s, Vt = torch.linalg.svd(Y_w, full_matrices=False)
                    s2 = s ** 2
                    d = s2 + N1

                    Pw = U @ torch.diag(1.0 / d) @ U.T
                    T = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T

                    R_inv = 1.0 / self.R_var
                    w = (dy * R_inv) @ Y.T @ Pw

                    ens_b = mu + w @ A + T @ A
                    mu = torch.mean(ens_b, dim=0)
                    ensemble[b] = mu + self.inflation * (ens_b - mu)

            analysis[:, t] = torch.mean(ensemble, dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble, dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b,
                ensemble=np.zeros((N, num_steps, 3)),
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
        coupling_type: str = "linear",
    ):
        self.N_ensemble = N_ensemble
        self.R_var = R_var
        self.inflation = inflation
        self.dt = dt
        self.device = device
        self.coupling_type = coupling_type

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
    ) -> BaselineResult:
        num_steps = observations.shape[0]
        ensemble = observations[0].clone().unsqueeze(0).repeat(self.N_ensemble, 1)
        ensemble += torch.randn((self.N_ensemble, 3), device=self.device) * 1.5

        analysis = np.zeros((num_steps, 3))
        ens_var = np.zeros((num_steps, 3))
        analysis[0] = torch.mean(ensemble, dim=0).cpu().numpy()
        ens_var[0] = torch.var(ensemble, dim=0).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[t - 1]
            Xe, Ye, Ze = ensemble[:, 0], ensemble[:, 1], ensemble[:, 2]
            dX = sigma * (Ye - Xe) + _apply_coupling(W, c1, self.coupling_type)
            dY = Xe * (rho - Ze) - Ye
            dZ = Xe * Ye - beta * Ze
            ensemble[:, 0] += dX * self.dt
            ensemble[:, 1] += dY * self.dt
            ensemble[:, 2] += dZ * self.dt

            if obs_mask[t]:
                y_t = observations[t]
                mean_e = torch.mean(ensemble, dim=0)
                A = ensemble - mean_e
                P_b = (A.T @ A) / (self.N_ensemble - 1)
                R = torch.eye(3, device=self.device) * self.R_var
                K = P_b @ torch.inverse(P_b + R)
                for n in range(self.N_ensemble):
                    perturbed = y_t + torch.randn(3, device=self.device) * np.sqrt(self.R_var)
                    ensemble[n] += K @ (perturbed - ensemble[n])

                mean_e = torch.mean(ensemble, dim=0)
                ensemble = mean_e + self.inflation * (ensemble - mean_e)

            analysis[t] = torch.mean(ensemble, dim=0).detach().cpu().numpy()
            ens_var[t] = torch.var(ensemble, dim=0).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        rmse = np.sqrt(np.mean((analysis - ref) ** 2, axis=0))
        return BaselineResult(trajectory=analysis, rmse=rmse, ensemble=np.zeros((self.N_ensemble, num_steps, 3)), ensemble_variance=ens_var)

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
    ) -> list:
        B, num_steps, _ = observations.shape
        ensemble = observations[:, 0].clone().unsqueeze(1).repeat(1, self.N_ensemble, 1)
        ensemble += torch.randn((B, self.N_ensemble, 3), device=self.device) * 1.5

        analysis = np.zeros((B, num_steps, 3))
        ens_var = np.zeros((B, num_steps, 3))
        analysis[:, 0] = torch.mean(ensemble, dim=1).cpu().numpy()
        ens_var[:, 0] = torch.var(ensemble, dim=1).cpu().numpy()

        for t in range(1, num_steps):
            W = forcing[:, t - 1, None]
            Xe, Ye, Ze = ensemble[:, :, 0], ensemble[:, :, 1], ensemble[:, :, 2]
            dX = sigma * (Ye - Xe) + _apply_coupling(W, c1, self.coupling_type)
            dY = Xe * (rho - Ze) - Ye
            dZ = Xe * Ye - beta * Ze
            ensemble[:, :, 0] += dX * self.dt
            ensemble[:, :, 1] += dY * self.dt
            ensemble[:, :, 2] += dZ * self.dt

            if obs_mask[:, t].any():
                y_t = observations[:, t]
                mean_e = torch.mean(ensemble, dim=1)
                A = ensemble - mean_e.unsqueeze(1)
                P_b = (A.transpose(1, 2) @ A) / (self.N_ensemble - 1)
                R = torch.eye(3, device=self.device).unsqueeze(0) * self.R_var
                K = P_b @ torch.inverse(P_b + R)
                for n in range(self.N_ensemble):
                    perturbed = y_t + torch.randn((B, 3), device=self.device) * np.sqrt(self.R_var)
                    ensemble[:, n] += (K @ (perturbed - ensemble[:, n]).unsqueeze(-1)).squeeze(-1)

                mean_e = torch.mean(ensemble, dim=1)
                ensemble = mean_e.unsqueeze(1) + self.inflation * (ensemble - mean_e.unsqueeze(1))

            analysis[:, t] = torch.mean(ensemble, dim=1).detach().cpu().numpy()
            ens_var[:, t] = torch.var(ensemble, dim=1).detach().cpu().numpy()

        ref = observations.cpu().numpy() if true_state is None else true_state.cpu().numpy()
        results = []
        for b in range(B):
            rmse_b = np.sqrt(np.mean((analysis[b] - ref[b]) ** 2, axis=0))
            results.append(BaselineResult(
                trajectory=analysis[b], rmse=rmse_b,
                ensemble=np.zeros((self.N_ensemble, num_steps, 3)),
                ensemble_variance=ens_var[b],
            ))
        return results
