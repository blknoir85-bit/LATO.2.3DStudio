import numpy as np
import torch


class VertFlowEulerCfgSampler:
    def _pred_v(self, model, x_t, t, cond, **kwargs):
        t_vec = torch.full(
            (x_t.shape[0],), 1000.0 * t, device=x_t.device, dtype=torch.float32
        )
        return model(x_t, t_vec, cond, **kwargs)

    def _cfg_v(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        v_cond = self._pred_v(model, x_t, t, cond, **kwargs)
        v_uncond = self._pred_v(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * v_cond - cfg_strength * v_uncond

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps=12,
        cfg_strength=3.0,
        rescale_t=1.0,
        solver="euler",
        **kwargs,
    ):
        x = noise
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        for i in range(steps):
            t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
            dt = t - t_prev
            v1 = self._cfg_v(model, x, t, cond, neg_cond, cfg_strength, **kwargs)
            x_euler = x - dt * v1
            if solver == "heun" and i < steps - 1:
                v2 = self._cfg_v(model, x_euler, t_prev, cond, neg_cond, cfg_strength, **kwargs)
                x = x - dt * (0.5 * (v1 + v2))
            else:
                x = x_euler
        return x


class TopoFlowEulerSampler:
    def _pred_v(self, model, x_t, t, verts, mask, cond, cond_mask):
        t_vec = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
        return model(x_t, t_vec, verts=verts, mask=mask, cond=cond, cond_mask=cond_mask)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        verts,
        mask,
        cond=None,
        cond_mask=None,
        steps=50,
        solver="euler",
    ):
        x = noise
        t_seq = np.linspace(0.0, 1.0, steps + 1)
        for i in range(steps):
            t, t_next = float(t_seq[i]), float(t_seq[i + 1])
            dt = t_next - t
            v1 = self._pred_v(model, x, t, verts, mask, cond, cond_mask)
            x_euler = x + dt * v1
            if solver == "heun" and i < steps - 1:
                v2 = self._pred_v(model, x_euler, t_next, verts, mask, cond, cond_mask)
                x = x + dt * (0.5 * (v1 + v2))
            else:
                x = x_euler
        return x
