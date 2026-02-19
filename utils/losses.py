import torch

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # pretraining
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss
def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    return brat_hjivi_loss


def init_temporal_consistency_loss(tc_loss_weight, tc_anchor_weight, tc_t0_mode):
    if tc_t0_mode not in ['weighted', 'fixed', 'off']:
        raise RuntimeError("tc_t0_mode must be one of: weighted, fixed, off.")

    def temporal_consistency_loss(v_curr, dvdt, dvds, tc_obs_flow, v_t0=None, tc_t0_boundary=None):
        flow_dot = torch.sum(dvds * tc_obs_flow, dim=-1)
        flow_residual = dvdt + torch.minimum(torch.zeros_like(flow_dot), flow_dot)
        tc_backup = torch.mean(torch.pow(flow_residual, 2))

        if tc_t0_mode == 'off':
            tc_anchor = torch.zeros((), device=v_curr.device, dtype=v_curr.dtype)
            boundary_weight = 0.0
        else:
            if v_t0 is None or tc_t0_boundary is None:
                raise RuntimeError("Temporal consistency with tc_t0_mode!=off requires v_t0 and tc_t0_boundary.")
            tc_anchor = torch.mean(torch.pow(v_t0 - tc_t0_boundary, 2))
            boundary_weight = tc_anchor_weight if tc_t0_mode == 'weighted' else 1.0

        tc_total = tc_loss_weight * tc_backup + boundary_weight * tc_anchor
        return {
            'tc_backup': tc_backup,
            'tc_anchor': tc_anchor,
            'tc_total': tc_total,
        }

    return temporal_consistency_loss
