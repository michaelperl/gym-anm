"""An MPC-based policy with constant forecasts."""
import numpy as np

from .mpc_constant import MPCAgentConstant


class MPCAgentConstantMA(MPCAgentConstant):
    """
    A Model Predictive Control (MPC)-based policy with constant forecasts.

    This class implements the :math:`\\pi_{MPC-N}^{constant}` policy, a variant
    of the general :math:`\\pi_{MPC-N}` policy in which the future demand and
    generation are assumed constant over the optimization horizon.

    For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_gen_non_slack_ids = list(set(self.non_slack_gen_ids) & set(self.agent.device_keys))
        self.agent_des_ids = list(set(self.des_ids) & set(self.agent.device_keys))
        self.action_space = self.agent.action_space

    def ma_act(self, env):
        """
        Select an action by solving the N-stage DC OPF.

        Parameters
        ----------
        env : py:class:`gym_anm.ANMEnv`
            The :code:`gym-anm` environment.

        Returns
        -------
        numpy.ndarray
            The action vector to apply in the environment.
        """
        P_load_forecasts, P_gen_forecasts = self.forecast(env)
        act = self._solve(env.simulator, P_load_forecasts, P_gen_forecasts)
        N_slack_gen = self.n_gen - 1

        P_gen = act[:N_slack_gen]
        Q_gen = act[N_slack_gen: 2 *N_slack_gen]
        P_des = act[2 * N_slack_gen: 2 * N_slack_gen + self.n_des]
        Q_des = act[2 * N_slack_gen + self.n_des:]
        a = []
        for ii, dev_id in enumerate(self.non_slack_gen_ids):
            if dev_id in self.agent_gen_non_slack_ids:
                a.append(P_gen[ii])
        for ii, dev_id in enumerate(self.non_slack_gen_ids):
            if dev_id in self.agent_gen_non_slack_ids:
                a.append(Q_gen[ii])
        for ii, dev_id in enumerate(self.des_ids):
            if dev_id in self.agent_des_ids:
                a.append(P_des[ii])
        for ii, dev_id in enumerate(self.des_ids):
            if dev_id in self.agent_des_ids:
                a.append(Q_des[ii])


        # Clip the actions, which are sometime beyond the space by a tiny
        # amount due to precision errors in the optimization problem
        # solution (e.g., of the order of 1e-10).
        a = np.clip(a, self.action_space.low, self.action_space.high)

        return a