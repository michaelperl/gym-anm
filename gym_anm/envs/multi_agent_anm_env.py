"""The base class for :code:`multi-agent gym-anm` environments."""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from logging import getLogger
from copy import deepcopy
import warnings
from scipy.sparse.linalg import MatrixRankWarning
from    anm_env import ANMEnv

from ..simulator import Simulator
from ..errors import ObsSpaceError, ObsNotSupportedError, EnvInitializationError, \
    EnvNextVarsError
from .utils import check_env_args
from ..simulator.components.constants import STATE_VARIABLES
from ..simulator.components import StorageUnit, Generator, Load



logger = getLogger(__file__)


class MA_ANMEnv(ANMEnv):
    """
    The base class for :code:`gym-anm` environments.

    Attributes
    ----------
    K : int
        The number of auxiliary variables.
    gamma : float
        The fixed discount factor in [0, 1].
    lamb : int or float
        The factor multiplying the penalty associated with violating
        operational constraints (used in the reward signal).
    delta_t : float
        The interval of time between two consecutive time steps (fraction of
        hour).
    simulator : :py:class:`gym_anm.simulator.simulator.Simulator`
        The electricity distribution network simulator.
    state_values : list of tuple of str
        The electrical quantities to include in the state vectors. Each tuple
        (x, y, z) refers to quantity x aevices/branches y, using units
        z.
    state_N : int
        The number of state variables.
    agents :  agents  in the environment
    N_agents : number of agents in environment
    action_space : gym.spaces.Box
        The joint action space from which the agents can select actions.
    obs_values : list of str or None
        Similarly to :py:obj:`state_values`, the values to include in the observation
        vectors. If a customized :py:func:`observation()` function is provided, :py:obj:`obs_values`
        is None.
    observation_space : gym.spaces.Box TODO
        The observation space from which observation vectors are constructed.
    observation_N : int
        The number of observation variables.
    done : bool
        True if a terminal state has been reached (if the network collapsed);
        False otherwise.
    render_mode : str
        The rendering mode. See :py:func:`render()`.
    timestep : int
        The current timestep.
    state : numpy.ndarray
        The current state vector :math:`s_t`.
    e_loss : float
        The energy loss during the last transition (part of the reward signal).
    penalty : float
        The penalty associated with violating operational constraints during the
        last transition (part of the reward signal).
    costs_clipping : tuple of float
        The clipping values for the costs (- rewards), where :py:obj:`costs_clipping[0]` is
        the clipping value for the absolute energy loss and :py:obj:`costs_clipping[1]` is
        the clipping value for the constraint violation penalty.
    pfe_converged : bool
        True if the last transition converged to a load flow solution (i.e.,
        the network is stable); False otherwise.
    np_random : numpy.random.RandomState
        The random state/seed of the environment.
    """

    def __init__(self, network, K, delta_t, gamma, lamb, agents,
                 aux_bounds=None, costs_clipping=None, seed=None):
        """
        Parameters
        ----------
        network : dict of {str : numpy.ndarray}
            The network input dictionary describing the power grid.
        K : int
            The number of auxiliary variables.
        delta_t : float
            The interval of time between two consecutive time steps (fraction of
            hour).
        gamma : float
            The discount factor in [0, 1].
        lamb : int or float
            The factor multiplying the penalty associated with violating
            operational constraints (used in the reward signal).
        aux_bounds : numpy.ndarray, optional
            The bounds on the auxiliary internal variables as a 2D array where
            the :math:`k^{th}`-1 auxiliary variable is bounded by
            :py:obj:`[aux_bounds[k, 0], aux_bounds[k, 1]]`. This can be useful if auxiliary
            variables are to be included in the observation vectors and a bounded
            observation space is desired.
        costs_clipping : tuple of float, optional
            The clipping values for the costs in the reward signal, where element
            0 is the clipping value for the energy loss cost and element 1 is the
            clipping value for the constraint-violation penalty (e.g., (1, 100)).
        seed : int, optional
            A random seed.
        """
        super().__init__(self, network, K, delta_t, gamma, lamb,
                 aux_bounds, costs_clipping, seed, observation='state') # observation : callable or list or str The observation space. It can be specified as "state" to construct a fully observable environment

        self.agents = agents
        self.N_agents = len(agents)


    def reset(self):
        """
        Reset the environment.

        If the observation space is provided as a callable object but the
        :py:func:`observation_bounds()` method is not overwritten, then the bounds on the
        observation space are set to :code:`(- np.inf, np.inf)` here (after the size of the
        observation vectors is known).

        Returns
        -------
        obs : numpy.ndarray
            The initial observation vector.
        """
        self.done = False
        self.render_mode = None
        self.timestep = 0
        self.e_loss = 0.
        self.penalty = 0.

        # Initialize the state.
        init_state_found = False
        n_init_states = 0
        n_init_states_max = 100
        while not init_state_found:
            n_init_states += 1
            self.state = self.init_state()

            # Check s_0 has the correct size.
            expected = 2 * self.simulator.N_device + self.simulator.N_des \
                       + self.simulator.N_non_slack_gen + self.K
            if self.state.size != expected:
                msg = "Expected size of initial state s0 is %d but actual is %d" \
                      % (expected, self.state.size)
                raise EnvInitializationError(msg)

            # Apply the initial state to the simulator.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', MatrixRankWarning)
                init_state_found = self.simulator.reset(self.state)

            if n_init_states == n_init_states_max:
                msg = "No non-terminal state found out of %d initial states for " \
                      "environment %s" % (n_init_states_max, self.__name__)
                raise EnvInitializationError(msg)

        # Reconstruct the sate vector in case the original state was infeasible.
        self.state = self._construct_state()

        # Construct the initial observation vector.
        obs_n = []
        for agent in self.agents:
            obs_n.append(self.agent_observation(agent,self.state))
            if agent.observation_space is None:
                agent.observation_space = spaces.Box(low=-np.ones(len(obs_n[-1])) * np.inf,
                                                    high=np.ones(len(obs_n[-1])) * np.inf)
                agent.observation_N = agent.observation_space.shape[0]

            err_msg = "Observation %r (%s) invalid of agent %s." % (obs_n[-1], type(obs_n[-1]), agent.name)
            assert agent.observation_space.contains(obs_n[-1]), err_msg

        # Update the observation space bounds if required.


        # Cast state and obs vectors to 0 (arbitrary) if a terminal state has
        # been reached.
        if self.done:
            self.state = self._terminal_state(self.state_N)
            obs_n = self._terminal_state_agents()
        return obs_n


    def agent_observation(self, agent, s_t):
        """
        Returns the agent's observation vector corresponding to the current state :math:`s_t`.


        Parameters
        ----------
        agent : Agent
            the agent who's observation is being extracted
        s_t : numpy.ndarray
            The current state vector :math:`s_t`.

        Returns
        -------
        numpy.ndarray
            The corresponding observation vector :math:`o_t`.
        """
        obs = self._extract_state_variables(agent.obs_values)
        obs = np.clip(obs, agent.observation_space.low,
                      agent.observation_space.high)
        return obs


    def step(self, action_n):
        """
        Take a control action and transition from state :math:`s_t` to state :math:`s_{t+1}`.

        Parameters
        ----------
        action_n : numpy.ndarray
            The action vectors of each of the n agents taken by the agents.

        Returns
        -------
        obs_n : numpy.ndarray
            The observation vector of each agent :math:`o_{t+1}`.
        reward_n : float
            The reward associated with the transition of each agent :math:`r_t`.
        done : bool
            True if a terminal state has been reached; False otherwise.
        info : dict
            A dictionary with further information (used for debugging).
        """

        for ii, agent in enumerate(self.agents):
            err_msg = "Action %r (%s) invalid for agent %s." % (action_n(ii), type(action_n(ii)))
            assert agent.action_space.contains(action_n(ii)), err_msg

        obs_n = []
        reward_n = []

        # 0. Remain in a terminal state and output reward=0 if the environment
        # has already reached a terminal state.
        if self.done:
            obs_n = self._terminal_state_agents()
            reward_n = [0] * self.N_agents
            return obs_n, 0., self.done, {}

        # 1a. Sample the internal stochastic variables.
        vars = self.next_vars(self.state)
        expected_size = self.simulator.N_load + self.simulator.N_non_slack_gen \
                        + self.K
        if vars.size != expected_size:
            msg = 'Next vars vector has size %d but expected is %d' % \
                  (vars.size, expected_size)
            raise EnvNextVarsError(msg)

        P_load = vars[:self.simulator.N_load]
        P_pot = vars[self.simulator.N_load: self.simulator.N_load +
                                            self.simulator.N_non_slack_gen]
        aux = vars[self.simulator.N_load + self.simulator.N_non_slack_gen:]
        err_msg = 'Only {} auxiliary variables are generated, but K={} are ' \
                  'expected.'.format(len(aux), self.K)
        assert len(aux) == self.K, err_msg

        # 1b. Convert internal variables to dictionaries.
        load_idx, gen_idx = 0, 0
        P_load_dict, P_pot_dict = {}, {}
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, Load):
                P_load_dict[dev_id] = P_load[load_idx]
                load_idx += 1
            elif isinstance(dev, Generator) and not dev.is_slack:
                P_pot_dict[dev_id] = P_pot[gen_idx]
                gen_idx += 1

        # 2. Extract the different actions from the action vector.
        P_set_points, Q_set_points = self._extract_ma_actions(action_n)

        # 3a. Apply the action in the simulator.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', MatrixRankWarning)
            _, r, e_loss, penalty, pfe_converged = \
                self.simulator.transition(P_load_dict, P_pot_dict, P_set_points,
                                          Q_set_points)

            # A terminal state has been reached if no solution to the power
            # flow equations is found.
            self.done = not pfe_converged

        # 3b. Clip the reward.
        if not self.done:
            self.e_loss = np.sign(e_loss) * np.clip(np.abs(e_loss), 0,
                                                    self.costs_clipping[0])
            self.penalty = np.clip(penalty, 0, self.costs_clipping[1])
            r = - (self.e_loss + self.penalty)
        else:
            # Very large reward if a terminal state has been reached.
            r = - self.costs_clipping[1] / (1 - self.gamma)
            self.e_loss = self.costs_clipping[0]
            self.penalty = self.costs_clipping[1]

        # 4. Construct the state and observation vector.
        if not self.done:
            for k in range(self.K):
                self.state[k-self.K] = aux[k]
            self.state = self._construct_state()
            for agent in self.agents:
                obs_n.append(self.agent_observation(agent,self.state))
                reward_n.append(self._get_reward(agent, r))
                err_msg = "Observation %r (%s) invalid of agent %s." % (obs_n[-1], type(obs_n[-1]), agent.name)
                assert agent.observation_space.contains(obs_n[-1]), err_msg
        # Cast state and obs vectors to 0 (arbitrary) if a terminal state is
        # reached.
        else:
            self.state = self._terminal_state(self.state_N)
            obs_n = self._terminal_state_agents()

        # 5. Update the timestep.
        self.timestep += 1

        return obs_n, reward_n, self.done, {}

    def _extract_ma_actions(self, action_n):
        P_set_points = {}

        Q_set_points = {}
        gen_non_slack_ids = [i for i, dev in self.simulator.devices.items()
                             if isinstance(dev, Generator) and not dev.is_slack]
        des_ids = [i for i, dev in self.simulator.devices.items()
                   if isinstance(dev, StorageUnit)]


        for ii, agent in enumerate(self.agents):
            agent_gen_non_slack_ids = list(set(gen_non_slack_ids) & set(agent.device_keys))
            agent_des_ids = list(set(des_ids) & set(agent.device_keys))
            N_gen = len(agent_gen_non_slack_ids)
            N_des = len(agent_des_ids)
            for a, dev_id in zip(action_n[ii,:N_gen], gen_non_slack_ids):
                P_set_points[dev_id] = a
            for a, dev_id in zip(action_n[ii,N_gen: 2 * N_gen], gen_non_slack_ids):
                Q_set_points[dev_id] = a
            for a, dev_id in zip(action_n[ii,2 * N_gen: 2 * N_gen + N_des], des_ids):
                P_set_points[dev_id] = a
            for a, dev_id in zip(action_n[ii,2 * N_gen + N_des:], des_ids):
                Q_set_points[dev_id] = a

        return  P_set_points, Q_set_points


    def _get_reward(self, agent, r):
        if agent.reward_callback is None:
            return r
        return agent.reward_callback(r)

    def _terminal_state_agents(self):
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._terminal_state(agent.observation_N))
        return obs_n
