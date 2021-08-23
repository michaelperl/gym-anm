from ..simulator.components.constants import STATE_VARIABLES
from gym import spaces

class Agent(object):
    def __init__(self, name, ind, observation, device_keys , simulator, reward_callback=None):
        self.name = name
        self.index = ind

        # Build observation space.
        self.obs_values = self._build_observation_space(observation)
        self.observation_space = self.observation_bounds()
        if self.observation_space is not None:
            self.observation_N = self.observation_space.shape[0]
        self.action_space = self._build_action_space(device_keys , simulator)
        self.device_keys = device_keys
        self.reward_callback = reward_callback



    def _build_observation_space(self, observation, state_values):
        """Handles the different ways of specifying an observation space."""

        # Case 1: environment is fully observable.
        if isinstance(observation, str) and observation == 'state':
            obs_values = deepcopy(self.state_values)

        # Case 2: observation space is provided as a list.
        elif isinstance(observation, list):
            obs_values = deepcopy(observation)
            # Add default units when none is provided.
            for idx, o in enumerate(obs_values):
                if len(o) == 2:
                    obs_values[idx] = tuple(list(o) + [STATE_VARIABLES[o[0]][0]])

        # Case 3: observation space is provided as a callable object.
        elif callable(observation):
            obs_values = None
            self.observation = observation

        else:
            raise ObsSpaceError()

        # Transform the 'all' option into a list of bus/branch/device IDs.
        return self._expand_all_ids(obs_values)

    def _expand_all_ids(self, values):
        """Helper function to translate the 'all' option in a list of IDS."""

        # Transform the 'all' option into a list of bus/branch/device IDs.
        if values is not None:
            for idx, o in enumerate(values):
                if isinstance(o[1], str) and o[1] == 'all':
                    if 'bus' in o[0]:
                        ids = list(self.simulator.buses.keys())
                    elif 'dev' in o[0]:
                        ids = list(self.simulator.devices.keys())
                    elif 'des' in o[0]:
                        ids = [i for i, d in self.simulator.devices.items()
                               if isinstance(d, StorageUnit)]
                    elif 'gen' in o[0]:
                        ids = [i for i, d in self.simulator.devices.items()
                               if isinstance(d, Generator) and not d.is_slack]
                    elif 'branch' in o[0]:
                        ids = list(self.simulator.branches.keys())
                    elif o[0] == 'aux':
                        ids = list(range(0, self.K))
                    else:
                        raise ObsNotSupportedError(o[0], STATE_VARIABLES.keys())

                    values[idx] = (o[0], ids, o[2])

        return values

    def observation_bounds(self):
        """
        Builds the observation space of the environment.

        If the observation space is specified as a callable object, then its
        bounds are set to :code:`(- np.inf, np.inf)^{N_o}` by default (this is done
        during the :py:func:`reset()` call, as the size of observation vectors is not
        known before then). Alternatively, the user can specify their own bounds
        by overwriting this function in a subclass.

        Returns
        -------
        gym.spaces.Box or None
            The bounds of the observation space.
        """
        lower_bounds, upper_bounds = [], []

        if self.obs_values is None:
            logger.warning('The observation space is unbounded.')
            # In this case, the size of the obs space is obtained after the
            # environment has been reset. See `reset()`.
            return None

        else:
            bounds = self.simulator.state_bounds
            for key, nodes, unit in self.obs_values:
                for n in nodes:
                    if key == 'aux':
                        if self.aux_bounds is not None:
                            lower_bounds.append(self.aux_bounds[n][0])
                            upper_bounds.append(self.aux_bounds[n][1])
                        else:
                            lower_bounds.append(-np.inf)
                            upper_bounds.append(np.inf)
                    else:
                        lower_bounds.append(bounds[key][n][unit][0])
                        upper_bounds.append(bounds[key][n][unit][1])

        space = spaces.Box(low=np.array(lower_bounds),
                           high=np.array(upper_bounds),
                           dtype=np.float64)

        return space

    def _build_action_space(self, device_keys , simulator):
        """
        Build the available loose action space :math:`\mathcal A`.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds = \
            simulator.get_action_space()

        lower_bounds, upper_bounds = [], []
        for x in [P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds]:
            device_ids = list(set(x.keys()) & set(device_keys)) # TODO test
            for dev_id in sorted(device_ids):
                lower_bounds.append(x[dev_id][0])
                upper_bounds.append(x[dev_id][1])

        space = spaces.Box(low=np.array(lower_bounds),
                           high=np.array(upper_bounds),
                           dtype=np.float64)

        return space