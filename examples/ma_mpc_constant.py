"""
This script shows how to run the MPC-based DC OPF policy
:math:`\\pi_{MPC-N}^{constant}` in an arbitrary gym-anm
environment.

This policy assumes constant demand and generation during the
optimization horizon.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast.
"""
import gym
import time
from gym_anm.agents import MPCAgentConstantMA
from gym_anm.envs.anm6_env.anm6_easy_ma import ANM6Easy_MA

def run():
    env = ANM6Easy_MA()
    o = env.reset()
    print('Environment reset and ready.')

    #initialize MPC policy for each agent
    mpc_agents = []
    for agent in env.agents:
        # Initialize the MPC policy.
        mpc_agent = MPCAgentConstantMA(env.simulator, agent.action_space, env.gamma,
                                 safety_margin=0.96, planning_steps=10, agent=agent)
        mpc_agents.append(mpc_agent)

    T = 50
    start = time.time()

    # Run the policy.
    for t in range(T):
        a_n = []
        for ii, agent in enumerate(env.agents):
            a_n.append(mpc_agents[ii].ma_act(env))
        o, r, _, _ = env.step(a_n)
        print(f't={t}')

    print('Done with {} steps in {} seconds!'.format(T, time.time() - start))

if __name__ == '__main__':
    run()