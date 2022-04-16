import gym
import numpy as np

env=gym.make("MountainCar-v0")

class BespokeAgent:
    def __init__(self,env):
        pass
    def decide(self,observation):
        position,velocity=observation
        lb=min(-0.09*(position+0.25)**2+0.03,0.3*(position+0.9)**4-0.008)
        ub=-0.07*(position+0.38)**2+0.07
        if lb<velocity<ub:
            action=2
        else:
            action=0
        return action
    
    def learn(self,*args):
        pass

agent=BespokeAgent(env)

def play_montecarlo(env,agent,render=False,train=False):
    episode_reward=0
    observation=env.reset()
    while True:
        if render:
            env.render()
        action=agent.decide(observation)
        next_observation,reward,done,_=env.step(action)
        episode_reward+=reward
        if train:
            agent.learn(observation,action,reward,done)
        if done:
            break
        observation=next_observation
    return episode_reward

# env.seed(0)
episode_rewards=[play_montecarlo(env,agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))
env.close()
