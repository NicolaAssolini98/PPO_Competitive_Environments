"""PPO agent script

This manages the training phase of both the discrete and the continuous agents that can perform updates in MC fashion. It is also possible to use gae.
"""

from collections import deque

import yaml
import numpy as np

import matplotlib.pyplot as plt

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(seed)

from utils.deepnetwork import DeepNetwork
from utils.memorybuffer import Buffer


class PPO:
    """
    Class for the PPO agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g., std for the Gaussian)

        Returns:
            None
        """

        self.env = env

        actors = []
        actor_optimizers = []
        critics = []
        critic_optimizers = []
        buffers = []

        for i in range(env.n):
            # da vedere se passsare già i dati o i per avere env.roba[i]
            actor = DeepNetwork.build(env, params['actor'], i, actor=True, name='actor')
            actor_optimizer = Adam()

            critic = DeepNetwork.build(env, params['critic'], i, name='critic')
            critic_optimizer = Adam()

            buffer = Buffer()

            actors.append(actor)
            actor_optimizers.append(actor_optimizer)
            critics.append(critic)
            critic_optimizers.append(critic_optimizer)
            buffers.append(buffer)

        self.actor = np.array(actors)
        self.actor_optimizer = np.array(actor_optimizers)
        self.critic = np.array(critics)
        self.critic_optimizer = np.array(critic_optimizers)
        self.buffer = np.array(buffers)

    def get_action(self, state, index, std=1.0):
        """Get the action to perform

        Args:
            state (list): agent current state
            std (float): std for the Gaussian in cc

        Returns:
            action (list): sampled action to perform, integers or floats
            mu (float): Gaussian's mean in cc
        """

        mu = self.actor[index](np.array([state])).numpy()[0]
        action = np.random.normal(loc=mu, scale=std)
        return action, mu

    def update(self, batch_size, epochs, eps, gamma, std=1.0):
        """Prepare the samples and the cumulative reward to update the network

        Args:
            batch_size (int): batch size
            epochs (int): n° of epochs to perform
            eps (float): clipping value for PPO-clip
            gamma (float): discount factor
            std (float): std for the Gaussian in cc

        Returns:
            None
        """

        for i in range(self.buffer.size):
            buff = self.buffer[i]
            states, probs, actions, rewards, obs_states, dones = buff.sample()
            buffer = [[states[i], probs[i], actions[i], rewards[i], obs_states[i], dones[i]] for i in
                      range(buff.size)]

            np.random.shuffle(buffer)
            batch_size = min(self.buffer.size, batch_size)
            batches = np.array_split(np.array(buffer), int(len(buffer) / batch_size))

            self.update_continuous(batches, epochs, eps, gamma, std, i)

            # After the update, we clear the buffer
            buff.clear()

    def update_continuous(self, batches, epochs, eps, gamma, std, i):
        """PPO does not want the new policy to move too far away from the old one
        ∇θJ(θ) ≈ 1/N * ∑ [min(∑ ∇ ratio(θ)*A, clip(ratio(θ), 1-ε, 1+ε)*A] where the advantage A is computed as desired, and the ratio(θ) = π'θ(a∣s)/πθ(a∣s) is similar to the importance sampling.
        In the continuous case the network outputs mu for a Gaussian distribution that we use as πθ

        Args:
            batches (list): list of minibatches
            epochs (int): n° of epochs to perform
            eps (float): clipping value for PPO-clip
            gamma (float): discount factor
            std (float): std for the Gaussian in cc

        Returns:
            None
        """

        for _ in range(epochs):
            for minibatch in batches:
                states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
                old_mu = np.array([sample[1] for sample in minibatch], dtype=np.float32)
                actions = np.array([sample[2] for sample in minibatch])
                rewards = np.array([sample[3] for sample in minibatch], dtype=np.float32)
                obs_states = np.array([sample[4] for sample in minibatch], dtype=np.float32)
                dones = np.array([sample[5] for sample in minibatch])

                # The updates require shape (n° samples, len(metric))
                rewards = rewards.reshape(-1, 1)
                dones = dones.reshape(-1, 1)

                with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                    # Compute the advantage A as in TD
                    states_values = self.critic[i](states)  # each element is an array with 1 value
                    obs_states_values = self.critic[i](obs_states)

                    # Both critic values and target are differentiable wrt critic.
                    # We have to cast the target to a numpy array
                    # (i.e., we do not want our "label" to be diff)
                    advantages = rewards + gamma * obs_states_values.numpy() * dones
                    advantages = tf.math.subtract(advantages, states_values)

                    # Compute the ratio(θ): π'θ(a∣s)/πθ(a∣s)
                    gauss_d = std * tf.sqrt(2 * np.pi)

                    # Compute πθ(a∣s)
                    gauss_old_n = tf.math.exp(-0.5 * ((actions - old_mu) / std) ** 2)
                    gauss_old_n = tf.cast(gauss_old_n, dtype=np.float32)
                    gauss_old_p = tf.math.divide(gauss_old_n, gauss_d)
                    # We combine the contribution of the actions in case of |actions| > 1
                    # It works also without this, but it optimize the learning process
                    gauss_old_p = tf.math.reduce_mean(gauss_old_p, axis=1, keepdims=True)

                    # Compute π'θ(a∣s)
                    mu = self.actor[i](states)
                    gauss_n = tf.math.exp(-0.5 * ((actions - mu) / std) ** 2)
                    gauss_n = tf.cast(gauss_n, dtype=np.float32)
                    gauss_p = tf.math.divide(gauss_n, gauss_d)
                    gauss_p = tf.math.reduce_mean(gauss_p, axis=1, keepdims=True)

                    ratio = tf.math.divide(gauss_p, gauss_old_p + 1e-10)

                    # Compute the two actor objectives of which we will take the min
                    actor_objective_1 = ratio * advantages
                    actor_objective_2 = tf.clip_by_value(ratio, 1 - eps, 1 + eps) * advantages

                    # Compute the actor objective
                    actor_objective = tf.math.minimum(actor_objective_1, actor_objective_2)
                    actor_loss = -tf.math.reduce_mean(actor_objective)

                    # Compute the actor gradient and update the network
                    actor_grad = tape_a.gradient(actor_loss, self.actor[i].trainable_variables)
                    self.actor_optimizer[i].apply_gradients(zip(actor_grad, self.actor[i].trainable_variables))

                    # Compute the critic loss in TD
                    critic_loss = tf.math.square(advantages)
                    critic_loss = tf.math.reduce_mean(critic_loss)

                    # Compute the critic gradient and update the network
                    critic_grad = tape_c.gradient(critic_loss, self.critic[i].trainable_variables)
                    self.critic_optimizer[i].apply_gradients(zip(critic_grad, self.critic[i].trainable_variables))

    def round_obs(self, obss):
        result = []
        # print(obss)
        for obs in obss:
            obs[:3] *= 0.1  # Normalize the Accelerometer inputs
            obs = np.around(obs, decimals=3)
            result.append(obs)
        #     print(obs)
        # print(result)
        return np.array(result)

    def train(self, tracker, n_episodes, n_step, verbose, params, hyperp):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., std for the Gaussian)
            hyperp (dict): algorithmic specific values (e.g., scaling of the std)

        Returns:
            None
        """

        mean_reward, mean_cost = deque(maxlen=100), deque(maxlen=100)

        std, std_scale = hyperp['std'], hyperp['std_scale']
        std_decay, std_min = params['std_decay'], params['std_min']

        for e in range(n_episodes):
            reward_god_ep = []
            reward_bad_ep = []
            asse_x = []


            ep_reward, steps, cost = 0, 0, 0

            states = self.env.reset()
            badN = 5000000
            for s in states:
                badN = min(badN, s.size)

            states = self.round_obs(states)

            for j in range(n_step):
                asse_x.append(j)
#            while True:
                actions = []
                mulist = []
                for i in range(self.env.n):
                    action, mu = self.get_action(states[i], i, std)
                    actions.append(action)
                    mulist.append(mu)

                actions = np.array(actions)
                mulist = np.array(mulist)

# TODO: DA VERIFICARE

                obs_states, obs_rewards, dones, infos = self.env.step(actions)

                obs_states = self.round_obs(obs_states)
                for i in range(self.env.n):
                    self.buffer[i].store(states[i], mulist[i], actions[i], obs_rewards[i], obs_states[i], 1 - int(dones[i]))

                temp1 = 0
                temp2 = 0
                for i in range(self.env.n):
                    if obs_states[i].size > badN:
                        temp1 += obs_rewards[i]
                        ep_reward += obs_rewards[i]
                    else:
                        temp2 += obs_rewards[i]
                        cost += obs_rewards[i]

                reward_god_ep.append(temp1)
                reward_bad_ep.append(temp2)
                steps += 1

                # costL = []
                # print(infos)
#                for info in infos:
               	#print(infos)
                #     costL.append(info['cost'])
                # cost += np.mean(costL)

                for done in dones:
                    if done: break

                states = obs_states
                

            if e % params['update_freq'] == 0:
                self.update(
                    params['buffer']['batch'],
                    params['n_epochs'],
                    params['eps_clip'],
                    params['gamma'],
                    std
                )

                if std_scale: std = max(std_min, std * std_decay)
                print('----')

#TODO per vedere grafico episodio da da eliminare

#            plt.plot(asse_x, reward_god_ep)
#            f = plt.savefig("imm/G_fig_" + str(e))
#            plt.close(f)
#            plt.plot(asse_x, reward_bad_ep)
#            f = plt.savefig("imm/B_fig_" + str(e))
#            plt.close(f)

            mean_reward.append(ep_reward)
            mean_cost.append(cost)

            tracker.update([e, ep_reward, cost])

            if e % verbose == 0: tracker.save_metrics()

            print(
                f'Ep: {e}, Std: {std:.3f}, Ep_Rew: {ep_reward:.3f}, Ep_Cost: {cost}, Mean_Rew: {np.mean(mean_reward):.3f}, Mean_Cost: {np.mean(mean_cost)}')


