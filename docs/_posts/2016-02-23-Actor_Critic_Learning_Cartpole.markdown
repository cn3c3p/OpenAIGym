---
layout: default
title:  "Actor Critic Learning on Cartpole V1"
date:   2017-02-23 17:50:00
categories: main
---

# Actor Critic Learning for Cartpole V1

<div class="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/1gyWjyHvPOA" frameborder="0" allowfullscreen></iframe>
</div>

## Actor Critic method
Actor Critic methods are a popular choice for on policy learning which combined a policy gradient update with a value function. Policy gradients are a way to modify your policy to go straight from a state to an action without the need of a value estimate. The critic critiques a state and estimates the value of a state. The actor proposes an action for the state. The actor uses the estimate from the critic to determine the direction to move its actions. This is done through something called a TD error expressed as
\\[\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\\]

One can think of this as $$\widehat V(s_t) = r_{t+1} + \gamma V(s_{t+1})$$ where $$\widehat V(s_t)$$ is an estimate of $$V(s_t)$$ at the current state. A positive TD would mean that the action taken was a good action and the agent should do more of that action. A negative TD would mean that the agent wants to steer away from the action that it just took. The critic learns depending on what the actor does and so this is an example of on policy learning.

In order to update the critic, I collect experience tuples of (s, a, r, s') and store these into separate experience buffers. I have an experience buffer for the actor and one for the critic. When the agent is exploring, I save the experience tuples inside both the critic and actor buffers. Otherwise, I just save the experience into the critic buffer. The intuition to do this is so that the actor can learn from the actions that it has attempted before. Additionally, the problem is small enough such that the critic can use all the experience tuples and not just train on just the tuples gathered from non-exploration. These tuples are then extracted from the experience buffer in batches for training.

Because I use Neural Networks as function approximators with parameters $$\theta$$ to approximate both the state value function and the action, I apply targets to the value function in the form of TD errors and actions to the actor to update their respective parameters through gradient descent. The critic updates itself through TD targets. The actor essentially queries the critic to determine whether or not to update its action towards the action taken at state $$s_t$$. I use positive temporal differencing in order to determine if the actor should update itself. That is, the actor does not weight the action based upon the magnitude of the TD error but treats all actions the same.

## Exploration
Actor exploration is also implemented in an $$\epsilon$$ greedy fashion. Because the actions are discrete, with probability $$\epsilon$$ the actor chooses a random discrete action. In a continuous action space, action exploration may be handled by using an algorithm called CACLA (Continuous Actor Critic Learning Automaton) which essentially searches the action space. The output of the discrete actor is a softmax probability distribution of the actions. The softmax is expressed as:
\\[ \sigma (z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}} \\]
where $$z_j$$ is the action output from the network and the k iterates through all possible actions. We can choose the action stochastically or deterministically through a max depending on how we want to evaluate the action. The actor is then trained using a categorical cross entropy loss, which just nudges the probability of choosing the target action higher.
Exploration is annealed linearly as the number of experiences grows.

## Training
As noted previously, the training is done by periodically selecting random batches from the experience buffer to present to the critic and actor networks. The actor only updates itself if there is a positive TD error from the critic. Otherwise, it ignores the experience altogether.

The update of the Critic looks like this:
{% highlight python %}
def update_critic(self):
  # Shuffle experiences
  if len(self.critic_experiences) > self.batch_size:
    stuff = range(0, len(self.critic_experiences))
    ind = np.random.choice(
      a=stuff,
      size=self.batch_size,
      replace=False
    )
    experiences = [self.critic_experiences[i] for i in ind]
  else:
    experiences = self.critic_experiences
    if len(experiences) == 0:
      return

  states, actions, rewards, states_next = zip(*experiences)

  next_states_values = self.critic.predict_on_batch(
    x=np.asarray(states_next)
  )

  # Flatten..
  next_states_values = [item for sublist in next_states_values for item in sublist]

  targets = [r + self.discount_factor*s_n_val for (r, s_n_val) in zip(rewards, next_states_values)]

  loss = self.critic.train_on_batch(
    x=np.asarray(states),
    y=np.asarray(targets)
  )
{% endhighlight %}

The update for the actor looks like this:

{% highlight python %}
def update_actor(self):
  # Shuffle experiences
  if len(self.actor_experiences) > self.batch_size:

    stuff = range(0, len(self.actor_experiences))

    ind = np.random.choice(
      a=stuff,
      size=self.batch_size,
      replace=False
    )
    experiences = [self.actor_experiences[i] for i in ind]
  else:
    experiences = self.actor_experiences

  if len(experiences) == 0:
    return

  s, a, r, s_n = zip(*experiences)

  current_predictions = self.critic.predict_on_batch(
    x=np.asarray(s)
  )

  next_predictions = self.critic.predict_on_batch(
    x=np.asarray(s_n)
  )

  # Flatten both lists

  current_state_values = [item for sublist in current_predictions for item in sublist]
  next_state_values = [item for sublist in next_predictions for item in sublist]

  current_state_values_p = [reward + self.discount_factor * q_s_n for reward, q_s_n in zip (r, next_state_values)]

  action_indices = [i for i, (q_s_n_p, q_s_p) in enumerate(zip(current_state_values_p, current_state_values)) if q_s_n_p > q_s_p]

  if len(action_indices) == 0:
    return

  targets = [a[action_index] for action_index in action_indices]
  # Modify the action to prepare it for training
  targets = [self.prepare_action_for_training(target) for target in targets]
  batch = [s[action_index] for action_index in action_indices]

  loss = self.actor.train_on_batch(
    x=np.asarray(batch),
    y=np.asarray(targets)
  )
{% endhighlight %}

I evaluate the agent every 50 episodes to check its performance. Once the performance has reached the maximum goal, I then set the agent to take the most optimal action at its current state. An error graph looks like this:

![Error graph](https://cloud.githubusercontent.com/assets/4509894/23288500/cdd90532-f9f8-11e6-84c2-0def8c5463ff.png)
