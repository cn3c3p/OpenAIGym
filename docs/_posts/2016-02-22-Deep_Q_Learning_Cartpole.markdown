---
layout: default
title:  "Deep Q Learning on Cartpole V1"
date:   2016-02-12 17:50:00
categories: main
---

# Deep Q Learning with Cartpole-v1

<iframe width="420" height="315" src="{{site.url}}/assets/Q_Learning_Cartpole_v1.mov" frameborder="0" align="right"></iframe>

## Q Learning
The goal of the agent is to maximize the cumulative rewards it sees. One way to do this is by using something called a Q function. Using a Q function, we can directly go from states to actions. Our policy would then be selecting the action with the greatest value at any given state. The Q function denoted as \\( Q(s,a)  \\) gives the value of taking a discrete action a at some state s. In order to update the Q function,
we employ the update rule
\\[Q(s,a) \leftarrow Q(s, a) + \alpha \left[r + \gamma * \max_{a'} Q(s', a') - Q(s,a)\right]\\]

to iteratively update the function. This equation essentially tells us to use the current reward we receive from taking action a from s and then adding the
next possible best state action value from state s' which we end up in after taking action a from s. Since we take the maximum value for our estimation, we are performing off policy learning since we are approximating the optimal Q function and therefore not care about any other action values.
## Training
In the cartpole problem, we are presented with a continuous state space with 2 discrete actions. In order to represent our Q function, we'll have to use a function
approximator in order to give us a general idea about the values of similar states. In this case, we use a deep neural network which has 8 features as input and outputs the values of the 2 actions separately. Our Q function approximator is denoted as \\( Q(s,a, \theta) \\) where $$\theta$$ is the parameters of our model.
To update, we minimize the squared error loss function expressed as
\\[L = \frac{1}{2}\Vert r + \gamma * \max_{a'}Q(s', a') - Q(s,a) \Vert^2 \\]
We can think of $$r + \gamma * \max_{a'}Q(s', a')$$ as the new target for $$Q(s,a)$$ that we want to nudge towards.
To minimize this loss function, the gradients are computed with respect to the parameters $\theta$ and a gradient descent operation is performed which moves the parameters towards a value that outputs a smaller loss.

During the simulation, we collect tuples of $$(s,a,r,s')$$. These tuples are collected in an experience buffer that the algorithm selects batches from for training. A larger batch size results in a more stable gradient.

Thus, in order to implement our update, we compute a forward pass from our current states s to receive $$Q(s,a)$$ for each action. We then do another forward pass from our next states s' to receive $$Q(s', a')$$ for each action in s'. Then we select the greatest value in s' in order to estimate the best possible value in s' with $$\max_{a'}Q(s',a')$$. So, in order to update the targets for $$Q(s,a)$$ we update the action value outputted from the network while keeping the other action values the same.

In code, the implementation would look something like this:

{% highlight python %}
def update(self):
  if len(self.experiences) > self.batch_size:
    stuff = range(0, len(self.experiences))
    ind = np.random.choice(
      a=stuff,
      size=self.batch_size,
      replace=False
    )
    experiences = [self.experiences[i] for i in ind]
  else:
    experiences = self.experiences
    if len(experiences) == 0:
      return

  states, actions, rewards, states_next, dones = zip(*experiences)

  Q_curr = self.model.predict_on_batch(
    x=np.asarray(states)
  )

  Q_next = self.model.predict_on_batch(
    x=np.asarray(states_next)
  )

  max_Q_next = [values[np.argmax(values)] for values in Q_next]

  targets = list()

  for curr_val, action, reward, next_val, done in zip(Q_curr, actions, rewards, max_Q_next, dones):
    target = curr_val
    if done:
      target[action] = reward
    else:
      target[action] = reward + self.discount_factor * next_val
    targets.append(target)

  self.model.fit(
    x=np.asarray(states),
    y=np.asarray(targets),
    batch_size=self.batch_size,
    nb_epoch=1,
    verbose=0
  )
{% endhighlight %}

## Exploration and Beyond

In order to explore, I have the agent perform $$\epsilon$$-greedy exploration policy which with probability $$\epsilon$$ takes a random action from its current state. This probability gets annealed over time so that the agent takes the action that it thinks is best for its current state.

It is noted that having 2 separate networks for proposing an action and updating provides more stability and less overestimation. Thus, I use one network for proposing an action and one for collecting the experience tuples and updating upon those tuples. Then after some fixed iteration, I then copy over the update network's parameters to the target network.

There is a limit of 500 reward points the environment allows you to have at each episode. I therefore set the reward to be -1 if the agent's episode finished without reaching 500. This would incentivize the agent to continue and avoid early termination.

I evaluate the agent every 50 episodes to gauge its progress. Once the reward limit has been reached, I then allow the agent to make optimal actions every time until the environment is solved.

Note: Adding dropout layers after the dense layer allowed for faster completion time. Dropout effectively acts as a regularizer because we are averaging across a variety of architectures.

[OpenAI-gh]: https://github.com/jchen114/OpenAIGym
[OpenAI]:    http://gym.openai.com
