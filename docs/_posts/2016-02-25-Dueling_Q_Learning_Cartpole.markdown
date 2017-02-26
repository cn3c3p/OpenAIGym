---
layout: default
title:  "Dueling Q Learning on Cartpole V1"
date:   2017-02-25 17:50:00
categories: main
---

# Dueling Q Network for Cartpole-V1

<div class="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/jTyR-mEYCuk?autoplay=1" frameborder="0" allowfullscreen></iframe>
</div>

## Double Dueling Q Network
To understand what the Dueling Q network is, it is a good idea to review what Advantage functions are. Advantage functions are a way to add a bias term to the reward expectation equation and yet not change the expectation. To see this, we write
\\[ J(\theta) = \mathop{\mathbb{E}}_{\pi^\theta}[r] \\]
\\[= \sum_s^S d(s) \sum_a^A \pi^\theta(s,a)R(s,a)\\]
as an expression of the average amount of reward the agent will receive based on the policy it is following. We want to maximize this. In order to do this, we compute the gradient with respect to $$\theta$$ and move in the direction towards maximum reward. Computing the gradient is expressed as:
\\[\nabla J(\theta) = \sum_s^S d(s) \sum_a^A \nabla \pi^\theta(s,a)R(s,a)\\]
\\[= \sum_s^S d(s) \sum_a^A \frac{\pi^\theta(s,a)}{\pi^\theta(s,a)} \nabla \pi^\theta(s,a)R(s,a)\\]
Since we know that
\\[\nabla\log(\pi^\theta(s,a)) = \frac{1}{\pi^\theta(s,a)} \nabla\pi^\theta(s,a)\\]
\\[\nabla\pi^\theta(s,a) = \nabla \log(\pi^\theta(s,a)) \pi^\theta(s,a)\\]
We can substitute this result into $$\nabla J(\theta)$$ to obtain:
\\[\nabla J(\theta) = \sum_s^S d(s) \sum_a^A \frac{\pi^\theta(s,a)}{\pi^\theta(s,a)} \nabla \log(\pi^\theta(s,a)) \pi^\theta(s,a)R(s,a)\\]
\\[=\sum_s^S d(s) \sum_a^A \pi^\theta(s,a) \nabla \log(\pi^\theta(s,a)) R(s,a)\\]
\\[=\mathop{\mathbb{E}}[\nabla \log(\pi^\theta(s,a)) R(s,a)]\\]
The idea is that we can then replace our instantaneous reward function $$R(s,a)$$ with a long term state action value function $$Q^\pi(s,a)$$ and try to maximize $$J(\theta)$$. Now, just using $$Q(s,a)$$ doesn't tell you how good an action is with respect to the other actions you can take at your current state. So, a natural extension is to average the values of the actions taken at the state and compare your current action with the average. The average function is nothing but the value function $$V(s)$$. Thus, we define the advantage function as $$A(s,a) = Q(s,a) - V(s)$$. If we plug this into our new formula for $$J(\theta)$$, we get:
\\[J'(\theta) = \mathop{\mathbb{E}}[A(s,a)]\\]
\\[=\mathop{\mathbb{E}}[Q(s,a) - V(s)]\\]
$$=\mathop{\mathbb{E}}[Q(s,a)] - \mathop{\mathbb{E}}[V(s)]$$

The key insight here is that we can still use the exact same gradient in order to maximize this new function.
\\[\nabla J'(\theta) = \sum_s^S d(s) \sum_a^A \nabla \pi^\theta(s,a)R(s,a) - \sum_s^S d(s) \sum_a^A \nabla \pi^\theta(s,a) V(s)\\]
\\[= \nabla J(\theta) - \sum_s^S d(s)V(s) \nabla^\theta \sum_a^A \pi^\theta(s,a)\\]
\\[= \nabla J(\theta) - \sum_s^S d(s)V(s) \nabla^\theta * 1 \\]
\\[= \nabla J(\theta) - \sum_s^S d(s)V(s) * 0 = \nabla J(\theta)\\]

Subtracting $$V(s)$$ from $$Q(s,a)$$ reduces our variance for our gradients and doesn't change its expectation.

<b>Tl;dr Advantage function tells you the performance of an action as compared to the other action options <b>

So the idea of the Dueling Q Network is to approximate the advantage function $$A(s)$$ and the value function $$V(s)$$. The function can then be combined such that $$Q(s,a) = A(s,a) + V(s)$$. A little trick that they did in the paper is to output the advantage as $$ A'(s,a) =  A(s,a) - \frac{1}{\vert A \vert} \sum_{a'}A(s,a') $$. I implemented this as a custom layer and merged the value output with this to form the final output for $$Q(s,a)$$.

{% highlight python %}

class Advantage_Layer(Dense):
	def call(self, x, mask=None):
		advantage = super(Advantage_Layer, self).call(x, mask)
		mean_adv = mean(advantage)
		return advantage - mean_adv

{% endhighlight %}

The Dueling Q Network is updated the same way as it is for a Q network. Additionally it can benefit from all the insights and techniques used to bolster the Q network. One of these techniques is the Double technique where one network proposes an action and another network evaluates the value of that action. This value is then used as a target.

The target is written as:
$$
y_j = \left\{
  \begin{aligned}
    r \qquad \text{if s' is terminal}\\
    r + \gamma * Q(s', \max_{a'} Q(s', a', \theta), \theta^{-}) \qquad \text{otherwise}
  \end{aligned}
\right.
$$
This technique addresses the problem of over estimation where a network is too optimistic about the values it produces.

## Exploration
The network implements an $$\epsilon$$ greedy exploration policy which starts at 1 and is annealed to 0.1 linearly as more experience is added to the experience buffer. The actions are chosen uniformly when agent is under exploration. As exploration is annealed, the cumulative rewards per episode increases. We stop exploring once the mean rewards reaches some value and then take optimal actions from then on.

## Training
As mentioned previously, I use a target network and an update network. The update network proposes the maximum actions for the next states. The target network provides the values for those actions. My update function thus looks like:
{% highlight python %}
def update(self, Q_net=None):
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

  # Double Deep Q
  Q_curr = self.model.predict_on_batch(
    x=np.asarray(states)
  ) # Q(s,a)

  Q_next_me = self.model.predict_on_batch(
    x=np.asarray(states_next)
  ) # Q(s', a')

  if not Q_net:
    Q_next = self.model.predict_on_batch(
      x=np.asarray(states_next)
    ) # Q'(s', a')
  else:
    Q_next = Q_net.model.predict_on_batch(
      x=np.asarray(states_next)
    ) # Q(s', a')

  next_max_as = [np.argmax(values) for values in Q_next_me] # max_a Q(s', a)

  max_Q_next = [values[max_a] for values, max_a in zip(Q_next, next_max_as)] # Q'(s', max_a Q(s',a))

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
The error graph is displayed below:
![Error graph](https://cloud.githubusercontent.com/assets/4509894/23336116/cf25ed32-fb7a-11e6-911a-720366b60491.png)
