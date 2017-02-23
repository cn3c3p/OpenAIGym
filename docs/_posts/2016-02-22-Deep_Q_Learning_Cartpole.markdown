---
layout: default
title:  "Deep Q Learning on Cartpole V1"
date:   2016-02-12 17:50:00
categories: main
---

# Deep Q Learning with Cartpole-v1
## Q Learning
The goal of the agent is to maximize the cumulative rewards it sees. One way to do this is by using something called a Q function. Using a Q function, we can directly go from states to actions. Our policy would then be selecting the action with the greatest value at any given state. The Q function denoted as \\( Q(s,a)  \\) gives the value of taking a discrete action a at some state s. In order to update the Q function,
we employ the update rule
\\[Q(s,a) \leftarrow Q(s, a) + \alpha \left[r + \gamma * \max_{a'} Q(s', a') - Q(s,a)\right]\\]

to iteratively update the function. This equation essentially tells us to use the current reward we receive from taking action a from s and then adding the
next possible best state action value from state s' which we end up in after taking action a from s. Since we take the maximum value for our estimation, we are performing off policy learning since we are approximating the optimal Q function and therefore not care about any other action values.
## Training
In the cartpole problem, we are presented with a continuous state space with 2 discrete actions. In order to represent our Q function, we'll have to use a function
approximator in order to give us a general idea about the values of similar states. In this case, we use a deep neural network which has 8 features as input and outputs the values of the 2 actions separately. Our Q function approximator is denoted as \\( Q(s,a, \theta) \\) where $\theta$ is the parameters of our model.
To update, we minimize the squared error loss function expressed as
$$L = \Vert r + \gamma * \max_{a'}Q(s', a') - Q(s,a) \Vert^2 $$
To minimize this loss function, the gradients are computed with respect to the parameters $\theta$ and a gradient descent operation is performed which moves the parameters towards a value that outputs a smaller loss.

During the simulation, we collect tuples of $(s,a,r,s')$

Thus, in order to implement our update, we compute a forward pass from our current states s to receive $Q(s,a)$ for each action.


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh].

[jekyll-gh]: https://github.com/mojombo/jekyll
[jekyll]:    http://jekyllrb.com
