<h1 align='center'> 🐍 Learn2Slither</h1>

> ⚠️ This tutorial assumes you have done [multilayer-perceptron](https://github.com/leogaudin/multilayer-perceptron) and [dslr](https://github.com/leogaudin/dslr).

## Table of Contents

- [Introduction](#introduction) 👋
- [Resources](#resources) 📖

## Introduction

Learn2Slither introduces a new concept in our machine learning journey: **reinforcement learning**.

Reinforcement learning is used to teach an agent how to behave in an environment by performing actions and observing the rewards it gets from them.

It is appropriate **for problems where it is not possible to have a dataset of examples to learn from**, but where it is possible to interact with the environment and learn from the feedback it provides.

In this guide, we will use a specific type of reinforcement learning called **Deep Q-Learning** to teach an agent how to play the game of Snake.

## Q-Learning

Q-Learning is a reinforcement learning algorithm that associate a "quality" to each action in a given state.

In Snake, an example could be "*If I have a wall in front of me, the quality of the action 'go forward' is very low*". Because you die.

The objective of Q-Learning is basically this: given a state, it must output the best action to take.

Once again, we have an obscure function to represent this:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

Let's demystify this:

- $Q(s, a)$ is the quality of the action $a$ in the state $s$.
- $r$ is the reward of taking action $a$ in the state $s$.
- $\gamma$ is the discount factor (basically, how much we care about the future).
- $s'$ is the next state.
- $a'$ is the next action.
- $\max_{a'} Q(s', a')$ is the maximum predicted quality of the next action in the next state (basically, asking our model the best quality attainable in the next state, how well the action we just took will lead us to a good outcome).

## How to start

So, we know we have to use Q-Learning, and that this algorithm requires inputting a state and outputting actions, so we need to:

1. Code the game of Snake in a way that we can get the state of the game at each step, and take an action to go to the next step.
2. Create a model/agent that can take the state of the game and output the best action to take, and apply Q-Learning at each step to tune its parameters.

## The Snake game

Watch [this video](https://www.youtube.com/watch?v=L8ypSXwyBds) to understand how to code the game of Snake with PyGame, as this guide will not cover it extensively.

However, an important takeaway is that our action space will be **straight, left, right**. Going behind will always result in death and can be ignored for better training and simplicity.

The observation space in the video is not really applicable to this project, as the subject clearly states that the snake can only see things straight, left, right, and behind, starting from its head.

That means giving the snake the relative position of the apples is not possible.

Don't worry, you will get ideas of how to represent the state of the game later on.

## The model, and how to handle PyTorch

You should now have a functioning game, and you should be able to get the state of the game at each step.

Now we are going to use PyTorch to create a model, similar to the one we used in [multilayer-perceptron](https://github.com/leogaudin/multilayer-perceptron), but this time we don't have to recode the whole framework.

We will use a simple neural network with 3 layers: an input layer, 2 hidden layers, and an output layer.

The model class will look as simple as this:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        n_observations,
        n_actions,
    ):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 42, dtype=torch.float32)
        self.layer2 = nn.Linear(42, 42, dtype=torch.float32)
        self.layer3 = nn.Linear(42, n_actions, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```

The `forward` method is the one that will be called when we input a state to the model.

As you can see, the logic is nothing new: a simple feedforward network, with all the information we want to pass in our state as an input, and the actions we can take as an output.

What will differ from the previous project is how to calculate the loss and update the model.

### The update rule

Let's take our Q-Learning formula from earlier:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

## Resources

- [📺 YouTube − Reinforcement Learning: Crash Course AI #9](https://www.youtube.com/watch?v=nIgIv4IfJ6s)
- [📺 YouTube − Reinforcement Learning from scratch](https://www.youtube.com/watch?v=vXtfdGphr3c)
- [📺 YouTube − Neural Network Learns to Play Snake](https://www.youtube.com/watch?v=zIkBYwdkuTk)
- [📺 YouTube − Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds)
- [📖 PyTorch − Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [📖 HuggingFace − The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [📖 arXiv − Accelerated Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1803.02811): information about batch sizes.
