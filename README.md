<h1 align='center'> 🐍 Learn2Slither</h1>

> ⚠️ This tutorial assumes you have done [multilayer-perceptron](https://github.com/leogaudin/multilayer-perceptron) and [dslr](https://github.com/leogaudin/dslr).

## Table of Contents

- [Introduction](#introduction) 👋

- [Q-Learning](#q-learning) 🧠

- [The Snake game](#the-snake-game) 🐍

- [The model, and how to handle PyTorch](#the-model-and-how-to-handle-pytorch) 🤖
    - [The update rule](#the-update-rule) 🔄
    - [Replay memory](#replay-memory) 💭
    - [PyTorch shenanigans](#pytorch-shenanigans) 🤯

- [Training the model](#training-the-model) 🚀
    - [Hyperparameters](#hyperparameters) 🎛
        - [`gamma`](#gamma)
        - [`epsilon_init`, `epsilon_min`, `epsilon_decay`](#epsilon_init-epsilon_min-epsilon_decay)
        - [`lr`](#lr)
        - [`max_memory`](#max_memory)
        - [`batch_size`](#batch_size)

    - [Rewards](#rewards) 🎁
        - [Tip if your snake starts to go in circles](#tip-if-your-snake-starts-to-go-in-circles) 🔄

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
- $\max_{a'} Q(s', a')$ is the maximum predicted quality of the next action in the next state (basically, asking our model the best quality attainable in the next state, the best we can hope for).

So, we know we have to use Q-Learning, and that this algorithm requires inputting a state and outputting actions, so we need to:

1. Code the game of Snake in a way that we can:
    1. get the state of the game at each step
    2. take an action to go to the next step.

2. Create a model/agent that:
    1. takes the state of the game
    2. outputs the best action to take
    3. applies Q-Learning at each step to tune its parameters.

## The Snake game

Watch [this video](https://www.youtube.com/watch?v=L8ypSXwyBds) to understand how to code the game of Snake with PyGame, as this guide will not cover it extensively.

However, an important takeaway is that our action space will be **straight, left, right**. Going behind will always result in death and can be ignored for better training and simplicity.

The observation space in the video is not really applicable to this project, as the subject clearly states that the snake can only see things straight, left, right, and behind, starting from its head.

That means **giving the snake the exact relative position of the apples is not possible**.

Don't worry, you will get ideas of how to represent the state of the game later on.

However, one crucial thing is what you return every time a step is played. Basically, your `play_step` should take an action and return:

- What **state** the game **was** in.
- What **action** was taken.
- What **reward** was given **for that action**.
- What **state** the game is in **after that action**.
- If the game is **done**.

You should be able to call it as follows:

```python
state, action, reward, next_state, done = game.play_step(action)
```

## The model, and how to handle PyTorch

You should now have a functioning game, and you should be able to get the state of the game at each step.

At the time of writing, the state used in this repository is an array of:

- **How much the snake is moving** (if it is going in circles or exploring).
- The **last move** the snake made (straight, left, right).
- The **danger** right next to the snake (if there is a wall or the snake's body).
- If there is a **green apple** in the snake's vision.
- If there is a **red apple** in the snake's vision.

Now we are going to use PyTorch to create a model, similar to the one we used in [`multilayer-perceptron`](https://github.com/leogaudin/multilayer-perceptron), but this time we don't have to recode the whole framework.

We will use a simple neural network with 4 layers: an input layer, 2 hidden layers, and an output layer.

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

As you may have understood earlier, this gives us the **maximum quality we can hope for given the action we just took**.

During the game, this will allow us to update our model.

If the game is done, there is no next state to consider, so this update rule simply becomes:

$$
Q(s, a) = r
$$

Let's take an example to see how we can implement it:

- Given a state $s$, your model output the following Q-values for respectively "go straight", "go left", "go right": $[0.1, 0.2, 0.3]$.

- "go right" is the maximum value, so you take it, and get a reward of $1$.

- The next state is $s'$, you give it to your model and get the following Q-values: $[0.2, 0.3, 0.4]$. So given the action "go right", you can hope for a maximum quality of $0.4$ in the future.

- The quality of the action "go right" in the state $s$ is then updated to:

$$
Q(s, \text{"go right"}) = 1 + \gamma \times 0.4
$$

Now that's great, but how do you calculate the loss given this information?

Well, you simply assign the quality you just calculated to the Q-value of the action you took for that state:

```python
prediction = [0.1, 0.2, 0.3]
next_state_prediction = [0.2, 0.3, 0.4]
target = prediction.clone()

action = 2
reward = 1
gamma = 0.9

max_future_q = reward + gamma * max(next_state_prediction)

target[action] = reward + max_future_q

loss = MSELoss(target, prediction)
```

> ⚠ The code above is a simplification to illustrate the concept.

### Replay memory

In practice, you will update your model at each step, but a game is not only defined by actions took one step at a time.

Because of that, you will also need to store the transitions you made during the game in a **replay memory**.

Using the example from above, you will append this set of information to a list after each move:

```python
state, action, reward, next_state, done = game.play_step(action)

replay_memory.append((state, action, reward, next_state, done))
```

This memory can be represented as a matrix of shape $(\text{nTransitions}, 5)$, where each row is a transition.

That will allow you to use the same function to train your model, whether it is on one transition or on a batch of transitions.

> 💡 A single step can simply be represented as a $(1, 5)$ matrix with a bit of manipulation.

Every time the game is done, you will sample a batch of transitions from the replay memory, and update your model with it.

```python
batch = random.sample(replay_memory, batch_size)
states, actions, rewards, next_states, dones = zip(*batch)

prediction = model(states)
target = prediction.clone()

for i in range(len(dones)):
    if dones[i]:
        target[i][actions[i]] = rewards[i]
    else:
        max_future_q = rewards[i] + gamma * max(model(next_states[i]))
        target[i][actions[i]] = rewards[i] + max_future_q

loss = MSELoss(target, prediction)
```

> 💡 As you may have noticed, the batches are not transitions in order, but rather random samples. This might sound counterintuitive, but it is actually relevant to decorrelate the actual sequences from their output and avoid overfitting.

### PyTorch shenanigans

If you are coming from `multilayer-perceptron`, you might get **confused** by how PyTorch works, especially **when it comes to backpropagation**.

If we take the code above, the backpropagation would basically be:

```python
# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
# ...
# loss = MSELoss(target, prediction)

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

That is weird, right? We only call some `backward` method on the loss, and then we call `step` on the optimizer.

One could think that we would need to pass the gradient with respect to the loss to the optimizer, and then tell it to perform backpropagation, but PyTorch handles this for us.

Everytime we perform an operation with a tensor, **PyTorch keeps track of the operations and the gradients**, so when we call `backward` on the loss, PyTorch knows how to update the parameters of the model.

> ⚠ That is also why you should be consistent with your crucial operations, for example not switching to NumPy to perform some operations, as PyTorch will not be able to track the gradients.

## Training the model

Now, you have a model, you have a game, and you have a way to update the model.

You can now start to figure out how to train the model.

> ⚠ You should separate the logic of training, and the logic of playing the game. For instance, with `Agent` and `Game` classes.

The play logic will look like this:

```python
state, action, reward, next_state, done = game.play_step(action)

self.memory.append((state, action, reward, next_state, done))
self.train_short_memory(state, action, reward, next_state, done)

if done:
    self.train_long_memory()
    game.reset()
```

### Hyperparameters

You will need to tune some hyperparameters to get the best results:

- `gamma`
- `epsilon_init`, `epsilon_min`, `epsilon_decay`
- `lr`
- `max_memory`
- `batch_size`

The guidelines given here might vary for your implementation, and the **best way to tune them is trial and error**, however, we will try to stay as general as possible.

#### `gamma`

The **discount factor** is a crucial hyperparameter in reinforcement learning, as it will determine how much you care about the future.

A **high discount factor** will make the agent **care more about the future**, while a **low discount factor** will make the agent **care more about the immediate reward**.

#### `epsilon_init`, `epsilon_min`, `epsilon_decay`

`epsilon` is the **exploration rate**, and is pretty straightforward:

Everytime the agent has to take an action, it will choose a random action with a probability of `epsilon`.

1. Generate a random number between 0 and 1.
2. If the number is less than `epsilon`, take a random action.
3. Otherwise, take the action given by the model.

The exploration rate will start at `epsilon_init`, and will decay at each step until it reaches `epsilon_min`.

It is generally **a good idea to start with a very high exploration rate**, like `0.9`.

Furthermore, you might want to keep `epsilon_min` a bit higher than `0` to keep some exploration in the model, even if it means performing worse during training.

For example, your model might stagnate and frequently hit walls because of a random action taken, but during evaluation, `epsilon` will be `0` and the model will perform better.

#### `lr`

The **learning rate** is the rate at which the model will update its parameters. You should already know that.

This one is particularly hard to arbitrate, so you might want to try different values, anywhere between `0.0001` and `0.1`.

#### `max_memory`

The **maximum size of the replay memory** will determine how much the model can learn from the past.

In a game like Snake, where the state is not complex over time and rather instantaneous, you can keep this value low if you want to save memory.

#### `batch_size`

The **size of the batch** used to train the model will determine how much the model will learn from each transition.

Generally, the bigger the batch, the more the model will learn, but the slower the training will be.

If you are short on memory, you might want to keep this value low, but never below 32.

You can check out this [discussion](https://ai.stackexchange.com/questions/23254/is-there-a-logical-method-of-deducing-an-optimal-batch-size-when-training-a-deep).

### Rewards

In this project, the subject gives indications about the rewards you should give to the agent:

> - If the snake eats a red apple: a negative reward.
> - If the snake eats a green apple: a positive reward.
> - If the snake eats nothing: a smaller negative reward.
> - If the snake is Game over (hit a wall, hit itself, null length): a bigger negative reward.

The relative magnitudes of the rewards are important. **If they are too low** for something we want the agent to do, **it will not care about it**.

An example of rewards could be:

- If the snake eats a red apple: `-25`
- If the snake eats a green apple: `25`
- If the snake eats nothing: `-2.5`
- If the snake is Game over: `-100`

However, once again, the best way to tune them is trial and error.

#### Tip if your snake starts to go in circles

A frequent problem with the Snake game is that the snake will start to go in circles, because it constitutes a safe way to minimize the rewards.

This might happen if the reward for eating a green apple is too low, but not only.

One trick for this is first to pass an indication of how much the snake is moving in the state, and then adapt the "eat nothing" reward to this.

> 💡 Simply penalizing it will have little to no effect, the agent also needs to receive this indication as an input to be able to exploit it.

For this, you can use the **standard deviation**.

If the standard deviation of the snake's position is low, it means it is going in circles.

If the standard deviation is high, it means it is exploring.

Let's take our base reward $-2.5$, and make it proportional to the standard deviation.

$$
\text{eatNothingReward} =  \frac{-2.5}{\text{std}^3}
$$

Where $\text{std}$ is the mean of the standard deviation of the $x$ and $y$ positions of the snake.

For a standard deviation of $0.5$, the reward will be $-20$, and for a standard deviation of $2$, the reward will be $-0.3125$.

This will make less attractive for the snake to repeat the same patterns.

## About this repository

The models available in this repository were trained using the following hyperparameters:

| Hyperparameter | final_1000.pth |
| --- | --- |
| `gamma` | 0.95 |
| `epsilon_init` | 0.9 |
| `epsilon_min` | 0.2 |
| `epsilon_decay` | 0.995 |
| `lr` | 0.01 |
| `max_memory` | 1000000 |
| `batch_size` | 1024 |
| `game_width` | 800 |
| `game_height` | 800 |
| `block_size` | 80 |
| `alive_reward` | -2.5 |
| `death_reward` | -100 |
| `green_apple_reward` | 25 |
| `red_apple_reward` | -25 |


## Resources

- [📺 YouTube − Reinforcement Learning: Crash Course AI #9](https://www.youtube.com/watch?v=nIgIv4IfJ6s)
- [📺 YouTube − Reinforcement Learning from scratch](https://www.youtube.com/watch?v=vXtfdGphr3c)
- [📺 YouTube − Neural Network Learns to Play Snake](https://www.youtube.com/watch?v=zIkBYwdkuTk)
- [📺 YouTube − Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds)
- [📖 PyTorch − Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [📖 HuggingFace − The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [📖 arXiv − Accelerated Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1803.02811): information about batch sizes.
