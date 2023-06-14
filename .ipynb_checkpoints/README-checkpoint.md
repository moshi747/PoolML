# Background on pool

In pool, a general goal is to use the cue ball (white ball) to pot an object ball (numbered ball) into a pocket. In many games, a critical part is to not only pot the object ball but to also leave the cue ball in a good position for the next shot (see https://en.wikipedia.org/wiki/Nine-ball). This is called *playing position* and can be achieved by varying the three scalars below:

speed, horizontal strike position (x-strike), vertical strike position (y-strike)

# Goal of this project

The goal is to use machine learning to approximate the following function

$$f\colon\text{position of the balls (cue ball + object balls)}\rightarrow \text{speed, x-strike, y-strike}$$

which outputs the best way to hit the cue ball to play position given the current layout of the balls. As most experienced pool players know, pool is a game of milimeters meaning the slightest of change is the layout of the balls can have a big impact on how you would play. Hence the ideal function $f$ would be very complex, expecially if there are many object balls.

We will rely on an open source pool simulator (https://ekiefl.github.io/projects/pooltool/) developed by Evan Kiefl.

# Simplest case

For now, we will consider the simplest case of when there are two object balls. In order for the AI to learn how effective a particular shot was, we need a way to value the result (the location of the cue ball and the remaining object ball after potting). While this can probably be done by some geometric calculation, we will use a DNN to achieve this.

### Creating a DNN for valuating layouts

We use supervised learning. Our data will consist of the position of the cue ball and object ball together with an integer $0\leq n\leq 100$ which measures the difficulty of the shot ($0$ being most difficult and $100$ being easiest).

So there are $4$ feature cue-x, cue-y, one-x, one-y for the $xy$-coordinates of the balls and one output $n$. To construct the data, we generate the positions of the balls randomly and simulate $100$ shots where we slightly adjust the aiming point every time. The number of times we pot the object ball will be our $n$.


```python
from data import generate_data

one_ball_data = generate_data(10000, tests=100)
```

Once enough data has been generated, we train the model.


```python
from regression import train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_ball_data.iloc[:,:-1], one_ball_data.iloc[:,-1:])

reg_net, train_losses, test_losses = train(X_train, X_test, y_train, y_test,
                                           hidden_sizes = [128, 128, 128], epochs = 10000, lr = 0.001)
```

Let's visualize the CV scores to see how the model is performing.


```python
# Add plot here
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (8,6))

plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')

plt.plot(np.arange(len(test_losses)), test_losses, label='test loss')

ax = fig.axes[0]

plt.legend()

ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')

plt.show()
```

Now that we have a baseline model for valuating the resulting position, let's move on to the next step of our original goal.

### Two methods

We present two different methods for constructing a model

* Supervised learning

* Reinforcement learning

In both cases, the goal is to learn the following function

$$Q\colon \text{x_cue, y_cue, x_one, y_one, x_two, y_two, speed, x-spin, y-spin} \rightarrow \{0,\ldots,100\}$$

which is defined as valuating the resulting position from shooting the cue ball in the given way potting the one ball. In reinforcement learning, the algorithm will also output a policy function which takes input the positions of the balls and outputs the hit on the cue ball that results in the highest output of the $Q$-function.

Let us now go into the details for each one

### Supervised learning model

In this case, our goal again is to construct a bunch of data by randomly generating points in the domain of $Q$ and logging its output. This is done using the following code.


```python
from data import generate_position_data

two_ball_data = generate_position_data(2000, reg_net)
```

We then use the same code as before to train a DNN on this data. The performance is as follows


```python
# Train and evaluate performance here

X_train, X_test, y_train, y_test = train_test_split(two_ball_data.iloc[:,:-1], two_ball_data.iloc[:,-1:])

reg_net_2, train_losses, test_losses = train(X_train, X_test, y_train, y_test,
                                             hidden_sizes = [128, 128, 128, 128], epochs = 10000, lr = 0.001)

fig = plt.figure(figsize = (8,6))

plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')

plt.plot(np.arange(len(test_losses)), test_losses, label='test loss')

ax = fig.axes[0]

plt.legend()

ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')

plt.show()
```

The MSE is not as satisfying as the case with a single object ball. The function we're trying to learn here is much more intricate. We also tried XGBoost, but the results were not as goot as the neural network.

### Reinforcement learning model

Here we created a GYM API for this task. The natural choices for the states are the position of the balls and the transition actions are given by the hit on the cue ball. The interesting part is setting up the reward function for the actions. We settled on the following

1. If object ball is not potted:

    a. If no object ball moves: $-10$
    
    b. Else: $-0.1$
    
2. Elif cue ball scratches (goes into a pocket): $-4$

3. Elif both object balls are potted: $0$

4. Else (target object ball potted and two balls remain on the table): $4(e^{n/100}-1)$ where $n$ is the valuation of the resulting position.

The reason we chose to include an exponential function in the last case is to emphasize the importance of getting as close to the largest $n$ as possible by offering a reward that is weighted exponentially rather than linealry on $n$. This seem to work well in practice as our experiments showed. The exact algorithm we employ is the soft actor-critic method given here https://spinningup.openai.com/en/latest/algorithms/sac.html


```python
from spinup.algos.pytorch.sac.sac import sac
from env import Pool_random

sac(Pool_random, epochs = 10, alpha = 0.1)
```

## Next steps

Now that we have the case of two balls finished, we can move on to 3, 4, ..., and eventually 9 balls for the game of 9-ball. The inductive step is as follows: Once we have a model for $n$ many balls, we can use our model to construct an auxilary model which learns the *value function* attached to any layout of $n$ balls. This value function can be thought of as the maximum possible reward attainable by following the optimal shooting choices (which we have a model for). Then we may construct a model for $n+1$ balls by either supervised or reinforcement learning as before trying to maximize the value function of the remaining layout after potting.

Another approach would be to simply apply a reinforcement or supervised learning algorithm to the case of 9 balls. In practice, we found the complexity coming from the number of balls makes it very hard to optimize the model. Hence, we are taking an iterative approach as shown here.

# Appendix

While a random layout of 9 balls seemed too much for reinforcement learning to optimize, having a fixed starting layout is much easier. Here we demonstrate using the GYM environemnt Pool_static which takes as input a fixed starting layout of balls and always starts here whenever reset is called.


```python
from env import Pool_static

sac(Pool_random, epochs = 10, alpha = 0.1)
```

Here is a video demonstrating the shots learned for a particular layout of 9-ball


```python

```
