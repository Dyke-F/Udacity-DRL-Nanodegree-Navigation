## Report Summary

In this notebook I implemented a single DRL-agent using DDQN (that is shown to reduce overestimation of expected returns as seen in classical vanilla DQN algorithms). Furthermore I used Experience Replay in minibatches of size 64 to prevent oszillation of learning while exploring the environment. 


# Ideas for future projects
Further improvements might include using a [Rainbow Agent](https://arxiv.org/pdf/1710.02298.pdf) to improve performance by using Priority Experience Replay, Duelling Networks, n-step bootstrapping (A3C), distributional learning and some more techniques. Moreover, hyperparameters in deep learning are currently mostly chosen based on try-and-error methods, and grid-search to estimate the best values for the learning rate, epsilon, discounts and also the number of layers and nodes within each layer (only to name a few) would even improve the agents performance.  
