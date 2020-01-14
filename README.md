# Udacity's Deep Reinforcement Learning Nanodegree: Collaboration and competition project

This repository contains my solution to the third project in [Udacity's DRL nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## The task

<img src=media/trained_agent.gif width=60%>

Meet Alice and Bob! Two agents that were trained with [Multi-Agent Deep Deterministic Policy Gradients (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf) to play table tennis. The environment is a modified version of [Unity ML-Agents' Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

A reward of `+0.1` is provided to each agent (independently) each time that they can hit the ball and send it over the net. A reward of `-0.1` is provided if the ball hits their ground or is sent out of grounds. The agents will learn cooperatively to choose the approppriate actions at each time step which will lead to the maximum cumulative reward.

### State and action spaces, goal

- **The state space** is `24` dimensional and contains positions and velocity of the rakets and ball.

- **The action space** is `2` dimensional, movement (toward and from net) and jumping.

- **Solution criteria**: the environment is considered as solved when the agent gets an average score of **0.5 over 100 consecutive episodes**.

## Set-up

To run this code in your own machine, please follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) and [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started).

Note: To develop in my machine, I used an updated version of Pytorch (`1.3.1`). You can reproduce the conda environment exactly following the instructions in `conda-requirements.txt`

## How to run

- **`Report.ipynb`** contains a detailed description of the implementation and allows you to visualize the performance of a trained agent.
- Running **`Tennis.ipynb`** trains the agents from scratch
- The parameters needed to clone the trained agents can be found in `models/`. Refer to the report for more details.
- The agents are defined in `ddpg_agent.py`
- The actual actor-critic networks are defined in `ddpg_model.py`