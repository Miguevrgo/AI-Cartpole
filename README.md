# AI-Cartpole

[![Build Status](https://img.shields.io/github/actions/workflow/status/Miguevrgo/AI-Cartpole/build.yml?branch=main)](https://github.com/YOUR_USERNAME/AI-Cartpole/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An Artificial Intelligence agent that learns to balance an inverted pendulum (the classic "CartPole" problem) using Reinforcement Learning (DQN) **implemented from scratch** in [Your Language: Rust/C++/C].

---

## Demo

## About This Project

This project is a complete implementation of the classic "CartPole" control problem. The agent's goal is to learn a policy—moving a cart left or right—to prevent a pole, hinged on top of it, from falling over.

Unlike most solutions that rely on libraries like `TensorFlow` or `PyTorch`, this project implements the **entire AI stack from scratch** as an exercise in systems programming and algorithm comprehension.

### Implemented From Scratch:

* **Physics Environment:** The inverted pendulum simulation (equations of motion) is hand-coded.
* **Neural Network:** A simple Multi-Layer Perceptron (MLP), including `forward pass` and `backpropagation`, is implemented with no external AI dependencies.
* **AI Algorithm:** The **Deep Q-Network (DQN)** learning algorithm, including:
    * The **Replay Buffer** (Experience Replay Memory).
    * The **Target Network** architecture for stable learning.

## Installation & Building

This project is written in **[Rust]**.

## References

The following papers have been very useful in the journey:

- Florian, Răzvan. (2005). Correct equations for the dynamics of the cart-pole system. 
