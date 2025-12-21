# AI-Cartpole

[![Build Status](https://img.shields.io/github/actions/workflow/status/Miguevrgo/AI-Cartpole/build.yml?branch=main)](https://github.com/Miguevrgo/AI-Cartpole/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An Artificial Intelligence agent that learns to balance an inverted pendulum using Deep Q-Network (DQN) Reinforcement Learning, **implemented from scratch** in Rust.

---

## Overview

This project provides a complete implementation of the classic CartPole control problem, where an AI agent learns to balance a pole on top of a moving cart by applying forces left or right. The system includes three modes:

- **Manual Mode**: Control the cart with keyboard arrows to understand the physics
- **Training Mode**: Watch the AI learn from scratch using Deep Q-Learning
- **Watching Mode**: See the trained agent perform without learning

## Key Features

### Implemented From Scratch

Unlike most solutions that rely on libraries like TensorFlow or PyTorch, this project implements the entire AI stack from the ground up:

- **Physics Simulation**: Complete inverted pendulum dynamics using the correct equations of motion with friction and damping
- **Neural Network**: Multi-Layer Perceptron (MLP) with:
  - ReLU activation functions
  - He weight initialization
  - Forward propagation
  - Backpropagation with gradient descent
- **Deep Q-Network Algorithm**:
  - Experience replay buffer for stable learning
  - Target network for Q-value stability
  - Epsilon-greedy exploration strategy with decay
  - Temporal difference learning with discount factor

### Architecture Details

**Neural Network**: 4 → 32 → 32 → 2
- Input: 4 state variables (position, velocity, angle, angular velocity)
- Hidden layers: 32 neurons each with ReLU activation
- Output: 2 Q-values (for left and right actions)

**Hyperparameters**:
- Learning rate: 0.001
- Discount factor (γ): 0.99
- Epsilon decay: 0.995 (1.0 → 0.01)
- Replay buffer: 10,000 experiences
- Batch size: 64
- Target network update: every 100 steps

## Installation & Usage

### Prerequisites

- Rust 1.70 or higher
- Cargo (comes with Rust)

### Building

```bash
git clone https://github.com/Miguevrgo/AI-Cartpole.git
cd AI-Cartpole
cargo build --release
```

### Running

```bash
cargo run --release
```

### Controls

- **Arrow Keys**: Control cart in manual mode (Left/Right)
- **M**: Switch to Manual mode
- **T**: Switch to Training mode (AI learns)
- **W**: Switch to Watching mode (AI performs without learning)
- **Space**: Reset current episode
- **R**: Create new untrained agent

## How It Works

### CartPole Physics

The system models an inverted pendulum using second-order differential equations:

- Cart mass: 1.0 kg
- Pole mass: 0.1 kg
- Pole length: 0.5 m
- Force magnitude: ±10 N
- Timestep: 0.02 s

The simulation includes:
- Gravitational acceleration
- Cart friction coefficient
- Pole angular friction
- Non-linear trigonometric dynamics

**Success Criteria**:
- Position: within ±2.4 units
- Angle: within ±12 degrees

### Deep Q-Learning

The agent learns through trial and error:

1. **Observe** the current state (position, velocity, angle, angular velocity)
2. **Choose** an action using epsilon-greedy strategy
3. **Execute** the action and observe reward and next state
4. **Store** the experience in replay buffer
5. **Sample** a random batch from buffer
6. **Compute** target Q-values using target network
7. **Update** main network weights via backpropagation
8. **Periodically sync** target network with main network

The reward structure is simple:
- +1 for each timestep the pole remains balanced
- 0 when the episode terminates (failure)

### Training Progress

Typical learning curve:
- Episodes 1-100: Random exploration, frequent failures
- Episodes 100-300: Learning basic balance, increasing duration
- Episodes 300-500: Achieving consistent balance
- Episodes 500+: Near-optimal performance (200+ steps)

## Project Structure

```
AI-Cartpole/
├── src/
│   ├── main.rs         # Application entry point, UI, game loop
│   ├── cartpole.rs     # Physics simulation and environment
│   ├── network.rs      # Neural network implementation
│   ├── agent.rs        # DQN agent with replay buffer
│   └── constants.rs    # Physics and hyperparameter constants
├── Cargo.toml          # Rust dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## Dependencies

- `macroquad` (0.4.14): Simple 2D game framework for visualization
- `ndarray` (0.16.1): N-dimensional arrays for neural network operations
- `rand` (0.9.2): Random number generation for exploration

## Technical Insights

### Why From Scratch?

Building the entire stack provides deep understanding of:
- How neural networks actually compute and learn
- The mechanics of reinforcement learning algorithms
- Physics simulation and numerical integration
- The interplay between exploration and exploitation

### Challenges Solved

1. **Stability**: Using target networks prevents Q-value divergence
2. **Sample Efficiency**: Experience replay decorrelates training samples
3. **Exploration**: Epsilon-greedy with decay balances learning phases
4. **Physics Accuracy**: Implementing correct equations of motion with friction
5. **Numerical Stability**: Proper weight initialization and learning rates

## Future Enhancements

Potential improvements for learning purposes:
- [ ] Double DQN for reduced Q-value overestimation
- [ ] Prioritized experience replay
- [ ] Dueling DQN architecture
- [ ] Model checkpointing and loading
- [ ] Training metrics visualization
- [ ] Curriculum learning with varying difficulty

## References

This implementation is based on the following research and resources:

- **Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
- **Florian, R. (2005).** "Correct equations for the dynamics of the cart-pole system." Technical Report.
- **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created as an educational project to understand reinforcement learning and systems programming from first principles.

---

**Note**: This is an educational implementation. For production RL applications, consider using established libraries like PyTorch, TensorFlow, or Stable-Baselines3.
 
