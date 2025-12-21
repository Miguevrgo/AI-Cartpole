use rand::Rng;
use std::collections::VecDeque;

use crate::cartpole::{Action, Cartpole};
use crate::constants::*;
use crate::network::Network;

#[derive(Clone)]
pub(crate) struct Experience {
    state: [f32; 4],
    action: Action,
    reward: f32,
    next_state: [f32; 4],
    done: bool,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(
        &mut self,
        state: [f32; 4],
        action: Action,
        reward: f32,
        next_state: [f32; 4],
        done: bool,
    ) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(Experience {
            state,
            action,
            reward,
            next_state,
            done,
        });
    }

    pub fn sample(&self, batch_size: usize) -> Option<Vec<Experience>> {
        if self.buffer.len() < batch_size {
            return None;
        }

        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }

        Some(samples)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct DQNAgent {
    q_network: Network,
    target_network: Network,
    best_network: Network,
    replay_buffer: ReplayBuffer,
    epsilon: f32,
    steps: usize,
    best_reward: f32,
}

impl DQNAgent {
    pub fn new() -> Self {
        let q_network = Network::new(&[64, 64, 2]);
        let target_network = q_network.clone();
        let best_network = q_network.clone();

        DQNAgent {
            q_network,
            target_network,
            best_network,
            replay_buffer: ReplayBuffer::new(REPLAY_BUFFER_SIZE),
            epsilon: EPSILON_START,
            steps: 0,
            best_reward: 0.0,
        }
    }

    pub fn select_action(&mut self, state: &Cartpole, explore: bool) -> Action {
        let mut rng = rand::rng();

        if explore && rng.random_range(0.0..1.0) < self.epsilon {
            if rng.random_range(0.0..1.0) < 0.5 {
                Action::Left
            } else {
                Action::Right
            }
        } else {
            let state_array = state.to_array();
            let network = if explore {
                &mut self.q_network
            } else {
                &mut self.best_network
            };
            let action_idx = network.predict(&state_array);
            Action::from_index(action_idx)
        }
    }

    pub fn store_experience(
        &mut self,
        state: &Cartpole,
        action: Action,
        reward: f32,
        next_state: &Cartpole,
        done: bool,
    ) {
        self.replay_buffer.push(
            state.to_array(),
            action,
            reward,
            next_state.to_array(),
            done,
        );
    }

    pub fn train(&mut self) -> Option<f32> {
        self.steps += 1;

        if self.steps % TRAIN_FREQ != 0 {
            return None;
        }

        if let Some(batch) = self.replay_buffer.sample(BATCH_SIZE) {
            let mut total_loss = 0.0;

            for experience in batch {
                let q_values = self.q_network.forward(&experience.state);
                let next_q_values = self.target_network.forward(&experience.next_state);

                let max_next_q = next_q_values
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let target_q = if experience.done {
                    experience.reward
                } else {
                    experience.reward + GAMMA * max_next_q
                };

                let mut target = q_values.clone();
                target[experience.action.to_index()] = target_q;

                self.q_network.backward(&target, &q_values);

                let loss = (target_q - q_values[experience.action.to_index()]).powi(2);
                total_loss += loss;
            }

            if self.steps % TARGET_UPDATE_FREQ == 0 {
                self.target_network.copy_weights_from(&self.q_network);
            }

            self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_END);

            Some(total_loss / BATCH_SIZE as f32)
        } else {
            None
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn update_best_network(&mut self, episode_reward: f32) {
        if episode_reward > self.best_reward {
            self.best_reward = episode_reward;
            self.best_network.copy_weights_from(&self.q_network);
        }
    }

    pub fn best_reward(&self) -> f32 {
        self.best_reward
    }
}
