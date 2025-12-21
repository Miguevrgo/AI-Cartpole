use macroquad::prelude::*;

mod agent;
mod cartpole;
mod constants;
mod network;

use agent::DQNAgent;
use cartpole::{Action, Cartpole};
use constants::*;

#[derive(PartialEq)]
enum Mode {
    Manual,
    Training,
    Watching,
}

struct GameState {
    env: Cartpole,
    agent: DQNAgent,
    mode: Mode,
    episode: usize,
    step: usize,
    total_reward: f32,
    episode_rewards: Vec<f32>,
    last_loss: Option<f32>,
}

impl GameState {
    fn new() -> Self {
        GameState {
            env: Cartpole::new(),
            agent: DQNAgent::new(),
            mode: Mode::Manual,
            episode: 0,
            step: 0,
            total_reward: 0.0,
            episode_rewards: Vec::new(),
            last_loss: None,
        }
    }

    fn reset_episode(&mut self) {
        if self.total_reward > 0.0 {
            self.episode_rewards.push(self.total_reward);
            self.agent.update_best_network(self.total_reward);
        }
        self.env.reset();
        self.episode += 1;
        self.step = 0;
        self.total_reward = 0.0;
    }
}

fn draw_state(state: &Cartpole) {
    let screen_w = screen_width();
    let screen_h = screen_height();

    let inner_width = POS_THRESHOLD * 2.5;
    let scale = screen_w / inner_width;

    let ground_y = screen_h * 0.75;
    draw_line(
        screen_w * 0.15,
        ground_y,
        screen_w * 0.85,
        ground_y,
        3.0,
        BLACK,
    );

    let track_half = POS_THRESHOLD * scale;
    let center_x = screen_w / 2.0;
    draw_line(
        center_x - track_half,
        ground_y + 5.0,
        center_x - track_half,
        ground_y - 5.0,
        2.0,
        RED,
    );
    draw_line(
        center_x + track_half,
        ground_y + 5.0,
        center_x + track_half,
        ground_y - 5.0,
        2.0,
        RED,
    );

    let cart_x = (state.pos * scale) + center_x;
    let cart_w = 100.0;
    let cart_h = 50.0;
    let cart_y = ground_y - cart_h;

    draw_rectangle(cart_x - cart_w / 2.0, cart_y, cart_w, cart_h, DARKGRAY);
    draw_rectangle_lines(cart_x - cart_w / 2.0, cart_y, cart_w, cart_h, 2.0, BLACK);

    let pole_len_pixels = POLE_LENGTH * 400.0;
    let end_x = cart_x + pole_len_pixels * state.pole_angle.sin();
    let end_y = cart_y - pole_len_pixels * state.pole_angle.cos();

    draw_circle(cart_x, cart_y, 8.0, YELLOW);
    draw_line(cart_x, cart_y, end_x, end_y, 8.0, RED);
    draw_circle(end_x, end_y, 12.0, BLUE);
}

fn draw_ui(game_state: &GameState) {
    let state = &game_state.env;

    draw_text("CartPole DQN", 10.0, 30.0, 40.0, BLACK);

    let mode_text = match game_state.mode {
        Mode::Manual => "MANUAL",
        Mode::Training => "TRAINING",
        Mode::Watching => "WATCHING AI",
    };
    draw_text(
        &format!("Mode: {} (M/T/W)", mode_text),
        10.0,
        70.0,
        25.0,
        DARKBLUE,
    );

    draw_text(
        &format!("Episode: {}", game_state.episode),
        10.0,
        100.0,
        20.0,
        BLACK,
    );
    draw_text(
        &format!("Step: {}", game_state.step),
        10.0,
        125.0,
        20.0,
        BLACK,
    );
    draw_text(
        &format!("Reward: {:.1}", game_state.total_reward),
        10.0,
        150.0,
        20.0,
        BLACK,
    );

    if !game_state.episode_rewards.is_empty() {
        let avg_reward: f32 = game_state
            .episode_rewards
            .iter()
            .rev()
            .take(10)
            .sum::<f32>()
            / game_state.episode_rewards.iter().rev().take(10).count() as f32;
        draw_text(
            &format!("Avg Reward (10): {:.1}", avg_reward),
            10.0,
            175.0,
            20.0,
            DARKGREEN,
        );
        let best = game_state.agent.best_reward();
        if best > 0.0 {
            draw_text(
                &format!("Best Reward: {:.1}", best),
                10.0,
                200.0,
                20.0,
                BLUE,
            );
        }
    }

    draw_text(
        &format!("Position: {:.3}", state.pos),
        10.0,
        235.0,
        18.0,
        GRAY,
    );
    draw_text(
        &format!("Velocity: {:.3}", state.velocity),
        10.0,
        260.0,
        18.0,
        GRAY,
    );
    draw_text(
        &format!("Angle: {:.2}°", state.pole_angle.to_degrees()),
        10.0,
        285.0,
        18.0,
        GRAY,
    );
    draw_text(
        &format!("Ang Vel: {:.3}", state.pole_velocity),
        10.0,
        310.0,
        18.0,
        GRAY,
    );

    if game_state.mode == Mode::Training {
        draw_text(
            &format!("Epsilon: {:.3} (exploration)", game_state.agent.epsilon()),
            10.0,
            345.0,
            18.0,
            PURPLE,
        );
        draw_text(
            &format!(
                "Buffer: {}/{}",
                game_state.agent.buffer_size(),
                REPLAY_BUFFER_SIZE
            ),
            10.0,
            370.0,
            18.0,
            PURPLE,
        );
        if let Some(loss) = game_state.last_loss {
            draw_text(&format!("Loss: {:.4}", loss), 10.0, 395.0, 18.0, PURPLE);
        }
    } else if game_state.mode == Mode::Watching {
        draw_text(
            "Using BEST network (no exploration)",
            10.0,
            345.0,
            18.0,
            DARKGREEN,
        );
        draw_text(
            &format!("Best reward: {:.1}", game_state.agent.best_reward()),
            10.0,
            370.0,
            18.0,
            DARKGREEN,
        );
    }

    let screen_h = screen_height();
    draw_text(
        "Controls: Arrow Keys (Manual) | Space: Reset | M/T/W: Switch Mode | R: New Agent",
        10.0,
        screen_h - 20.0,
        18.0,
        DARKGRAY,
    );
}

#[macroquad::main("CartPole DQN")]
async fn main() {
    let mut game_state = GameState::new();

    loop {
        if is_key_pressed(KeyCode::M) {
            game_state.mode = Mode::Manual;
        } else if is_key_pressed(KeyCode::T) {
            game_state.mode = Mode::Training;
        } else if is_key_pressed(KeyCode::W) {
            game_state.mode = Mode::Watching;
        } else if is_key_pressed(KeyCode::Space) {
            game_state.reset_episode();
        } else if is_key_pressed(KeyCode::R) {
            game_state.agent = DQNAgent::new();
            game_state.episode_rewards.clear();
            game_state.episode = 0;
        }

        let action = match game_state.mode {
            Mode::Manual => {
                if is_key_down(KeyCode::Left) {
                    Action::Left
                } else if is_key_down(KeyCode::Right) {
                    Action::Right
                } else {
                    Action::None
                }
            }
            Mode::Training => game_state.agent.select_action(&game_state.env, true),
            Mode::Watching => game_state.agent.select_action(&game_state.env, false),
        };

        let prev_state = game_state.env;
        let result = game_state.env.step(action);

        game_state.step += 1;
        game_state.total_reward += result.reward;

        if game_state.mode == Mode::Training {
            game_state.agent.store_experience(
                &prev_state,
                action,
                result.reward,
                &result.new_state,
                result.finished,
            );

            if let Some(loss) = game_state.agent.train() {
                game_state.last_loss = Some(loss);
            }
        }

        if result.finished {
            game_state.reset_episode();
        }

        clear_background(LIGHTGRAY);
        draw_state(&result.new_state);
        draw_ui(&game_state);

        next_frame().await;
    }
}
