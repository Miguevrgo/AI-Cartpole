use rand::Rng;

use crate::constants::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Left,
    Right,
    None,
}

impl Action {
    pub fn to_index(&self) -> usize {
        match self {
            Action::Left => 0,
            Action::Right => 1,
            Action::None => 2,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Action::Left,
            1 => Action::Right,
            _ => Action::None,
        }
    }
}

pub struct StepResult {
    pub new_state: Cartpole,
    pub reward: f32,
    pub finished: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Cartpole {
    pub velocity: f32,
    pub pos: f32,
    pub pole_angle: f32,
    pub pole_velocity: f32,
}

impl Default for Cartpole {
    fn default() -> Self {
        Self::new()
    }
}

impl Cartpole {
    pub fn new() -> Self {
        Cartpole {
            velocity: 0.0,
            pos: 0.0,
            pole_angle: 0.0,
            pole_velocity: 0.0,
        }
    }

    pub fn to_array(&self) -> [f32; 4] {
        [
            self.pos / POS_THRESHOLD,
            self.velocity / 2.0,
            self.pole_angle / ANGLE_THRESHOLD,
            self.pole_velocity / 2.0,
        ]
    }

    pub fn reset(&mut self) {
        let mut rng = rand::rng();
        self.velocity = rng.random_range(-0.05..0.05);
        self.pos = 0.0;
        self.pole_angle = rng.random_range(-0.05..0.05);
        self.pole_velocity = rng.random_range(-0.05..0.05);
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        let force = match action {
            Action::Right => FORCE_MAGNITUDE,
            Action::Left => -FORCE_MAGNITUDE,
            Action::None => 0.0,
        };

        let _sgn_x = if self.velocity == 0.0 {
            0.0
        } else {
            self.velocity.signum()
        };

        let sin_theta = self.pole_angle.sin();
        let cos_theta = self.pole_angle.cos();
        let pole_vel_sq = self.pole_velocity.powi(2);

        let temp =
            (force + MASS_POLE * POLE_LENGTH * pole_vel_sq * sin_theta) / (MASS_CART + MASS_POLE);
        let pole_ang_acc_top = GRAVITY * sin_theta - cos_theta * temp;
        let pole_ang_acc_bottom = POLE_LENGTH
            * (4.0 / 3.0 - (MASS_POLE * cos_theta * cos_theta) / (MASS_CART + MASS_POLE));
        let pole_ang_acc = pole_ang_acc_top / pole_ang_acc_bottom;

        let cart_acceleration =
            temp - (MASS_POLE * POLE_LENGTH * pole_ang_acc * cos_theta) / (MASS_CART + MASS_POLE);

        self.velocity += cart_acceleration * TAU;
        self.pos += self.velocity * TAU;
        self.pole_angle += self.pole_velocity * TAU;
        self.pole_velocity += pole_ang_acc * TAU;

        let finished = self.pos.abs() > POS_THRESHOLD || self.pole_angle.abs() > ANGLE_THRESHOLD;

        let reward = if finished {
            -1.0
        } else {
            let angle_bonus = 1.0 - (self.pole_angle.abs() / ANGLE_THRESHOLD);
            let pos_bonus = 1.0 - (self.pos.abs() / POS_THRESHOLD);
            0.1 * (1.0 + angle_bonus + pos_bonus * 0.5)
        };

        StepResult {
            new_state: Cartpole {
                pos: self.pos,
                velocity: self.velocity,
                pole_angle: self.pole_angle,
                pole_velocity: self.pole_velocity,
            },
            reward,
            finished,
        }
    }
}
