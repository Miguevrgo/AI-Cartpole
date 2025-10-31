use rand::Rng;

use crate::constants::*;

#[derive(Clone, Copy)]
pub enum Action {
    Left,
    Right,
    None,
}

pub struct StepResult {
    pub new_state: Cartpole,
    pub finished: bool,
}

pub struct Cartpole {
    pub velocity: f32,
    pub pos: f32,
    pub pole_angle: f32,
    pub pole_velocity: f32,
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

        let sgn_x = if self.velocity == 0.0 {
            0.0
        } else {
            self.velocity.signum()
        };

        let sin_theta = self.pole_angle.sin();
        let cos_theta = self.pole_angle.cos();
        let pole_vel_sq = self.pole_velocity.powi(2);

        // Pole angular accelearation as a coefficient of top / bottom
        let pole_ang_acc_top = GRAVITY * sin_theta
            + cos_theta
                * (((-force
                    - MASS_POLE
                        * POLE_LENGTH
                        * pole_vel_sq
                        * (sin_theta + MU_C * sgn_x * cos_theta))
                    / (MASS_CART + MASS_POLE))
                    + MU_C * GRAVITY * sgn_x)
            - (MU_P * self.pole_velocity) / (MASS_POLE * POLE_LENGTH);

        let pole_ang_acc_bottom = POLE_LENGTH
            * ((4.0 / 3.0)
                - ((MASS_POLE * cos_theta) / (MASS_CART + MASS_POLE) * (cos_theta - MU_C * sgn_x)));

        let pole_ang_acc = pole_ang_acc_top / pole_ang_acc_bottom;

        let norm_force = (MASS_CART + MASS_POLE) * GRAVITY
            - MASS_POLE * POLE_LENGTH * (pole_ang_acc * sin_theta + pole_vel_sq * cos_theta);

        // Cart angular accelearation as a coefficient of top / bottom
        let cart_ang_acc_top = force
            + MASS_POLE * POLE_LENGTH * (pole_vel_sq * sin_theta - pole_ang_acc * cos_theta)
            - MU_C * norm_force * sgn_x;

        let cart_ang_acc_bottom = MASS_CART + MASS_POLE;

        let cart_acceleration = cart_ang_acc_top / cart_ang_acc_bottom;

        self.velocity += cart_acceleration * TAU;
        self.pos += self.velocity * TAU;
        self.pole_angle += self.pole_velocity * TAU;
        self.pole_velocity += pole_ang_acc * TAU;

        let finished = self.pos.abs() > POS_THRESHOLD || self.pole_angle.abs() > ANGLE_THRESHOLD;

        StepResult {
            new_state: Cartpole {
                pos: self.pos,
                velocity: self.velocity,
                pole_angle: self.pole_angle,
                pole_velocity: self.pole_velocity,
            },
            finished,
        }
    }
}
