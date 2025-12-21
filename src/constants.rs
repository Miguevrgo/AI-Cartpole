use std::f32::consts::PI;

pub const GRAVITY: f32 = 9.8;
pub const MASS_CART: f32 = 1.0;
pub const MASS_POLE: f32 = 0.1;
pub const POLE_LENGTH: f32 = 0.5;
pub const FORCE_MAGNITUDE: f32 = 10.0;
pub const TAU: f32 = 0.005;
pub const ANGLE_THRESHOLD: f32 = 35.0 * PI / 180.0;
pub const POS_THRESHOLD: f32 = 2.4;

pub const LEARNING_RATE: f32 = 0.001;
pub const GAMMA: f32 = 0.99;
pub const EPSILON_START: f32 = 1.0;
pub const EPSILON_END: f32 = 0.01;
pub const EPSILON_DECAY: f32 = 0.995;
pub const BATCH_SIZE: usize = 64;
pub const REPLAY_BUFFER_SIZE: usize = 10000;
pub const TARGET_UPDATE_FREQ: usize = 100;
