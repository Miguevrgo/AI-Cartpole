use macroquad::prelude::*;

mod cartpole;
mod constants;

use cartpole::{Action, Cartpole};
use constants::*;

fn draw_state(state: &Cartpole) {
    let screen_w = screen_width();
    let screen_h = screen_height();

    let world_width = THRESHOLD_POS * 50.0;
    let scale = screen_w / world_width;

    let ground_y = screen_h * 0.75;

    draw_line(0.0, ground_y, screen_w, ground_y, 2.0, BLACK);

    let cart_x = (state.pos * scale) + (screen_w / 2.0);
    let cart_w = 100.0;
    let cart_h = 50.0;
    let cart_y = ground_y - cart_h;

    draw_rectangle(cart_x - cart_w / 2.0, cart_y, cart_w, cart_h, DARKGRAY);

    let pole_len_pixels = (POLE_LENGTH * 750.0) * scale;
    let pole_angle = state.pole_angle;

    let pivot_x = cart_x;
    let pivot_y = cart_y;

    let end_x = pivot_x + pole_len_pixels * pole_angle.sin();
    let end_y = pivot_y - pole_len_pixels * pole_angle.cos();

    draw_line(pivot_x, pivot_y, end_x, end_y, 15.0, RED);
}

#[macroquad::main("Cartpole")]
async fn main() {
    let mut env = Cartpole::new();
    env.reset();

    loop {
        let mut action = Action::None;
        if is_key_down(KeyCode::L) {
            action = Action::Left;
        } else if is_key_down(KeyCode::H) {
            action = Action::Right
        }

        let result = env.step(action);

        clear_background(LIGHTGRAY);
        draw_state(&result.new_state);

        draw_text(
            &format!("Cart Pos: {:.6}", result.new_state.pos),
            10.0,
            30.0,
            30.0,
            BLACK,
        );
        draw_text(
            &format!("Cart Velocity: {:.6}", result.new_state.velocity),
            10.0,
            60.0,
            30.0,
            BLACK,
        );
        draw_text(
            &format!(
                "Pole Angle: {:.6}",
                result.new_state.pole_angle.to_degrees()
            ),
            10.0,
            90.0,
            30.0,
            BLACK,
        );
        draw_text(
            &format!("Pole Velocity: {:.6}", result.new_state.pole_velocity),
            10.0,
            120.0,
            30.0,
            BLACK,
        );

        next_frame().await;
    }
}
