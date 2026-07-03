// Controller: game loop + screen state machine. See DESIGN.md §4, §6, §8.

import { SPEEDS, DEFAULT_SPEED, MAX_CATCHUP_STEPS } from './config.js';
import { GameState, Status } from './state.js';
import { createAgent } from './agent/index.js';
import { InputHandler } from './input.js';
import { UI } from './ui.js';
import { render, resizeCanvas } from './render.js';

const Screen = { MENU: 'menu', PLAYING: 'playing', GAME_OVER: 'game_over' };

const canvas = document.querySelector('#board');
const ctx = canvas.getContext('2d');
const input = new InputHandler();
const ui = new UI(document.body);

let screen = Screen.MENU;
let state = null;
let agent = null;
let accumulator = 0;
let last = 0;
let stepInterval = SPEEDS[DEFAULT_SPEED]; // ms per tick; set from the menu each game

function startGame() {
  state = new GameState(ui.selectedSize);
  agent = createAgent(ui.selectedMode, { input });
  agent.reset?.();
  stepInterval = SPEEDS[ui.selectedSpeed];
  resizeCanvas(canvas, ctx, state.size);
  accumulator = 0;
  screen = Screen.PLAYING;
  ui.showPlaying();
  render(ctx, state);
}

function toMenu() {
  screen = Screen.MENU;
  ui.showMenu();
}

// Enter: start from menu, or return to menu from game over.
input.on('enter', () => {
  if (screen === Screen.MENU) startGame();
  else if (screen === Screen.GAME_OVER) toMenu();
});

// Esc: bail out of a game back to the menu.
input.on('escape', () => {
  if (screen === Screen.PLAYING) toMenu();
});

function frame(now) {
  if (!last) last = now;
  const dt = now - last;
  last = now;

  if (screen === Screen.PLAYING) {
    accumulator += dt;
    // Clamp so a backgrounded tab doesn't teleport the snake on return.
    accumulator = Math.min(accumulator, MAX_CATCHUP_STEPS * stepInterval);

    while (accumulator >= stepInterval && screen === Screen.PLAYING) {
      const dir = agent.nextDirection(state.view()); // human OR computer
      if (dir) state.setDirection(dir);              // engine still validates it
      state.tick();
      accumulator -= stepInterval;

      if (state.status === Status.GAME_OVER) {
        screen = Screen.GAME_OVER;
        ui.showGameOver(state.score);
      }
    }
    ui.setScore(state.score);
    render(ctx, state);
  }

  requestAnimationFrame(frame);
}

// Boot.
ui.showMenu();
requestAnimationFrame(frame);
