// UI: menu & game-over overlays. See DESIGN.md §8.
//
// The menu holds two selections — board size and control mode — as button
// groups. Overlays are shown/hidden by toggling a CSS class rather than tearing
// down the canvas.

import { BOARD_SIZES, CONTROL_MODES, DEFAULT_CONTROL, AI_ENABLED, SPEEDS, DEFAULT_SPEED } from './config.js';

const CONTROL_LABELS = { human: 'You', random: 'Random', ai: 'AI' };
const SPEED_LABELS = { slow: 'Slow', medium: 'Medium', fast: 'Fast' };

export class UI {
  constructor(root) {
    this.menu = root.querySelector('#menu');
    this.gameOver = root.querySelector('#game-over');
    this.hud = root.querySelector('#hud');
    this.scoreEl = root.querySelector('#score');
    this.modeEl = root.querySelector('#hud-mode');
    this.finalScoreEl = root.querySelector('#final-score');

    this.selectedSize = BOARD_SIZES[0];
    this.selectedSpeed = DEFAULT_SPEED;
    this.selectedMode = DEFAULT_CONTROL;

    this.buildSizeButtons(root.querySelector('#size-options'));
    this.buildSpeedButtons(root.querySelector('#speed-options'));
    this.buildControlButtons(root.querySelector('#control-options'));
  }

  buildSizeButtons(container) {
    this.sizeButtons = BOARD_SIZES.map((size) => {
      const btn = document.createElement('button');
      btn.className = 'option';
      btn.textContent = `${size}×${size}`;
      btn.addEventListener('click', () => {
        this.selectedSize = size;
        this.refresh();
      });
      container.appendChild(btn);
      return { size, btn };
    });
  }

  buildSpeedButtons(container) {
    this.speedButtons = Object.keys(SPEEDS).map((speed) => {
      const btn = document.createElement('button');
      btn.className = 'option';
      btn.textContent = SPEED_LABELS[speed];
      btn.addEventListener('click', () => {
        this.selectedSpeed = speed;
        this.refresh();
      });
      container.appendChild(btn);
      return { speed, btn };
    });
  }

  buildControlButtons(container) {
    this.controlButtons = CONTROL_MODES.map((mode) => {
      const btn = document.createElement('button');
      btn.className = 'option';
      btn.textContent = CONTROL_LABELS[mode];
      const disabled = mode === 'ai' && !AI_ENABLED;
      if (disabled) {
        btn.disabled = true;
        btn.title = 'Coming soon — a trained AI will control the snake here.';
      } else {
        btn.addEventListener('click', () => {
          this.selectedMode = mode;
          this.refresh();
        });
      }
      container.appendChild(btn);
      return { mode, btn };
    });
  }

  // Highlight the active selections.
  refresh() {
    for (const { size, btn } of this.sizeButtons) {
      btn.classList.toggle('selected', size === this.selectedSize);
    }
    for (const { speed, btn } of this.speedButtons) {
      btn.classList.toggle('selected', speed === this.selectedSpeed);
    }
    for (const { mode, btn } of this.controlButtons) {
      btn.classList.toggle('selected', mode === this.selectedMode);
    }
  }

  showMenu() {
    this.refresh();
    this.menu.classList.remove('hidden');
    this.gameOver.classList.add('hidden');
    this.hud.classList.add('hidden');
  }

  showPlaying() {
    this.menu.classList.add('hidden');
    this.gameOver.classList.add('hidden');
    this.hud.classList.remove('hidden');
    this.modeEl.textContent = CONTROL_LABELS[this.selectedMode];
    this.setScore(0);
  }

  showGameOver(score) {
    this.finalScoreEl.textContent = score;
    this.gameOver.classList.remove('hidden');
  }

  setScore(score) {
    this.scoreEl.textContent = score;
  }
}
