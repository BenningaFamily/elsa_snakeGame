// Central configuration constants. See DESIGN.md §11.

export const BOARD_SIZES = [8, 10, 12];      // selectable on the menu
export const APPLE_MIN = 5;                  // apple count rolled once per round
export const APPLE_MAX = 7;
export const START_LENGTH = 3;               // initial snake length
export const CELL_PX = 40;                   // pixel size of a grid cell
export const MAX_CATCHUP_STEPS = 5;          // clamp on accumulator after tab throttling

// Movement speed: milliseconds per logic tick (lower = faster). Selectable on
// the menu. Smaller interval → the snake advances more cells per second.
export const SPEEDS = { slow: 200, medium: 130, fast: 80 };
export const DEFAULT_SPEED = 'medium';       // pre-selected speed on the menu

export const CONTROL_MODES = ['human', 'random', 'ai']; // maps to createAgent()
export const DEFAULT_CONTROL = 'human';      // pre-selected control mode
export const AI_ENABLED = true;              // AI control live (aiAgent.js); set false to hide the button

// Palette (canvas colors). HUD/overlay colors live in styles.css.
// Light two-tone checkerboard board with a rich green snake and glossy apples.
export const COLORS = {
  boardLight: '#eef6e0',   // lighter checkerboard cell
  boardDark: '#e4efd0',    // slightly darker checkerboard cell
  snakeBody: '#33a35a',    // rich green body
  snakeHead: '#2b8f4e',    // slightly deeper head
  eye: '#ffffff',
  pupil: '#1c2b22',
  apple: '#e14b3b',
  appleShine: '#ff8f7d',   // highlight glint
  leaf: '#5aa63c',         // apple stem/leaf
};
