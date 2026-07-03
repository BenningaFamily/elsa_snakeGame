// Renderer: draws a GameState onto the canvas 2D context. See DESIGN.md §9.
//
// Style: a soft two-tone checkerboard board (no hard grid lines), a smooth
// continuous rounded snake with a small face, and glossy rounded apples.

import { CELL_PX, COLORS } from './config.js';
import { parse } from './state.js';

// Size the canvas for a board of `size` cells, accounting for HiDPI so edges
// stay crisp on retina displays.
export function resizeCanvas(canvas, ctx, size) {
  const px = size * CELL_PX;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = px * dpr;
  canvas.height = px * dpr;
  canvas.style.width = `${px}px`;
  canvas.style.height = `${px}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Pixel center of a cell.
const cx = (x) => x * CELL_PX + CELL_PX / 2;
const cy = (y) => y * CELL_PX + CELL_PX / 2;

export function render(ctx, state) {
  drawBoard(ctx, state.size);
  for (const k of state.apples) {
    const { x, y } = parse(k);
    drawApple(ctx, x, y);
  }
  drawSnake(ctx, state.snake, state.direction);
}

function drawBoard(ctx, size) {
  const px = size * CELL_PX;
  ctx.fillStyle = COLORS.boardLight;
  ctx.fillRect(0, 0, px, px);
  // Subtle checkerboard for a clean, grid-line-free look.
  ctx.fillStyle = COLORS.boardDark;
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      if ((x + y) % 2 === 1) ctx.fillRect(x * CELL_PX, y * CELL_PX, CELL_PX, CELL_PX);
    }
  }
}

function drawSnake(ctx, snake, direction) {
  const width = CELL_PX * 0.72;

  // Continuous rounded body: a single thick, round-jointed stroke through the
  // segment centers. Rounded caps/joins give the smooth, pill-like look.
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = COLORS.snakeBody;
  ctx.lineWidth = width;

  if (snake.length === 1) {
    // Degenerate one-cell snake: draw a dot so there's something to stroke.
    const h = snake[0];
    ctx.beginPath();
    ctx.arc(cx(h.x), cy(h.y), width / 2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.snakeBody;
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(cx(snake[0].x), cy(snake[0].y));
    for (let i = 1; i < snake.length; i++) ctx.lineTo(cx(snake[i].x), cy(snake[i].y));
    ctx.stroke();
  }

  drawHead(ctx, snake[0], direction, width);
}

function drawHead(ctx, head, dir, width) {
  const hx = cx(head.x);
  const hy = cy(head.y);
  const r = width / 2 + 1.5; // slightly larger than the body

  // Rounded head in a deeper shade.
  ctx.beginPath();
  ctx.arc(hx, hy, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.snakeHead;
  ctx.fill();

  // Eyes: offset forward along the heading and out to each side.
  const perp = { x: -dir.y, y: dir.x };
  const fwd = CELL_PX * 0.1;
  const side = CELL_PX * 0.17;
  const eyeR = CELL_PX * 0.1;
  const pupilR = CELL_PX * 0.05;
  for (const s of [-1, 1]) {
    const ex = hx + dir.x * fwd + perp.x * side * s;
    const ey = hy + dir.y * fwd + perp.y * side * s;
    ctx.beginPath();
    ctx.arc(ex, ey, eyeR, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.eye;
    ctx.fill();
    // Pupil looks slightly ahead in the direction of travel.
    ctx.beginPath();
    ctx.arc(ex + dir.x * eyeR * 0.4, ey + dir.y * eyeR * 0.4, pupilR, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.pupil;
    ctx.fill();
  }
}

function drawApple(ctx, x, y) {
  const ax = cx(x);
  const ay = cy(y);
  const r = CELL_PX * 0.32;

  // Body.
  ctx.beginPath();
  ctx.arc(ax, ay, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.apple;
  ctx.fill();

  // Glossy highlight.
  ctx.beginPath();
  ctx.arc(ax - r * 0.32, ay - r * 0.35, r * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.appleShine;
  ctx.fill();

  // Little leaf/stem on top.
  ctx.beginPath();
  ctx.ellipse(ax + r * 0.35, ay - r * 0.95, r * 0.28, r * 0.14, -Math.PI / 4, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.leaf;
  ctx.fill();
}
