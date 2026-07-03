// Agent interface + factory. See DESIGN.md §5.1.
//
// Every controller — human or computer — implements:
//   nextDirection(view) => Direction | null   // called once per tick, must not mutate view
//   reset?()                                    // optional, called when a new game starts
//
// `view` is the read-only GameView snapshot from GameState.view():
//   { size, snake (head first), snakeCells, apples, direction }
//
// The engine still validates whatever an agent returns (reverse guard +
// collision rules), so an agent can lose the game but never corrupt its state.

import { AI_ENABLED } from '../config.js';
import { HumanAgent } from './human.js';
import { RandomAgent } from './randomAgent.js';
import { AiAgent } from './aiAgent.js';

// deps: { input } for the human agent; rng optional for determinism/tests.
export function createAgent(mode, deps = {}) {
  switch (mode) {
    case 'human':
      return new HumanAgent(deps.input);
    case 'random':
      return new RandomAgent(deps.rng);
    case 'ai':
      if (!AI_ENABLED) {
        throw new Error('AI control is not available yet (AI_ENABLED is false).');
      }
      return new AiAgent(deps);
    default:
      throw new Error(`Unknown control mode: ${mode}`);
  }
}
