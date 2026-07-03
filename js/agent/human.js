// HumanAgent: returns the direction buffered from the keyboard. See DESIGN.md §5.1.
// Routing human control through the Agent interface keeps the loop single-path.

export class HumanAgent {
  constructor(input) {
    this.input = input; // InputHandler exposing consumeDirection()
  }

  // Called once per tick before state.tick(). Returns the pressed direction
  // since the last tick, or null to continue straight.
  nextDirection() {
    return this.input.consumeDirection();
  }

  reset() {
    this.input.consumeDirection(); // drop any stale buffered key
  }
}
