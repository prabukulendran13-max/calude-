# Carrom Duel

A carrom board game you play against the computer, in the browser, with no internet needed.

- **Play offline:** save `carrom-duel/index.html` to your phone or computer and open it in any browser. It's one self-contained file with no downloads (the rounded font is the only thing fetched online, and it falls back to a system font).
- **Modes:** Disc Pool (race to pot your 9 pucks), Carrom (the Queen must be potted and covered before your last puck), Freestyle (points: white 20, black 10, Queen 50, first to 120).
- **Computer:** Easy, Medium or Hard. The computer plans each shot by simulating candidate shots with the same physics the board uses, then adds aiming error to match its level.
- **Controls:** slide the striker with the bar under the board (or drag it on the board), then pull back from the striker, or from anywhere on the board, and let go to shoot.

## Files

- `carrom-duel/game.html`: the game source (page body, as published).
- `carrom-duel/index.html`: standalone build of the same game. Regenerate it after editing `game.html` with `carrom-duel/build.sh`.
