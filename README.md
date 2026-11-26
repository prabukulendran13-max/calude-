# Carrom Duel

A carrom board game you play against the computer, in the browser, with no internet needed.

- **Install it as an app:** it's an installable web app (PWA) with its own icon. It opens full screen and works with no connection once installed (see below).
- **Modes:** Disc Pool (race to pot your 9 pucks), Carrom (the Queen must be potted and covered before your last puck), Freestyle (points: white 20, black 10, Queen 50, first to 120).
- **The bot:** Chill Bot (easy), Pro Bot (medium) or Beast Bot (hard). It plans each shot by simulating candidate shots with the same physics the board uses, then adds aiming error to match its level.
- **Look and feel:** a loud, playful style for younger players: electric-blue ground, honey-wood board in a chunky indigo frame, sticker-outlined pieces (so white pucks stand out against the wood), combo callouts, pocket bursts, screen shake, confetti and vibration on Android.
- **Controls:** slide the striker with the bar under the board (or drag it on the board), then pull back from the striker, or from anywhere on the board, and let go to shoot. The power arrow grows and turns red as you pull harder; a straight dotted line shows the direction.
- **Turn timer:** 15 seconds a shot, shown as a ring around your picture (can be switched off in the game menu).
- **Feel:** board proportions, striker weight, glide and the aiming display were matched to a screen recording of Carrom Pool's practice mode, with the board's own look kept original.

## Install on your phone

The app needs to be served from a web address once, so your phone can install it.

1. **Turn on GitHub Pages:** in this repository's settings, under **Pages**, choose **Deploy from a branch**, pick `claude/blissful-cray-r0hfuo` and the `/ (root)` folder, then save. GitHub Pages on a private repository needs a paid GitHub plan; on the free plan, make the repository public first (**Settings → General → Change visibility**).
2. **Open the address on your phone:** after a minute or two it is live at `https://prabukulendran13-max.github.io/calude-/`.
3. **Add it to your home screen:**
   - iPhone: in **Safari**, tap **Share → Add to Home Screen**.
   - Android: in **Chrome**, tap **⋮ → Install app**.

After that it opens from its icon, full screen, with or without a connection. When you're online it fetches updates quietly, and they show the next time you open it.

## Files

- `carrom-duel/game.html`: the game source (the page body, as published on claude.ai).
- `carrom-duel/build.py`: builds the app. Run `python3 carrom-duel/build.py` after editing `game.html`.
- `carrom-duel/index.html`, `carrom-duel/sw.js`: generated app page and offline worker. Don't edit these by hand.
- `carrom-duel/manifest.webmanifest`, `carrom-duel/app/`: app name, icons and the bundled Lilita One and Baloo 2 fonts (SIL Open Font License, see `app/OFL-*.txt`).
- `index.html`, `.nojekyll`: send the site's root address to the game, for GitHub Pages.
