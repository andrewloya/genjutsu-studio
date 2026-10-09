# LYRC Genjutsu Studio

Swap the person in any clip for any character. The scene, the camera, the dance moves and the lip sync stay; the person changes.
Runs on your computer, renders on your own [Evolink](https://evolink.ai) account.

## What you need
- A Mac with Apple Silicon (M1 or newer) works best. Linux or Windows (through WSL) with an NVIDIA GPU works too. Plain CPU works, just slowly.
- [Homebrew](https://brew.sh) on Mac.
- An Evolink account with credits, and an API key from evolink.ai → API Keys.
- About 5 GB of free space.

## Set up (once)
1. Download it: the zip on the [Releases page](https://github.com/andrewloya/genjutsu-studio/releases/latest) (or the green **Code** button → **Download ZIP**). Unzip it somewhere you'll keep it.
   Or with git: `git clone https://github.com/andrewloya/genjutsu-studio`
2. Mac: double-click **setup.command**. If macOS says it can't be opened: System Settings → Privacy & Security → scroll down → **Open Anyway**. (Or open Terminal, drag the folder in after typing `cd `, press Enter, then run `./setup.command`.)
   Linux/WSL: `./setup.sh`.
3. Wait 5–10 minutes while it installs.

## Use it
1. Double-click **start.command** (Linux/WSL: `./start.sh`). The studio opens at http://127.0.0.1:8790.
2. First time only: paste your Evolink key. It's saved in this folder's `.env` and only ever sent to Evolink.
3. **Your clip:** drop a video (4–15 s of it gets used; slide Start and Length).
4. **Character:** upload a picture, or "Make one" from a description (~$0.05).
5. **Whole body** swaps the whole person. **Head only** keeps their outfit and swaps the head (good for costumes).
6. **Lip sync:** CLEAN (default) keeps the character's face exactly like the picture. MAX syncs lips tighter but can tint the face.
7. Press **Make it**. The prep runs free on your computer (1–5 min), then one render goes to Evolink (~5 min).
8. Like it? Press **1080p** on the result to upscale it.

## What it costs (your Evolink credits)
| | |
|---|---|
| Draft (480p) | ~$0.17 per second of clip (8 s ≈ $1.34) |
| 1080p upscale | ~$0.93 per second (8 s ≈ $7.10) |
| Make a character | ~$0.05 |
If Evolink blocks a render (content filter), you're not charged. The studio only runs one render at a time, and shows today's spend at the top.

## Tips
- The face needs to be lit and visible for good lip sync.
- One main person per shot works best. Shots with other people in them can be skipped.
- Baggy clothes are handled: the character gets matching long sleeves instead of giant arms ("Match my outfit").
- Famous movie characters and trademarked costumes get blocked by Evolink. Normal people in normal scenes go through.

## Play fair
Only use footage you're allowed to use, and get a person's OK before putting their face in a video.

## Where things go
Your clips, characters and results stay in this folder (`sources/`, `characters/`, `jobs/`). Clips are uploaded to Evolink's file service to render (deleted after 72 h).

## Credits
Built by loya / [LYRC](https://lyrc.studio). Based on the open-source Genjutsu workflow by Sirio Berati (MIT). See LICENSE and THIRD-PARTY.md.
