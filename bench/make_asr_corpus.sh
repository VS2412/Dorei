#!/usr/bin/env bash
# Generate a SYNTHETIC ASR eval corpus: piper TTS → 16kHz mono s16le WAVs
# (the exact format the recorder produces), plus manifest.tsv for aria-bench.
#
# CAVEAT: synthetic studio-clean TTS speech is an EASY case — it does not
# capture the user's accent, mic, or room. It gives a repeatable floor and
# catches regressions, nothing more. The honest corpus accumulates from real
# use at ~/.local/share/aria/bench_corpus/ (fix its manifest transcripts by
# hand where whisper misheard, then run:
#   aria-bench asr --manifest ~/.local/share/aria/bench_corpus/manifest.tsv
#
# Phrases avoid digits on purpose: whisper may emit "3" for "three", which
# the WER normalizer would count as an error even though ARIA handles both.
set -euo pipefail

OUT="$(cd "$(dirname "$0")" && pwd)/wavs"
PIPER_MODEL="${PIPER_MODEL:-$HOME/.local/share/piper/en_US-lessac-medium.onnx}"

mkdir -p "$OUT"
: > "$OUT/manifest.tsv"

phrases=(
  "open firefox"
  "open the terminal"
  "take a screenshot"
  "turn the volume down"
  "switch to the next workspace"
  "what's in my clipboard"
  "play some music"
  "open spotify on this device"
  "what's my battery level"
  "how much disk space do I have"
  "set a timer for two minutes"
  "search the web for rust tutorials"
  "close this window"
  "remember that my favorite editor is neovim"
  "what windows are open right now"
)

i=0
for p in "${phrases[@]}"; do
  i=$((i+1))
  f=$(printf "synth_%02d" "$i")
  echo "$p" | piper --model "$PIPER_MODEL" --output_file "$OUT/$f.raw.wav" --quiet
  ffmpeg -y -loglevel error -i "$OUT/$f.raw.wav" \
         -ar 16000 -ac 1 -c:a pcm_s16le "$OUT/$f.wav"
  rm -f "$OUT/$f.raw.wav"
  printf '%s\t%s\n' "$f.wav" "$p" >> "$OUT/manifest.tsv"
done

echo "wrote $i wavs + manifest.tsv to $OUT"
