// Turns the microphone into what voice questions are made of (the Listener in app.js, and Whisper
// via app/transcribe.py): 16 kHz, 16-bit mono PCM, posted to the page in 100 ms chunks. Downsampled by averaging each run of input
// samples (the browser records at 44.1 or 48 kHz), which is enough low-pass for speech.
const RATE = 16000;
const CHUNK = RATE / 10;

class Pcm16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / RATE;  // input samples per output sample
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.out = new Int16Array(CHUNK);
    this.len = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i];
      this.count++;
      if (++this.pos < this.step) continue;
      this.pos -= this.step;
      const v = Math.max(-1, Math.min(1, this.sum / this.count));
      this.sum = this.count = 0;
      this.out[this.len++] = v * 0x7fff;
      if (this.len === CHUNK) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(CHUNK);
        this.len = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm16k", Pcm16k);
