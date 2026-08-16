class KeyboardTracker {
  constructor({
    pauseThresholdMs = 3000,
    presenceCheckIntervalMs = 250,
    isUserPresent = () => false,
    onIdle = () => {}
  } = {}) {
    this.pauseThresholdMs = pauseThresholdMs;
    this.presenceCheckIntervalMs = presenceCheckIntervalMs;
    this.isUserPresent = isUserPresent;
    this.onIdle = onIdle;
    this._keydown = event => this.handleKeyDown(event);
    this.elements = [];
    this.reset();
  }

  attach(element) {
    const elements = (Array.isArray(element) ? element : [element]).filter(Boolean);
    this.detach();
    this.elements = elements;
    // Capture at document level so Backspace is recorded from every
    // assessment editor: code, standard input, final answer, and chat.
    this.elements.forEach(target => target.addEventListener('keydown', this._keydown, true));
  }

  detach() {
    this.elements.forEach(target => target.removeEventListener('keydown', this._keydown, true));
    this.elements = [];
  }

  reset(questionDisplayedAtMs = Date.now()) {
    clearTimeout(this.idleTimer);
    clearInterval(this.presenceTimer);
    this.idleTimer = null;
    this.presenceTimer = null;
    this.questionDisplayedAtMs = questionDisplayedAtMs;
    this.lastKeypressAtMs = null;
    this.backspaceCount = 0;
    this.firstBackspaceAtMs = null;
    this.lastBackspaceAtMs = null;
    this.completedThinkingPauseMs = 0;
    this.currentThinkingPauseMs = 0;
    this.lastPresenceCheckAtMs = null;
    this.lastSavedPauseSeconds = -1;
  }

  handleKeyDown(event) {
    // Count physical key presses, not browser auto-repeat events while the key is held.
    if (event.repeat) return;
    if (['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Escape'].includes(event.key)) {
      return;
    }

    const now = Date.now();
    this.updateThinkingPause(now);
    this.completedThinkingPauseMs += this.currentThinkingPauseMs;
    this.currentThinkingPauseMs = 0;

    if (event.key === 'Backspace') {
      this.backspaceCount += 1;
      this.firstBackspaceAtMs ??= now;
      this.lastBackspaceAtMs = now;
    }

    this.lastKeypressAtMs = now;
    this.lastPresenceCheckAtMs = now;
    this.lastSavedPauseSeconds = -1;
    this.startPresenceTimer();

    clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.onIdle(this.measurements());
    }, this.pauseThresholdMs);

    // Persist every correction immediately so a Backspace press is not lost
    // if the participant changes focus or leaves before the idle timer fires.
    if (event.key === 'Backspace') this.onIdle(this.measurements());
  }

  startPresenceTimer() {
    if (this.presenceTimer) return;

    this.presenceTimer = setInterval(() => {
      if (this.lastKeypressAtMs === null) return;
      this.updateThinkingPause(Date.now());
      const measurements = this.measurements();
      if (measurements.thinking_pause_seconds > this.lastSavedPauseSeconds) {
        this.lastSavedPauseSeconds = measurements.thinking_pause_seconds;
        this.onIdle(measurements);
      }
    }, this.presenceCheckIntervalMs);
  }

  updateThinkingPause(now) {
    if (this.lastKeypressAtMs === null) return;

    const eligibleFromMs = this.lastKeypressAtMs + this.pauseThresholdMs;
    const checkFromMs = Math.max(this.lastPresenceCheckAtMs ?? now, eligibleFromMs);
    if (now > checkFromMs && this.isUserPresent()) {
      this.currentThinkingPauseMs += now - checkFromMs;
    }
    this.lastPresenceCheckAtMs = now;
  }

  measurements(now = Date.now()) {
    this.updateThinkingPause(now);
    return {
      backspace_count: this.backspaceCount,
      backspace_time_seconds: this.firstBackspaceAtMs === null
        ? 0
        : Math.max(0, Math.round((this.lastBackspaceAtMs - this.firstBackspaceAtMs) / 100) / 10),
      thinking_pause_seconds: Math.floor(
        (this.completedThinkingPauseMs + this.currentThinkingPauseMs) / 1000
      )
    };
  }

  finalize() {
    clearTimeout(this.idleTimer);
    clearInterval(this.presenceTimer);
    this.idleTimer = null;
    this.presenceTimer = null;
    return this.measurements();
  }
}

window.KeyboardTracker = KeyboardTracker;
