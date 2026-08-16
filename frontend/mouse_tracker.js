class MouseTracker {
  constructor({ actionGapMs = 750, flushIntervalMs = 5000, onSave = async () => {} } = {}) {
    this.actionGapMs = actionGapMs;
    this.flushIntervalMs = flushIntervalMs;
    this.onSave = onSave;
    this._move = event => this.recordCursor(event);
    this._wheel = event => this.recordScroll(event);
    this.reset();
  }

  reset() {
    this.stopListeners();
    clearTimeout(this.scrollActionTimer);
    clearInterval(this.flushTimer);
    this.scrollActionTimer = null;
    this.flushTimer = null;
    this.active = false;
    this.startedAtMs = null;
    this.lastActiveAtMs = null;
    this.scrollUpCount = 0;
    this.scrollDownCount = 0;
    this.scrollTimestepCount = 0;
    this.scrollTimestamps = [];
    this.mouseMoveCount = 0;
    this.cursorDistancePx = 0;
    this.lastCursorX = null;
    this.lastCursorY = null;
    this.pendingScrollDirections = [];
    this.lastScrollAtMs = null;
  }

  start(startedAtMs = Date.now()) {
    this.reset();
    this.active = true;
    this.startedAtMs = startedAtMs;
    this.lastActiveAtMs = startedAtMs;
    document.addEventListener('pointermove', this._move, { passive: true });
    document.addEventListener('wheel', this._wheel, { passive: true });
    this.flushTimer = setInterval(() => {
      this.flush('periodic_autosave', false).catch(error => console.warn('Mouse tracking autosave failed:', error));
    }, this.flushIntervalMs);
  }

  stopListeners() {
    document.removeEventListener('pointermove', this._move);
    document.removeEventListener('wheel', this._wheel);
  }

  stop() {
    this.stopListeners();
    clearTimeout(this.scrollActionTimer);
    clearInterval(this.flushTimer);
    this.scrollActionTimer = null;
    this.flushTimer = null;
    this.active = false;
  }

  recordCursor(event = null) {
    if (!this.active) return;
    if (event && Number.isFinite(event.clientX) && Number.isFinite(event.clientY)) {
      if (this.lastCursorX !== null && this.lastCursorY !== null) {
        this.cursorDistancePx += Math.hypot(event.clientX - this.lastCursorX, event.clientY - this.lastCursorY);
      }
      this.lastCursorX = event.clientX;
      this.lastCursorY = event.clientY;
      this.mouseMoveCount += 1;
    }
    this.lastActiveAtMs = Date.now();
  }

  recordScroll(event) {
    if (!this.active || event.deltaY === 0) return;

    const now = Date.now();
    this.recordCursor(event);
    this.lastScrollAtMs = now;
    this.pendingScrollDirections.push(event.deltaY > 0 ? 'down' : 'up');

    clearTimeout(this.scrollActionTimer);
    this.scrollActionTimer = setTimeout(() => {
      this.finishScrollAction(true);
    }, this.actionGapMs);
  }

  finishScrollAction(force = false, now = Date.now()) {
    if (!this.pendingScrollDirections.length) return;
    if (!force && this.lastScrollAtMs !== null && now - this.lastScrollAtMs < this.actionGapMs) return;

    const directions = [...new Set(this.pendingScrollDirections)];
    if (directions.includes('down')) this.scrollDownCount += 1;
    if (directions.includes('up')) this.scrollUpCount += 1;
    // One contiguous scroll action is one timestep. Therefore an up+down
    // action is counted once, not twice.
    this.scrollTimestepCount += 1;

    this.scrollTimestamps.push({
      direction: directions.join('+'),
      timestamp: new Date(now).toISOString()
    });
    this.pendingScrollDirections = [];
    this.lastScrollAtMs = null;
  }

  measurements(now = Date.now()) {
    this.finishScrollAction(false, now);
    return {
      scroll_up_count: this.scrollUpCount,
      scroll_down_count: this.scrollDownCount,
      scroll_timestep_count: this.scrollTimestepCount,
      scroll_event_timestamps: [...this.scrollTimestamps],
      mouse_move_count: this.mouseMoveCount,
      cursor_distance_px: Math.round(this.cursorDistancePx)
    };
  }

  async flush(saveReason = 'periodic_autosave', isFinal = false) {
    if (!this.active && !isFinal) return;
    if (isFinal) this.finishScrollAction(true);
    await this.onSave({
      ...this.measurements(),
      is_final: isFinal
    });
  }
}

window.MouseTracker = MouseTracker;
