// Frontend and API are served by the same FastAPI application.
const API_BASE_URL = '/api';
const questions = [
  {
    id: 'C1',
    topic: 'Positive, Negative, or Zero',
    category: 'Coding & Programming',
    subquestions: [
      { id: 'C1-MAIN', difficulty: 'Easy', expectedTime: '3–5 minutes', title: 'Generate a Java or Python number-checking program', text: 'Write a prompt to generate a Java or Python program that checks whether a number is positive, negative, or zero.' },
      { id: 'C1-SUB1', difficulty: 'Easy', expectedTime: '3–5 minutes', title: 'Explain and test the conditions', text: 'Explain how the generated program handles the positive, negative, and zero conditions. Provide one test input and expected output for each condition.' }
    ]
  },
  {
    id: 'L1',
    topic: 'Four-Friend Seating Arrangement',
    category: 'Logical Reasoning',
    subquestions: [
      { id: 'L1-MAIN', difficulty: 'Easy', expectedTime: '3–5 minutes', title: 'Determine the seating order', text: 'Four friends—A, B, C, and D—sit in a row. A sits at one end, B sits immediately to the right of C, and D is not adjacent to A. Determine the final seating order.' },
      { id: 'L1-SUB1', difficulty: 'Easy', expectedTime: '3–5 minutes', title: 'Explain the elimination process', text: 'Show how each condition eliminates invalid seating arrangements. State whether the information produces one unique order or multiple valid orders.' }
    ]
  },
  {
    id: 'A3',
    topic: 'Train Travel Time and Average Speed',
    category: 'Quantitative Aptitude',
    subquestions: [
      { id: 'A3-MAIN', difficulty: 'Medium', expectedTime: '7–10 minutes', title: 'Calculate total travel time and average speed', text: 'A train travels 240 km. It covers the first half at 60 km/h and the second half at 80 km/h, with a 20-minute stop in between. Determine the total travel time and average speed.' },
      { id: 'A3-SUB1', difficulty: 'Medium', expectedTime: '7–10 minutes', title: 'Separate travel and stoppage time', text: 'Calculate the moving time for each half separately, then explain how including the 20-minute stop changes the overall average speed.' }
    ]
  }
];

const state = {
  participant: null,
  participantSessionToken: '',
  current: 0,
  subquestion: 0,
  selectedLLM: '',
  started: false,
  startTime: null,
  paas: null,
  answerDraft: '',
  codeDraft: '',
  compilerInput: '',
  compilerLanguage: 'java',
  compilerResult: null,
  codeRunning: false,
  finishing: false,
  chatStreaming: false,
  chatAbortController: null,
  chatStopRequested: false,
  submitting: false,
  answers: [],
  messages: [],
  sessionStart: null,
  assessmentDeadline: null,
  timeLimitReached: false,
  timerHandle: null,
  tracking: { clicks: 0, tabSwitches: 0 },
  vision: { status: 'Camera not started', latest: null, samples: 0, calibrated: false },
  facial: { status: 'Facial expression not started', latest: null, samples: 0 },
  lastInteractionAt: Date.now(),
  authToken: '',
  authRole: '',
  llmProviders: null
};

let cameraStream = null;
let cameraCaptureVideo = null;
let visionTimer = null;
let visionFastTimeout = null;
let visionCaptureInFlight = false;
let lastEyePersistAt = 0;
let facialCaptureInFlight = false;
let lastFacialPersistAt = 0;
let monitoringTimer = null;
let dashboardTimer = null;
const keyboardTracker = new KeyboardTracker({
  pauseThresholdMs: 3000,
  // Keyboard telemetry must remain usable when the optional camera signal is
  // unavailable. Do not make one laptop input device gate another.
  isUserPresent: () => true,
  onIdle: measurements => {
    if (state.started && state.participant) {
      saveKeyboardWithRetry(measurements, 'idle_autosave', false).catch(error => {
        showExamError(`Keyboard tracking autosave failed: ${error.message}`);
      });
    }
  }
});
const mouseTracker = new MouseTracker({
  onSave: measurements => saveMouseMeasurements(measurements)
});

const app = document.getElementById('app');
const PARTICIPANT_STATE_KEY = 'cognitrack_participant_state';

function saveParticipantState() {
  if (!state.participant || !state.participantSessionToken) return;
  const saved = {
    participant: state.participant,
    participantSessionToken: state.participantSessionToken,
    current: state.current,
    subquestion: state.subquestion,
    selectedLLM: state.selectedLLM,
    started: state.started,
    startTime: state.startTime,
    paas: state.paas,
    answerDraft: state.answerDraft,
    codeDraft: state.codeDraft,
    compilerInput: state.compilerInput,
    compilerLanguage: state.compilerLanguage,
    answers: state.answers,
    messages: state.messages,
    sessionStart: state.sessionStart,
    assessmentDeadline: state.assessmentDeadline,
    timeLimitReached: state.timeLimitReached,
    tracking: state.tracking,
    lastInteractionAt: state.lastInteractionAt
  };
  sessionStorage.setItem(PARTICIPANT_STATE_KEY, JSON.stringify(saved));
}

function clearParticipantState() {
  sessionStorage.removeItem(PARTICIPANT_STATE_KEY);
}

function resetParticipantSession() {
  clearParticipantState();
  state.participant = null;
  state.participantSessionToken = '';
  state.current = 0;
  state.subquestion = 0;
  state.selectedLLM = '';
  state.started = false;
  state.startTime = null;
  state.paas = null;
  state.answerDraft = '';
  state.codeDraft = '';
  state.compilerInput = '';
  state.compilerResult = null;
  state.codeRunning = false;
  state.finishing = false;
  state.chatStreaming = false;
  state.chatAbortController = null;
  state.chatStopRequested = false;
  state.submitting = false;
  state.answers = [];
  state.messages = [];
  state.sessionStart = null;
  state.assessmentDeadline = null;
  state.timeLimitReached = false;
  state.tracking = { clicks: 0, tabSwitches: 0 };
  state.facial = { status: 'Facial expression not started', latest: null, samples: 0 };
}

function restoreParticipantState() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(PARTICIPANT_STATE_KEY) || 'null');
    if (!saved?.participant || !saved.participantSessionToken) return false;
    Object.assign(state, saved);
    state.finishing = false;
    state.chatStreaming = false;
    state.submitting = false;
    state.codeRunning = false;
    state.vision = { status: 'Camera will reconnect after refresh', latest: null, samples: 0, calibrated: false };
    state.facial = { status: 'Facial expression will reconnect after refresh', latest: null, samples: 0 };
    return true;
  } catch (_error) {
    clearParticipantState();
    return false;
  }
}

async function resumeParticipantState() {
  if (!state.assessmentDeadline) {
    renderInstructions();
    return;
  }
  if (Date.now() >= state.assessmentDeadline) {
    clearParticipantState();
    renderLogin();
    return;
  }
  const questionWasStarted = state.started;
  state.started = false;
  try {
    await startVisionCamera();
    switchToNormalVisionCapture();
  } catch (error) {
    state.vision.status = `Camera unavailable after refresh: ${error.message}`;
  }
  renderExam();
  if (questionWasStarted && state.selectedLLM && state.vision.calibrated) {
    await startCurrentQuestion(currentQuestion());
  }
  saveParticipantState();
}

document.addEventListener('keydown', e => {
  if (state.participant) {
    state.lastInteractionAt = Date.now();
    saveParticipantState();
  }
});
document.addEventListener('click', () => { if (state.participant) { state.tracking.clicks++; state.lastInteractionAt = Date.now(); saveParticipantState(); } });
document.addEventListener('scroll', () => { if (state.participant) { state.lastInteractionAt = Date.now(); saveParticipantState(); } }, { passive: true });
document.addEventListener('visibilitychange', () => { if (document.hidden && state.participant) { state.tracking.tabSwitches++; saveParticipantState(); } });

function participantId() {
  return 'Assigned securely when assessment begins';
}

function renderLogin() {
  clearInterval(dashboardTimer);
  const pid = participantId();
  app.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div class="brand"><div class="logo">C</div><div><h1>CogniTrack <span style="color:#38dd9a">AI</span></h1><small>Enterprise Cognitive Assessment Platform</small></div></div>
        <div class="role-actions">
          <button class="role-login-btn" id="hostLoginBtn" type="button">Host Login</button>
          <button class="role-login-btn admin" id="adminLoginBtn" type="button">Admin Login</button>
          <div class="badge">● Secure Session</div>
        </div>
      </header>
      <section class="login-layout">
        <div class="hero">
          <div class="eyebrow">AI-POWERED TALENT INTELLIGENCE</div>
          <h2>Measure reasoning, coding and AI <span>decision-making.</span></h2>
          <p>Questions adapt to the selected profile while behavioural and interaction analytics run securely in the background.</p>
          <div class="feature-row"><div class="feature-pill">Encrypted Session</div><div class="feature-pill">Adaptive Questions</div><div class="feature-pill">Interaction Analytics</div></div>
        </div>
        <form id="loginForm" class="card login-card">
          <h3>Candidate Sign In</h3>
          <div class="field"><label>Participant ID (Auto-generated)</label><input class="readonly" id="participantId" value="${pid}" readonly /></div>
          <div class="field"><label>Full Name</label><input id="fullName" placeholder="Enter full name" required /></div>
          <div class="grid-2">
            <div class="field"><label>Age Group</label><select id="ageGroup" required><option value="">Select age group</option><option>18–22</option><option>23–28</option><option>29–35</option><option>36–45</option><option>46+</option></select></div>
            <div class="field"><label>Domain</label><select id="domain" required><option value="">Select domain</option><option>Artificial Intelligence</option><option>Computer Science</option><option>Data Science</option><option>Software Development</option><option>Other</option></select></div>
          </div>
          <div class="field"><label>AI Familiarity Level</label><select id="level" required><option value="">Select familiarity level</option><option>Beginner</option><option>Intermediate</option><option>Advanced</option></select></div>
          <label class="consent"><input id="consent" type="checkbox" /> <span>I agree to participate in this assessment and allow interaction data to be collected for the stated research purpose.</span></label>
          <button class="primary" type="submit">Login →</button>
          <div class="error" id="loginError"></div>
        </form>
      </section>
    </div>`;

  document.getElementById('hostLoginBtn').addEventListener('click', () => renderRoleLogin('host'));
  document.getElementById('adminLoginBtn').addEventListener('click', () => renderRoleLogin('admin'));

  document.getElementById('loginForm').addEventListener('submit', async e => {
    e.preventDefault();
    const consent = document.getElementById('consent').checked;
    if (!consent) return document.getElementById('loginError').textContent = 'Consent is required to continue.';
    state.participant = {
      participant_id: '',
      full_name: document.getElementById('fullName').value.trim(),
      age_group: document.getElementById('ageGroup').value,
      domain: document.getElementById('domain').value,
      ai_familiarity: document.getElementById('level').value
    };
    const loginTime = Date.now();
    state.sessionStart = new Date(loginTime).toISOString();
    state.assessmentDeadline = null;
    state.timeLimitReached = false;
    try {
      const result = await safePost('/sessions/start', state.participant);
      state.participant.participant_id = result.participant_id;
      state.participantSessionToken = result.participant_session_token;
      saveParticipantState();
      state.lastInteractionAt = Date.now();
      clearInterval(monitoringTimer);
      monitoringTimer = setInterval(() => sendMonitoringHeartbeat('active'), 10000);
      sendMonitoringHeartbeat('active').catch(error => console.warn('Monitoring heartbeat failed:', error));
      renderInstructions();
    } catch (error) {
      document.getElementById('loginError').textContent = `Unable to start: ${error.message}`;
      state.participant = null;
      state.assessmentDeadline = null;
    }
  });
}

function renderInstructions() {
  app.innerHTML = `<div class="instructions-overlay"><div class="card instructions-card"><div class="instruction-heading"><h2>Assessment Instructions</h2></div><ol><li>You have one hour after clicking <strong>Start Assessment</strong> to complete the entire assessment.</li><li>The assessment contains three tasks.</li><li>Each task has one main question followed by one related sub-question.</li><li>Complete both questions before the assessment advances to the next task.</li><li>Keep your face visible and remain in a well-lit area while camera tracking runs in the background.</li><li>Select an LLM to begin, and switch between ChatGPT, Ollama, and Groq at any time.</li><li>Submit your final answer for all six questions.</li></ol><button class="primary" id="startAssessment" type="button">Start Assessment</button></div></div>`;
  saveParticipantState();
  document.getElementById('startAssessment').addEventListener('click', async event => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = 'Requesting webcam permission…';
    try {
      await startVisionCamera();
      const assessmentStartedAt = new Date().toISOString();
      await safePost('/sessions/assessment-start', {
        participant_id: state.participant.participant_id,
        assessment_started_at: assessmentStartedAt,
        camera_permission: 'granted',
        calibration_status: 'not-required'
      }, state.participantSessionToken);
      state.sessionStart = assessmentStartedAt;
      state.assessmentDeadline = Date.now() + 60 * 60 * 1000;
      state.timeLimitReached = false;
      saveParticipantState();
      renderExam();
      switchToNormalVisionCapture();
    } catch (error) {
      state.vision.status = `Camera failed: ${error.message}`;
      button.disabled = false;
      button.textContent = 'Start Assessment';
      const existing = document.getElementById('cameraStartError');
      if (existing) existing.textContent = error.message;
      else button.insertAdjacentHTML('afterend', `<div class="error" id="cameraStartError">${escapeHtml(error.message)}</div>`);
      stopVisionCamera();
    }
  });
}

function currentQuestion() {
  const group = questions[state.current];
  const questionPart = state.subquestion === 0 ? 'Main Question' : 'Sub-question 1 of 1';
  return { ...group, ...group.subquestions[state.subquestion], topicId: group.id, questionPart };
}

function renderLlmOption(name, configuredKey) {
  const configured = state.llmProviders?.[configuredKey];
  const unavailable = configured === false;
  const selected = state.selectedLLM === name;
  return `<option value="${name}" ${selected ? 'selected' : ''} ${unavailable ? 'disabled' : ''}>${name}${unavailable ? ' (not configured)' : ''}</option>`;
}

function renderExam() {
  const q = currentQuestion();
  const codingQuestion = q.category === 'Coding & Programming';
  app.innerHTML = `
    <div class="exam-shell">
      <header class="exam-top">
        <div class="brand"><div class="logo">C</div><div><h1>CogniTrack <span style="color:#38dd9a">AI</span></h1><small>${state.participant.participant_id}</small></div></div>
        <div class="center">Task ${state.current + 1} of ${questions.length} · ${q.questionPart}</div>
        <div class="exam-actions"><div class="timer" id="timer">--:--</div><button class="danger-btn" id="endSession" ${state.finishing ? 'disabled' : ''}>${state.finishing ? 'Ending Session…' : 'End Session'}</button></div>
      </header>
      <main class="exam-grid">
        <section class="stack">
          <article class="card panel">
            <div class="section-label">${q.questionPart} · ${q.category} · ${q.difficulty}</div>
            <h2 class="question-title">${q.title}</h2>
            <p class="question-text">${q.text}</p>
          </article>
          ${codingQuestion ? `<article class="card panel">
            <div class="section-label">Online Code Compiler</div>
            <p class="compiler-note">Choose Java or Python, then run your code in an external sandbox before submitting. Java programs must use <code>public class Main</code>.</p>
            <label class="compiler-input-label" for="compilerLanguage">Programming language</label>
            <select id="compilerLanguage" class="compiler-language" ${state.started ? '' : 'disabled'}><option value="java" ${state.compilerLanguage === 'java' ? 'selected' : ''}>Java</option><option value="python" ${state.compilerLanguage === 'python' ? 'selected' : ''}>Python</option></select>
            <textarea id="codeEditor" class="code-editor" spellcheck="false" placeholder="${state.compilerLanguage === 'java' ? 'public class Main {\n  public static void main(String[] args) {\n    // write your code here\n  }\n}' : 'print(\"Hello, world!\")'}" ${state.started ? '' : 'disabled'}>${escapeHtml(state.codeDraft)}</textarea>
            <label class="compiler-input-label" for="compilerInput">Standard input (optional)</label>
            <textarea id="compilerInput" class="compiler-input" spellcheck="false" placeholder="Input passed to your program" ${state.started ? '' : 'disabled'}>${escapeHtml(state.compilerInput)}</textarea>
            <div class="actions"><button class="secondary" id="runCode" ${state.started && !state.codeRunning ? '' : 'disabled'}>${state.codeRunning ? 'Running…' : 'Run Code'}</button></div>
            ${renderCompilerResult()}</article>` : ''}
          <article class="card panel">
            <div class="section-label">Your Final Answer</div>
            <textarea id="answer" class="answer-box" placeholder="Paste or write your final answer here..." ${state.started ? '' : 'disabled'}>${escapeHtml(state.answerDraft)}</textarea>
            <div class="actions"><button class="success-btn" id="submitAnswer" ${state.submitting ? 'disabled' : ''}>${state.submitting ? 'Submitting...' : 'Submit Answer'}</button></div>
            <div class="error" id="examError"></div>
          </article>
          <article class="card panel" id="visionMetricsPanel">
            <div class="section-label">Live Behaviour</div>
            <div id="visionMetrics">${renderVisionMetrics()}</div>
          </article>
        </section>
        <aside class="stack">
          <article class="card panel">
            <div class="section-label">Choose AI Model</div>
            <p style="color:var(--muted)">Choose the AI model for this question part.</p>
            <select class="ai-model-select" id="aiModelSelect">
              <option value="">Select an AI model</option>
              ${renderLlmOption('ChatGPT', 'chatgpt_configured')}
              ${renderLlmOption('Ollama', 'ollama_configured')}
              ${renderLlmOption('Groq', 'groq_configured')}
            </select>
          </article>
          ${state.selectedLLM ? `<article class="card panel chat-box">
            <div class="section-label">Ask ${state.selectedLLM}</div>
            <div class="messages" id="messages">${renderMessages()}</div>
            <div class="chat-row"><input id="chatInput" class="chat-input" placeholder="${state.selectedLLM ? `Ask ${state.selectedLLM}...` : 'Choose an AI model first'}" ${state.started && !state.chatStreaming ? '' : 'disabled'} />${state.chatStreaming ? '<button class="primary ask-model-btn stop-model-btn" style="width:auto" id="stopChat" type="button" title="Stop generating" aria-label="Stop generating">■</button>' : `<button class="primary ask-model-btn" style="width:auto" id="sendChat" type="button">${state.selectedLLM ? `Ask ${state.selectedLLM}` : 'Send'}</button>`}</div>
          </article>` : ''}
        </aside>
      </main>
    </div>`;

  bindExamEvents(q);
  startDisplayTimer();
  saveParticipantState();
}

function renderVisionMetrics() {
  const metrics = state.vision.latest;
  if (!metrics) return '<small>Eye metrics will appear when camera tracking starts.</small>';
  const facial = state.facial.latest;
  const facialScores = facial ? [
    ['Angry', 'angry_score'], ['Disgust', 'disgust_score'], ['Fearful', 'fearful_score'],
    ['Happy', 'happy_score'], ['Neutral', 'neutral_score'], ['Sad', 'sad_score'],
    ['Surprised', 'surprised_score']
  ].map(([label, key]) => `<span>${label} <strong>${Math.round(Number(facial[key] || 0) * 100)}%</strong></span>`).join('') : '';
  return `<div class="vision-metrics">
    <span>Eye direction <strong>${escapeHtml(metrics.direction || '—')}</strong></span>
    <span>Blinks <strong>${Number(metrics.blink_count || 0)}</strong></span>
    <span>Fatigue <strong>${escapeHtml(metrics.fatigue || '—')}</strong></span>
    <span>Tracking confidence <strong>${Math.round(Number(metrics.tracking_confidence || 0) * 100)}%</strong></span>
    ${facial ? `<span>Facial expression <strong>${escapeHtml(facial.emotion || 'unknown')}</strong></span><span>Expression confidence <strong>${Math.round(Number(facial.model_confidence || 0) * 100)}%</strong></span>${facialScores}` : ''}
  </div>`;
}

function refreshVisionMetrics() {
  const container = document.getElementById('visionMetrics');
  if (container) container.innerHTML = renderVisionMetrics();
}

async function startVisionCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Webcam access requires HTTPS or localhost');
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    cameraCaptureVideo = document.createElement('video');
    cameraCaptureVideo.srcObject = cameraStream;
    cameraCaptureVideo.muted = true;
    cameraCaptureVideo.playsInline = true;
    await cameraCaptureVideo.play();
    state.vision.status = 'Camera connected; eye tracking active';
    state.vision.calibrated = true;
  } catch (error) {
  state.vision.status = `Camera unavailable: ${error.message}`;
    throw new Error(`Webcam access is required to start the assessment: ${error.message}`);
  }
}

function stopVisionCamera() {
  clearInterval(visionTimer);
  clearTimeout(visionFastTimeout);
  visionTimer = null;
  visionFastTimeout = null;
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  cameraCaptureVideo = null;
  facialCaptureInFlight = false;
  lastFacialPersistAt = 0;
  state.vision.calibrated = false;
  state.facial.status = 'Facial expression stopped';
}

function switchToNormalVisionCapture() {
  clearInterval(visionTimer);
  clearTimeout(visionFastTimeout);
  // Blink closures are brief; 500 ms sampling could skip an entire blink.
  // The in-flight guard still prevents overlapping requests when inference is slower.
  visionTimer = setInterval(() => captureVisionFrame().catch(error => console.error('Eye tracking frame failed:', error)), 200);
}

async function captureVisionFrame() {
  if (!state.vision.calibrated || visionCaptureInFlight) return;
  visionCaptureInFlight = true;
  try {
    const result = await captureEyeTracking();
    if (!result) return;
    state.vision.latest = { ...(state.vision.latest || {}), ...result.metrics };
    state.vision.samples++;
    state.vision.status = result.metrics.face_detected ? 'Analysis active' : 'No face detected';
    refreshVisionMetrics();
    captureFacialExpression().catch(error => {
      state.facial.status = `Facial expression paused: ${error.message}`;
    });
  } catch (error) {
    state.vision.status = `Analysis paused: ${error.message}`;
  } finally {
    visionCaptureInFlight = false;
  }
}

async function captureFacialExpression() {
  if (!cameraStream || !state.participant || facialCaptureInFlight || !cameraCaptureVideo?.videoWidth) return;
  facialCaptureInFlight = true;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = Math.round(320 * cameraCaptureVideo.videoHeight / cameraCaptureVideo.videoWidth);
    canvas.getContext('2d').drawImage(cameraCaptureVideo, 0, 0, canvas.width, canvas.height);
    const q = currentQuestion();
    const now = Date.now();
    const startedAt = state.sessionStart ? new Date(state.sessionStart).getTime() : now;
    const elapsedSecond = Math.max(1, Math.floor(Math.max(0, now - startedAt) / 1000) + 1);
    const persist = now - lastFacialPersistAt >= 1000;
    if (persist) lastFacialPersistAt = now;
    const response = await safePost('/facial-expression/frame', {
      participant_id: state.participant.participant_id,
      question_id: q.id,
      task_number: state.current + 1,
      elapsed_second: elapsedSecond,
      captured_at: new Date().toISOString(),
      image: canvas.toDataURL('image/jpeg', 0.75),
      persist
    }, state.participantSessionToken);
    state.facial.latest = { ...(state.facial.latest || {}), ...response.metrics };
    state.facial.samples++;
    state.facial.status = response.metrics.face_detected ? 'Facial expression active' : 'No face detected';
    refreshVisionMetrics();
  } finally {
    facialCaptureInFlight = false;
  }
}

async function captureEyeTracking() {
  if (!cameraStream || !state.participant) return;
  const track = cameraStream.getVideoTracks()[0];
  if (!track || track.readyState !== 'live') throw new Error('camera stream ended');
  if (!cameraCaptureVideo?.videoWidth) throw new Error('camera is not ready');
  const canvas = document.createElement('canvas');
  canvas.width = 192;
  canvas.height = Math.round(192 * cameraCaptureVideo.videoHeight / cameraCaptureVideo.videoWidth);
  canvas.getContext('2d').drawImage(cameraCaptureVideo, 0, 0, canvas.width, canvas.height);
  const q = currentQuestion();
  const now = Date.now();
  const persist = now - lastEyePersistAt >= 1000;
  if (persist) lastEyePersistAt = now;
  return safePost('/eye-tracking/frame', {
    participant_id: state.participant.participant_id,
    question_id: q.id,
    task_number: state.current + 1,
    captured_at: new Date().toISOString(),
    image: canvas.toDataURL('image/jpeg', 0.6),
    persist
  }, state.participantSessionToken);
}

async function sendMonitoringHeartbeat(status = 'active') {
  if (!state.participant) return;
  const q = currentQuestion();
  const completedParts = questions.slice(0, state.current).reduce((total, question) => total + question.subquestions.length, 0) + state.subquestion;
  const totalParts = questions.reduce((total, question) => total + question.subquestions.length, 0);
  const progressPercent = status === 'completed' ? 100 : Math.round(completedParts / totalParts * 1000) / 10;
  return safePut('/monitoring/heartbeat', {
    participant_id: state.participant.participant_id,
    status,
    question_id: q.id,
    question_number: state.current + 1,
    subquestion_number: state.subquestion + 1,
    progress_percent: progressPercent,
    question_started: state.started,
    inactivity_seconds: Math.floor((Date.now() - state.lastInteractionAt) / 1000),
    session_duration_seconds: Math.max(0, Math.floor((Date.now() - new Date(state.sessionStart).getTime()) / 1000)),
    tab_switches: state.tracking.tabSwitches,
    vision_status: state.vision.status,
    fatigue: state.vision.latest?.fatigue || 'unknown',
    captured_at: new Date().toISOString()
  }, state.participantSessionToken);
}

function renderMessages() {
  if (!state.messages.length && state.selectedLLM) {
    const firstName = escapeHtml((state.participant?.full_name || 'there').trim().split(/\s+/)[0]);
    const modelName = escapeHtml(state.selectedLLM);
    const modelIcon = state.selectedLLM === 'ChatGPT' ? 'C' : state.selectedLLM === 'Ollama' ? 'O' : 'Q';
    return `<div class="model-welcome"><div class="model-welcome-icon model-${state.selectedLLM.toLowerCase()}">${modelIcon}</div><h3>Hello ${firstName}, how can I help you today?</h3><p>You are now using ${modelName}. Type your question below to get started.</p></div>`;
  }
  if (!state.messages.length) return `<div class="model-welcome"><h3>Choose your AI assistant</h3><p>Select ChatGPT, Ollama, or Groq to begin.</p></div>`;
  return state.messages.map(m=>`<div class="message ${m.role}">${escapeHtml(m.content)}</div>`).join('');
}
function renderCompilerResult() {
  if (!state.compilerResult) return '';
  const result = state.compilerResult;
  const sections = [];
  if (result.compile_output) sections.push(`Compiler diagnostics:\n${result.compile_output}`);
  if (result.output) sections.push(`Program output:\n${result.output}`);
  const output = sections.join('\n\n') || 'Program completed with no output.';
  const failed = result.error || Number(result.exit_code ?? 0) !== 0;
  const label = result.error ? 'Compiler error' : failed ? 'Compilation / runtime result' : 'Program output';
  return `<div class="compiler-result ${failed ? 'compiler-result-error' : ''}"><strong>${label}</strong><pre>${escapeHtml(output)}</pre></div>`;
}
function bindExamEvents(q) {
  const answerEditor = document.getElementById('answer');
  document.getElementById('aiModelSelect').addEventListener('change', async event => {
    if (!event.target.value) return;
    state.selectedLLM = event.target.value;
    if (state.started) renderExam();
    else await startCurrentQuestion(q);
  });
  document.getElementById('answer').addEventListener('input', event => { state.answerDraft = event.target.value; saveParticipantState(); });
  const codeEditor = document.getElementById('codeEditor');
  if (codeEditor) {
    codeEditor.addEventListener('input', event => { state.codeDraft = event.target.value; saveParticipantState(); });
    document.getElementById('compilerLanguage').addEventListener('change', event => { state.compilerLanguage = event.target.value; renderExam(); });
    document.getElementById('compilerInput').addEventListener('input', event => { state.compilerInput = event.target.value; saveParticipantState(); });
    document.getElementById('runCode').onclick = runCode;
  }
  const sendChatButton = document.getElementById('sendChat');
  const stopChatButton = document.getElementById('stopChat');
  const chatInput = document.getElementById('chatInput');
  // Capture Backspace from the whole assessment page, including laptop
  // keyboard input after focus moves between answer, chat, and code fields.
  keyboardTracker.attach(document);
  if (sendChatButton && chatInput) {
    sendChatButton.onclick = sendChat;
    chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
  }
  if (stopChatButton) stopChatButton.onclick = stopChat;
  document.getElementById('submitAnswer').onclick = submitCurrentAnswer;
  document.getElementById('endSession').onclick = () => {
    if (state.finishing) return;
    finishAssessment(true);
  };
}

async function startCurrentQuestion(q) {
  if (!state.selectedLLM || state.started) return;
  const startTime = Date.now();
  try {
    await safePost('/questions/start', { participant_id: state.participant.participant_id, topic_id: q.topicId, question_id: q.id, question_number: state.current + 1, subquestion_number: state.subquestion + 1, question_part: q.questionPart, llm: state.selectedLLM, trial_number: state.current * questions[state.current].subquestions.length + state.subquestion + 1, started_at: new Date(startTime).toISOString() }, state.participantSessionToken);
    state.started = true;
    state.startTime = startTime;
    keyboardTracker.reset(startTime);
    mouseTracker.start(startTime);
    renderExam();
  } catch (error) {
    showExamError(`Unable to start question: ${error.message}`);
  }
}

function goToPreviousQuestion() {
  if (state.started || state.submitting) {
    return showExamError('Submit or end the current question before going back.');
  }
  if (state.current === 0 && state.subquestion === 0) return;
  if (state.subquestion > 0) {
    state.subquestion--;
  } else {
    state.current--;
    state.subquestion = questions[state.current].subquestions.length - 1;
  }
  const previous = [...state.answers].reverse().find(answer => answer.question_id === currentQuestion().id);
  state.selectedLLM = previous?.llm || '';
  state.answerDraft = previous?.answer || '';
  state.messages = previous?.chat_history || [];
  state.startTime = null;
  state.started = false;
  renderExam();
}

async function runCode() {
  if (!state.started || state.codeRunning) return;
  const sourceCode = document.getElementById('codeEditor').value.trim();
  state.codeDraft = sourceCode;
  state.compilerInput = document.getElementById('compilerInput').value;
  if (!sourceCode) return showExamError('Write code before running it.');
  state.codeRunning = true;
  renderExam();
  try {
    state.compilerResult = await safePost('/code/run', { language: state.compilerLanguage, source_code: sourceCode, stdin: state.compilerInput }, state.participantSessionToken);
  } catch (error) {
    state.compilerResult = { error: true, output: error.message };
  } finally {
    state.codeRunning = false;
    renderExam();
  }
}

async function sendChat() {
  if (state.chatStreaming) return;
  if (!state.started) return showExamError('Select an LLM to begin the question.');
  if (!state.selectedLLM) return showExamError('Select an LLM first.');
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;
  state.messages.push({ role:'user', content:text });
  input.value = '';
  const q = currentQuestion();
  const payload = { participant_id: state.participant.participant_id, topic_id:q.topicId, question_id:q.id, task_name:q.title, question_number:state.current + 1, subquestion_number:state.subquestion + 1, question_part:q.questionPart, trial_number:state.current * questions[state.current].subquestions.length + state.subquestion + 1, provider:state.selectedLLM, message:text, history:state.messages.map(message => ({ ...message })) };
  const assistantMessage = { role:'assistant', content:state.selectedLLM === 'Ollama' ? 'Ollama is answering…' : 'Searching the web…' };
  state.messages.push(assistantMessage);
  state.chatStreaming = true;
  state.chatStopRequested = false;
  state.chatAbortController = new AbortController();
  renderExam();
  let receivedToken = false;
  try {
    await safeStreamPost('/llm/chat/stream', payload, event => {
      if (event.type === 'status' && !receivedToken) assistantMessage.content = event.message;
      if (event.type === 'token') {
        if (!receivedToken) assistantMessage.content = '';
        receivedToken = true;
        assistantMessage.content += event.content;
      }
      if (event.type === 'error') throw new Error(event.message);
      const messagesElement = document.getElementById('messages');
      if (messagesElement) {
        messagesElement.innerHTML = renderMessages();
        messagesElement.scrollTop = messagesElement.scrollHeight;
      }
    }, state.participantSessionToken, state.chatAbortController.signal);
    if (!receivedToken) assistantMessage.content = 'The model completed without returning text. Please retry.';
  } catch (error) {
    if (state.chatStopRequested || error.name === 'AbortError') {
      assistantMessage.content = `${assistantMessage.content || ''}\n\nGeneration stopped.`.trim();
    } else {
      assistantMessage.role = 'error';
      assistantMessage.content = `Message failed: ${error.message}. Please retry.`;
    }
  } finally {
    state.chatStreaming = false;
    state.chatAbortController = null;
    state.chatStopRequested = false;
  }
  renderExam();
}

function stopChat() {
  if (!state.chatStreaming || !state.chatAbortController) return;
  state.chatStopRequested = true;
  state.chatAbortController.abort();
}

async function submitCurrentAnswer() {
  if (state.submitting) return;
  const answer = document.getElementById('answer').value.trim();
  if (!state.started) return showExamError('Select an LLM to begin the question.');
  if (!state.selectedLLM) return showExamError('Select an LLM first.');
  if (answer.length < 15) return showExamError('Please provide a complete answer before submitting.');
  state.submitting = true;
  state.answerDraft = answer;
  const q = currentQuestion();
  const record = {
    participant_id: state.participant.participant_id,
    topic_id: q.topicId,
    question_id: q.id,
    question_number: state.current + 1,
    subquestion_number: state.subquestion + 1,
    question_part: q.questionPart,
    llm: state.selectedLLM,
    trial_number: state.current * questions[state.current].subquestions.length + state.subquestion + 1,
    answer,
    paas_rating: null,
    started_at: new Date(state.startTime).toISOString(),
    submitted_at: new Date().toISOString(),
    duration_seconds: Math.round((Date.now() - state.startTime)/1000),
    chat_history: state.messages,
    interaction_summary: { ...state.tracking }
  };
  try {
    await saveKeyboardWithRetry(keyboardTracker.finalize(), 'question_submit', true);
    await mouseTracker.flush('question_submit', true);
    await safePost('/answers/submit', record, state.participantSessionToken);
  } catch (error) {
    state.submitting = false;
    renderExam();
    showExamError(`Answer was not saved: ${error.message}. Please retry.`);
    return;
  }
  mouseTracker.stop();
  state.answers.push(record);
  const completedLastSubquestion = state.subquestion === questions[state.current].subquestions.length - 1;
  const completedLastTopic = state.current === questions.length - 1;
  if (completedLastSubquestion && completedLastTopic) {
    state.submitting = false;
    state.started = false;
    state.startTime = null;
    return renderPaasAssessment();
  }
  if (completedLastSubquestion) {
    state.current++;
    state.subquestion = 0;
  } else {
    state.subquestion++;
  }
  state.started = false;
  state.startTime = null;
  state.paas = null;
  state.answerDraft = '';
  state.codeDraft = '';
  state.compilerInput = '';
  state.compilerLanguage = 'java';
  state.compilerResult = null;
  state.codeRunning = false;
  state.messages = [];
  state.submitting = false;
  renderExam();
  if (state.selectedLLM) startCurrentQuestion(currentQuestion());
}

function renderPaasAssessment() {
  clearInterval(state.timerHandle);
  app.innerHTML = `<div class="success-screen"><div class="card success-card"><h1>Assessment Complete</h1><p>How mentally demanding was the overall assessment?</p><div class="paas-grid">${Array.from({length:10},(_,i)=>`<button class="paas-btn ${state.paas===i+1?'active':''}" data-paas="${i+1}" type="button">${i+1}</button>`).join('')}</div><div class="scale-labels"><span>Very Easy</span><span>Very Hard</span></div><button class="primary" id="submitAssessment" type="button" ${state.paas ? '' : 'disabled'}>Submit Assessment</button><div class="error" id="paasError"></div></div></div>`;
  document.querySelectorAll('.paas-btn').forEach(button => {
    button.addEventListener('click', () => {
      state.paas = Number(button.dataset.paas);
      renderPaasAssessment();
    });
  });
  document.getElementById('submitAssessment').addEventListener('click', () => finishAssessment(false));
}

async function finishAssessment(endedEarly) {
  if (state.finishing) return;
  state.finishing = true;
  const endButton = document.getElementById('endSession');
  if (endButton) {
    endButton.disabled = true;
    endButton.textContent = 'Ending Session…';
  }
  clearInterval(state.timerHandle);
  const payload = {
    participant: state.participant,
    session_started_at: state.sessionStart,
    session_ended_at: new Date().toISOString(),
    ended_early: endedEarly,
    overall_paas_rating: endedEarly ? null : state.paas,
    answers: state.answers,
    interaction_summary: state.tracking
  };
  let completionError = '';
  try {
    if (endedEarly && state.started && state.participant) {
      try {
        await saveKeyboardWithRetry(keyboardTracker.finalize(), 'session_end', true);
      } catch (error) {
        console.warn('Final keyboard tracking save failed:', error);
      }
      try {
        await mouseTracker.flush('session_end', true);
      } catch (error) {
        console.warn('Final mouse tracking save failed:', error);
      } finally {
        mouseTracker.stop();
      }
    }
    await safePost('/sessions/complete', payload, state.participantSessionToken);
  } catch (error) {
    completionError = error.message;
    localStorage.setItem('cognitrack_pending_session', JSON.stringify(payload));
  }
  if (!completionError) localStorage.removeItem('cognitrack_pending_session');
  clearParticipantState();
  clearInterval(monitoringTimer);
  await sendMonitoringHeartbeat(endedEarly ? 'ended' : 'completed').catch(error => console.warn('Final monitoring heartbeat failed:', error));
  clearInterval(visionTimer);
  clearTimeout(visionFastTimeout);
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  cameraCaptureVideo = null;
  localStorage.setItem('cognitrack_last_session', JSON.stringify(payload));
  if (endedEarly) {
    const candidateId = escapeHtml(state.participant?.participant_id || 'Not assigned');
    const heading = completionError ? 'Session Ended' : 'Session Ended Successfully';
    const statusMessage = completionError
      ? `Your session has ended, but the final save could not be confirmed: ${escapeHtml(completionError)}`
      : 'Your session has ended successfully. Your completed responses have been saved.';
    app.innerHTML = `<div class="success-screen"><div class="card success-card"><div class="success-icon">✓</div><h1>${heading}</h1><p style="color:var(--muted);font-size:18px"><strong>Candidate ID:</strong> ${candidateId}</p><p>${statusMessage}</p><p style="color:var(--muted)">Returning to the login page in a few seconds.</p><button class="primary" id="restart" type="button">Return to Login Now</button></div></div>`;
    let returnedToLogin = false;
    const returnToLogin = () => {
      if (returnedToLogin) return;
      returnedToLogin = true;
      resetParticipantSession();
      renderLogin();
    };
    document.getElementById('restart').onclick = returnToLogin;
    window.setTimeout(returnToLogin, 3500);
    return;
  }
  app.innerHTML = `<div class="success-screen"><div class="card success-card"><div class="success-icon">✓</div><h1>${endedEarly ? 'Session Ended' : 'Assessment Submitted Successfully'}</h1><p style="color:var(--muted);font-size:18px">Participant ID: ${state.participant.participant_id}</p><p>${endedEarly ? 'Your completed responses have been saved.' : 'All three main questions and three related sub-questions are saved.'}</p><button class="primary" id="restart">Return to Login</button></div></div>`;
  if (completionError) {
    app.innerHTML = `<div class="success-screen"><div class="card success-card"><div class="success-icon">!</div><h1>Session Ended</h1><p style="color:var(--muted);font-size:18px">Participant ID: ${escapeHtml(state.participant.participant_id)}</p><p>You have been logged out. The server could not confirm the final save: ${escapeHtml(completionError)}</p><button class="primary" id="restart">Return to Login</button></div></div>`;
  }
  document.getElementById('restart').onclick = () => location.reload();
}

function startDisplayTimer() {
  clearInterval(state.timerHandle);
  const timer = document.getElementById('timer');
  const render = () => {
    if (!timer) return;
    const remaining = Math.max(0, Math.ceil(((state.assessmentDeadline || Date.now()) - Date.now()) / 1000));
    const hours = Math.floor(remaining / 3600);
    const minutes = Math.floor((remaining % 3600) / 60);
    const seconds = remaining % 60;
    timer.textContent = `${String(hours).padStart(2,'0')}:${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}`;
    timer.classList.toggle('time-up', remaining === 0);
    timer.title = remaining === 0 ? 'Assessment time has ended.' : 'Time remaining for the assessment';
    if (remaining === 0 && !state.timeLimitReached) {
      state.timeLimitReached = true;
      clearInterval(state.timerHandle);
      finishAssessment(true);
    }
  };
  render();
  state.timerHandle = setInterval(render,1000);
}
function showExamError(msg) { const el = document.getElementById('examError'); if (el) el.textContent = msg; }
function escapeHtml(s) { return s.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}[c])); }
async function safePost(path, body, token='') {
  return safeJsonRequest('POST', path, body, token);
}

async function safeStreamPost(path, body, onEvent, token='', signal) {
  const headers = {'Content-Type':'application/json'};
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { method:'POST', headers, body:JSON.stringify(body), signal });
  } catch (_error) {
    throw new Error('the server is unavailable');
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `server returned HTTP ${response.status}`);
  }
  if (!response.body) throw new Error('streaming is not supported by this browser');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream:!done});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

async function safePut(path, body, token='') {
  return safeJsonRequest('PUT', path, body, token);
}

async function safeJsonRequest(method, path, body, token='') {
  const headers = {'Content-Type':'application/json'};
  if (token) headers.Authorization = `Bearer ${token}`;
  let res;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      res = await fetch(`${API_BASE_URL}${path}`, { method, headers, body:JSON.stringify(body) });
      break;
    } catch (_error) {
      if (attempt < 2) await new Promise(resolve => setTimeout(resolve, 400));
    }
  }
  if (!res) throw new Error(`cannot connect to the local server. Reload ${window.location.origin}/ and try again.`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `server returned HTTP ${res.status}`);
  return data;
}

function saveKeyboardMeasurements(measurements, saveReason, isFinal) {
  const q = currentQuestion();
  return safePut('/keyboard', {
    participant_id: state.participant.participant_id,
    question_id: q.id,
    question_category: q.category,
    ...measurements,
    is_final: isFinal
  }, state.participantSessionToken);
}

function saveMouseMeasurements(measurements) {
  const q = currentQuestion();
  return safePut('/mouse', {
    participant_id: state.participant.participant_id,
    question_id: q.id,
    question_category: q.category,
    scroll_up_count: measurements.scroll_up_count,
    scroll_down_count: measurements.scroll_down_count,
    scroll_timestep_count: measurements.scroll_timestep_count,
    scroll_event_timestamps: measurements.scroll_event_timestamps,
    mouse_move_count: measurements.mouse_move_count,
    cursor_distance_px: measurements.cursor_distance_px
  }, state.participantSessionToken);
}

async function saveKeyboardWithRetry(measurements, saveReason, isFinal) {
  try {
    return await saveKeyboardMeasurements(measurements, saveReason, isFinal);
  } catch (_firstError) {
    await new Promise(resolve => setTimeout(resolve, 500));
    return saveKeyboardMeasurements(measurements, saveReason, isFinal);
  }
}



function renderRoleLogin(role) {
  const roleName = role === 'admin' ? 'Admin' : 'Host';
  app.innerHTML = `
    <div class="role-page">
      <header class="topbar">
        <div class="brand"><div class="logo">C</div><div><h1>CogniTrack <span style="color:#38dd9a">AI</span></h1><small>${roleName} Portal</small></div></div>
        <button class="secondary" id="backToParticipant">← Participant Login</button>
      </header>
      <main class="role-login-wrap">
        <form id="roleLoginForm" class="card role-login-card">
          <div class="role-shield">${role === 'admin' ? 'A' : 'H'}</div>
          <div class="eyebrow">SECURE ${roleName.toUpperCase()} ACCESS</div>
          <h2>${roleName} Login</h2>
          <p>Enter your authorized ${roleName.toLowerCase()} credentials.</p>
          <div class="field"><label>Username</label><input id="roleUsername" autocomplete="username" required placeholder="Enter username" /></div>
          <div class="field"><label>Password</label><input id="rolePassword" type="password" autocomplete="current-password" required placeholder="Enter password" /></div>
          <button class="primary" type="submit">Login to ${roleName} Portal</button>
          <div class="error" id="roleLoginError"></div>
        </form>
      </main>
    </div>`;

  document.getElementById('backToParticipant').onclick = renderLogin;
  document.getElementById('roleLoginForm').addEventListener('submit', async event => {
    event.preventDefault();
    const errorBox = document.getElementById('roleLoginError');
    errorBox.textContent = '';
    let result;
    try {
      const response = await fetch(`${API_BASE_URL}/auth/${role}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: document.getElementById('roleUsername').value.trim(),
          password: document.getElementById('rolePassword').value
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        errorBox.textContent = data.detail || `Login failed (HTTP ${response.status}).`;
        return;
      }
      result = data;
    } catch (error) {
    errorBox.textContent = `Backend is not running at ${window.location.origin}. Start the local server and try again.`;
      return;
    }
    if (!result?.success) {
      errorBox.textContent = 'Authentication failed. Please try again.';
      return;
    }
    state.authToken = result.access_token;
    state.authRole = result.role;
    sessionStorage.setItem('cognitrack_auth_token', result.access_token);
    sessionStorage.setItem('cognitrack_auth_role', result.role);
    renderRoleDashboard(result.role);
  });
}

function renderRoleDashboard(role) {
  const roleName = role === 'admin' ? 'Admin' : 'Host';
  const dashboardTitle = role === 'admin' ? 'Administrator Data Dashboard' : 'Host Agent Orchestration Console';
  const dashboardDescription = role === 'admin'
    ? 'Platform-wide participant, session, answer, chat, eye, keyboard, and monitoring information.'
    : 'The Host Agent dispatches each heartbeat to specialist agents and combines their findings for human review.';
  app.innerHTML = `
    <div class="role-page">
      <header class="topbar">
        <div class="brand"><div class="logo">C</div><div><h1>CogniTrack <span style="color:#38dd9a">AI</span></h1><small>${roleName} Dashboard</small></div></div>
        <div class="role-actions"><span class="badge">● ${roleName} authenticated</span><button class="danger-btn" id="roleLogout" type="button">Logout</button></div>
      </header>
      <main class="dashboard-wrap">
        <section class="dashboard-heading"><div><div class="eyebrow">${roleName.toUpperCase()} PORTAL</div><h2>${dashboardTitle}</h2><p>${dashboardDescription}</p></div><button class="secondary" id="refreshDashboard">Refresh</button></section>
        <div id="dashboardContent"><article class="card panel">Loading live participant data…</article></div>
      </main>
    </div>`;
  document.getElementById('roleLogout').onclick = () => {
    clearInterval(dashboardTimer);
    const token = state.authToken;
    state.authToken = '';
    state.authRole = '';
    sessionStorage.removeItem('cognitrack_auth_token');
    sessionStorage.removeItem('cognitrack_auth_role');
    renderLogin();
    // Do not make navigation depend on the server response. The local session
    // is cleared immediately; revoke the server token in the background.
    safePost('/auth/logout', {}, token).catch(() => {});
  };
  document.getElementById('refreshDashboard').onclick = refreshDashboard;
  refreshDashboard();
  clearInterval(dashboardTimer);
  dashboardTimer = setInterval(refreshDashboard, 5000);
}

async function refreshDashboard() {
  const container = document.getElementById('dashboardContent');
  if (!container || !state.authToken) return;
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/overview`, {
      headers: { Authorization: `Bearer ${state.authToken}` }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    const summary = data.summary;
    const participants = data.participants || [];
    if (data.role === 'host') {
      container.innerHTML = `
      <section class="monitor-summary">
        <article class="card"><small>Total participants</small><strong>${summary.total_participants}</strong></article>
        <article class="card"><small>Active now</small><strong>${summary.active_participants}</strong></article>
        <article class="card"><small>Need attention</small><strong>${summary.participants_needing_attention}</strong></article>
        <article class="card"><small>Answers submitted</small><strong>${summary.answers_submitted}</strong></article>
      </section>
      <section class="monitor-layout">
        <article class="card panel monitor-table-card">
          <div class="section-label">Live Participant Activity</div>
          ${participants.length ? `<div class="monitor-table"><table><thead><tr><th>Participant ID</th><th>Current task</th><th>Status</th><th>Progress</th><th>Time in application</th><th>Attention</th></tr></thead><tbody>${participants.map(renderHostParticipantRow).join('')}</tbody></table></div>` : '<p>No participant has started an assessment yet.</p>'}
        </article>
      </section>`;
    } else {
      container.innerHTML = renderAdminAnalytics(data);
      container.querySelectorAll('[data-admin-export]').forEach(button => {
        button.onclick = () => downloadTrackingExport(button.dataset.adminExport);
      });
      container.querySelectorAll('[data-admin-dataset]').forEach(button => {
        button.onclick = () => openAdminDataPage(button.dataset.adminDataset);
      });
      const participantGraphButton = container.querySelector('[data-toggle-participants]');
      if (participantGraphButton) {
        participantGraphButton.onclick = () => {
          const graph = container.querySelector('#participantCountGraph');
          const hidden = graph.hasAttribute('hidden');
          graph.toggleAttribute('hidden', !hidden);
          participantGraphButton.textContent = hidden ? 'Hide participant graph' : 'Show participant graph';
        };
      }
      const storedResultsButton = container.querySelector('[data-toggle-results]');
      if (storedResultsButton) {
        storedResultsButton.onclick = () => {
          const table = container.querySelector('#storedParticipantTable');
          const hidden = table.hasAttribute('hidden');
          table.toggleAttribute('hidden', !hidden);
          storedResultsButton.textContent = hidden ? 'Hide participant details' : 'View participant details';
        };
      }
    }
  } catch (error) {
    container.innerHTML = `<article class="card panel error">Dashboard unavailable: ${escapeHtml(error.message)}</article>`;
  }
}

function renderAdminAnalytics(data) {
  const summary = data.summary;
  return `<section class="monitor-summary admin-summary">
      <article class="card"><small>Stored participants</small><strong>${summary.total_participants}</strong></article>
      <article class="card"><small>Submitted answers</small><strong>${summary.answers_submitted}</strong></article>
      <article class="card"><small>Completed sessions</small><strong>${summary.completed_sessions}</strong></article>
      <article class="card"><small>LLM chat messages</small><strong>${summary.chat_messages}</strong></article>
    </section>
    <section class="admin-chart-grid">
      <article class="card panel"><div class="section-label">Tracking Data Coverage</div>${renderBarChart(data.analytics?.component_totals || {})}</article>
      <article class="card panel"><div class="section-label">LLM Usage Comparison</div>${renderBarChart(data.analytics?.llm_usage || {})}</article>
      <article class="card panel participant-graph-card">
        <div class="section-label">Participant Count</div>
        <p>View the number of stored users in the assessment database.</p>
        <button class="secondary" data-toggle-participants>Show participant graph</button>
        <div id="participantCountGraph" hidden>${renderBarChart({'Stored users': summary.total_participants})}</div>
      </article>
    </section>
    <article class="card panel admin-export-panel">
      <div class="section-label">Tracking Tables</div>
      <p>Each table shows only the fields relevant to that tracking component.</p>
      <div class="admin-export-buttons">
        <button class="secondary" data-admin-dataset="eye-tracking">Eye Tracking Table</button>
        <button class="secondary" data-admin-dataset="facial-expression">Facial Expression Table</button>
        <button class="secondary" data-admin-dataset="keyboard">Keyboard Table</button>
        <button class="secondary" data-admin-dataset="mouse">Mouse/Cursor Table</button>
        <button class="secondary" data-admin-dataset="prompt-tracking">Prompt Tracking Table</button>
      </div>
      <div class="section-label">Download Tracking CSV Files</div>
      <p>Export tracking records from PostgreSQL for analysis.</p>
      <div class="admin-export-buttons">
        <button class="secondary" data-admin-export="mouse">Download Mouse Tracking CSV</button>
        <button class="secondary" data-admin-export="keyboard">Download Keyboard Tracking CSV</button>
        <button class="secondary" data-admin-export="eye">Download Eye Tracking CSV</button>
        <button class="secondary" data-admin-export="facial">Download Facial Expression CSV</button>
        <button class="secondary" data-admin-export="prompt">Download Prompt Tracking CSV</button>
      </div>
    </article>
    `;
}

async function downloadTrackingExport(exportName) {
  try {
    const response = await fetch(`${API_BASE_URL}/admin/exports/${encodeURIComponent(exportName)}.csv`, {
      headers: { Authorization: `Bearer ${state.authToken}` }
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = `cognitrack-${exportName}-tracking.csv`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(`Tracking export failed: ${error.message}`);
  }
}

async function openAdminDataPage(dataset) {
  const container = document.getElementById('dashboardContent');
  if (!container || !state.authToken) return;
  clearInterval(dashboardTimer);
  container.innerHTML = '<article class="card panel">Loading administrator records…</article>';
  try {
    const response = await fetch(`${API_BASE_URL}/admin/data/${encodeURIComponent(dataset)}`, {
      headers: { Authorization: `Bearer ${state.authToken}` }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    const records = data.records || [];
    container.innerHTML = `<article class="card panel admin-detail-page">
      <div class="detail-page-heading"><div><div class="section-label">ADMINISTRATOR DATA</div><h3>${escapeHtml(data.title || 'Details')}</h3><p>${records.length} record(s) shown. Each page loads its own protected dataset.</p></div><button class="secondary" id="backToAdminDashboard">Back to dashboard</button></div>
      ${renderAdminRecordTable(records, dataset)}
    </article>`;
    document.getElementById('backToAdminDashboard').onclick = () => renderRoleDashboard('admin');
  } catch (error) {
    container.innerHTML = `<article class="card panel error">Unable to load administrator data: ${escapeHtml(error.message)}<br><br><button class="secondary" id="backToAdminDashboard">Back to dashboard</button></article>`;
    document.getElementById('backToAdminDashboard').onclick = () => renderRoleDashboard('admin');
  }
}

function renderAdminRecordTable(records, dataset = '') {
  if (!records.length) return '<p>No records have been saved for this category yet.</p>';
  const columns = [...new Set(records.flatMap(record => Object.keys(record)))];
  const label = column => column.replaceAll('_', ' ');
  return `<div class="monitor-table admin-record-table"><table><thead><tr>${columns.map(column => `<th>${escapeHtml(label(column))}</th>`).join('')}</tr></thead><tbody>${records.map(record => `<tr>${columns.map(column => `<td>${escapeHtml(formatAdminValue(record[column]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function formatAdminValue(value) {
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function renderBarChart(values) {
  const entries = Object.entries(values);
  if (!entries.length) return '<p>No stored data yet.</p>';
  const maximum = Math.max(1, ...entries.map(([, value]) => Number(value)));
  return `<div class="bar-chart">${entries.map(([label,value]) => `<div class="bar-row"><span>${escapeHtml(label)}</span><div><i style="width:${Math.max(2, Number(value) / maximum * 100)}%"></i></div><strong>${Number(value)}</strong></div>`).join('')}</div>`;
}

function renderStoredResultRow(item) {
  const total = Math.max(0, Number(item.time_spent_seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = Math.floor(total % 60);
  const duration = `${hours ? `${String(hours).padStart(2,'0')}:` : ''}${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}`;
  return `<tr><td><strong>${escapeHtml(String(item.participant_id))}</strong></td><td>${escapeHtml(String(item.full_name || 'Unknown participant'))}</td><td><strong>${duration}</strong></td></tr>`;
}

function formatDuration(totalSeconds) {
  const durationSeconds = Math.max(0, Number(totalSeconds || 0));
  const hours = Math.floor(durationSeconds / 3600);
  const minutes = Math.floor((durationSeconds % 3600) / 60);
  const seconds = durationSeconds % 60;
  return `${hours ? `${String(hours).padStart(2,'0')}:` : ''}${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}`;
}

function renderHostParticipantRow(item) {
  const attention = item.watcher_state === 'attention' ? 'Needs review' : 'Normal';
  return `<tr>
    <td><strong>${escapeHtml(String(item.participant_id))}</strong></td>
    <td>${escapeHtml(String(item.question_id || 'Not started'))}</td>
    <td><span class="monitor-state">${escapeHtml(String(item.status))}</span></td>
    <td>${Number(item.progress_percent || 0).toFixed(1)}%</td>
    <td><strong>${formatDuration(item.session_duration_seconds)}</strong></td>
    <td><span class="monitor-state">${attention}</span></td>
  </tr>`;
}

async function restoreAuthenticatedRole() {
  const token = sessionStorage.getItem('cognitrack_auth_token') || '';
  const expectedRole = sessionStorage.getItem('cognitrack_auth_role') || '';
  if (!token || !expectedRole) return false;
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || result.role !== expectedRole) throw new Error('Invalid session');
    state.authToken = token;
    state.authRole = result.role;
    renderRoleDashboard(result.role);
    return true;
  } catch (_error) {
    sessionStorage.removeItem('cognitrack_auth_token');
    sessionStorage.removeItem('cognitrack_auth_role');
    return false;
  }
}

async function loadLlmProviderStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
    const result = await response.json().catch(() => ({}));
    if (response.ok && result.llm_providers) state.llmProviders = result.llm_providers;
  } catch (_error) {
    // Keep all choices visible if health status cannot be loaded; the API will
    // still return a specific provider error when a request is attempted.
    state.llmProviders = null;
  }
}

restoreAuthenticatedRole().then(async restored => {
  if (restored) return;
  await loadLlmProviderStatus();
  if (restoreParticipantState()) {
    await resumeParticipantState();
  } else {
    renderLogin();
  }
});
