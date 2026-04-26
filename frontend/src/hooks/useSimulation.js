import { useCallback, useRef, useEffect } from 'react';
import { useSimulationState, useSimulationDispatch } from '../store/simulationStore';
import { PRIMARY_SCENARIO, AGENT_ROLES, M2M_TRANSLATIONS } from '../data/scenarios';

const LOCAL_HTTP_BASE = 'http://localhost:8000';
const LOCAL_WS_BASE = 'ws://localhost:8000/ws';

export function getApiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) return configured.replace(/\/$/, '');
  if (typeof window === 'undefined') return LOCAL_HTTP_BASE;
  const host = window.location.hostname;
  const localHost = host === 'localhost' || host === '127.0.0.1';
  return localHost ? LOCAL_HTTP_BASE : window.location.origin;
}

function getWebSocketUrl() {
  const configured = import.meta.env.VITE_WS_URL?.trim();
  if (configured) return configured;
  if (typeof window === 'undefined') return LOCAL_WS_BASE;
  const host = window.location.hostname;
  const localHost = host === 'localhost' || host === '127.0.0.1';
  if (localHost) return LOCAL_WS_BASE;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws`;
}

function normalizePromptText(prompt) {
  let text = (prompt || '').replace(/\r\n/g, '\n').trim();
  if (!text) return '';

  const noiseMarkers = [
    '\nAuto-Generated RCA',
    '\n202',
    '\nINFO:',
    '\nCommander\n',
    '\nManager\n',
    '\nDetective\n',
    '\nCoder\n',
    '\nFixes Applied',
    '\nEscalations',
    '\nResolution',
  ];

  let cutIdx = text.length;
  noiseMarkers.forEach((marker) => {
    const idx = text.indexOf(marker);
    if (idx !== -1) cutIdx = Math.min(cutIdx, idx);
  });
  text = text.slice(0, cutIdx).trim();

  for (let i = 0; i < 3; i += 1) {
    const wrapped = text.match(/^\[ORCHESTRATION_REQUEST\]\s*Parse intent:\s*"([\s\S]*)"\s*$/i);
    if (wrapped) {
      text = wrapped[1].trim();
      continue;
    }
    text = text.replace(/^\[ORCHESTRATION_REQUEST\]\s*Parse intent:\s*/i, '').trim();
    break;
  }

  return text.replace(/^["']+|["']+$/g, '').trim();
}

export function useSimulation() {
  const state = useSimulationState();
  if (typeof window !== 'undefined') {
    window.__mockModeRef = state.mockMode;
  }
  const dispatch = useSimulationDispatch();
  const timersRef = useRef([]);
  const tickRef = useRef(null);
  const stateRef = useRef(state);
  const apiBase = getApiBase();

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    if (!state.isRunning || state.scenarioComplete) {
      if (tickRef.current) {
        clearInterval(tickRef.current);
        tickRef.current = null;
      }
    }
  }, [state.isRunning, state.scenarioComplete]);

  const clearAllTimers = useCallback(() => {
    timersRef.current.forEach(t => clearTimeout(t));
    timersRef.current = [];
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  const orchestrate = useCallback((prompt, { customOnly = false } = {}) => {
    clearAllTimers();
    const cleanPrompt = normalizePromptText(prompt);

    dispatch({ type: 'START_SIMULATION' });
    
    // Add initial orchestration chat message
    dispatch({
      type: 'ADD_MESSAGE',
      payload: {
        id: Date.now().toString(),
        agent: 'MANAGER',
        m2m: `[ORCHESTRATION_REQUEST] Parse intent: "${cleanPrompt}"`,
        think: customOnly
          ? 'Running custom prompt against a single OpenEnv task.'
          : 'Sending prompt to the orchestrator so it can select the OpenEnv task, validator mode, and live evidence path.',
        timestamp: new Date().toISOString()
      }
    });

    fetch(`${apiBase}/api/orchestrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: cleanPrompt, custom_only: customOnly })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'error') {
          dispatch({
            type: 'ADD_MESSAGE',
            payload: {
              id: Date.now().toString(),
              agent: 'COMMANDER',
              m2m: '[ORCHESTRATION_ERROR] Provider unavailable',
              think: data.error || 'The backend could not start the live OpenEnv run.',
              timestamp: new Date().toISOString()
            }
          });
          return;
        }
        // Backend LLM parsed the constraints and launched the sandbox
        dispatch({
          type: 'ADD_MESSAGE',
          payload: {
            id: Date.now().toString(),
            agent: 'COMMANDER',
            m2m: `[JSON_PROVISION] ${JSON.stringify(data.commander_payload)}`,
            think: `OpenEnv orchestration armed for ${data.commander_payload?.title || 'incident response'}. Validator: ${data.commander_payload?.validator_runtime?.label || 'unknown'}.`,
            timestamp: new Date().toISOString()
          }
        });
    })
    .catch(err => console.warn("Backend unavailable:", err));

    // Start the tick clock
    const tickInterval = 1000;
    tickRef.current = setInterval(() => {
      dispatch({ type: 'TICK', payload: tickInterval });
    }, tickInterval);

    // Remove fake hardcoded PRIMARY_SCENARIO events. We rely entirely on WebSockets and API returns now!
    
  }, [apiBase, dispatch, clearAllTimers]);

  const start = useCallback(() => {
    orchestrate("Deploy Llama-3 with a strict 500MB constraint");
  }, [orchestrate]);

  const pause = useCallback(() => {
    dispatch({ type: 'PAUSE_SIMULATION' });
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, [dispatch]);

  const resume = useCallback(() => {
    dispatch({ type: 'RESUME_SIMULATION' });
    tickRef.current = setInterval(() => {
      dispatch({ type: 'TICK', payload: 1000 });
    }, 1000);
  }, [dispatch]);

  const stop = useCallback(() => {
    clearAllTimers();
    dispatch({ type: 'STOP_SIMULATION' });
  }, [dispatch, clearAllTimers]);

  const setSpeed = useCallback((speed) => {
    dispatch({ type: 'SET_SPEED', payload: speed });
  }, [dispatch]);

  const toggleMockMode = useCallback(() => {
    dispatch({ type: 'TOGGLE_MOCK_MODE' });
  }, [dispatch]);

  // WebSocket logic has been moved to SimulationProvider for global stability.
  // useSimulation now focuses only on control functions and local timers.

  return { start, pause, resume, stop, setSpeed, toggleMockMode, orchestrate };
}

function processEvent(event, dispatch) {
  switch (event.type) {
    case 'chat': {
      const agentInfo = AGENT_ROLES[event.payload.agent] || { id: event.payload.agent.toLowerCase(), name: event.payload.agent, emoji: '🤖', color: '#71717a' };

      // API Integration: Code Submission
      if (event.payload.m2m.startsWith('CODE_SUBMIT')) {
        const isMock = typeof window !== 'undefined' && window.__mockModeRef !== undefined ? window.__mockModeRef : true;
        fetch(`${getApiBase()}/api/code/submit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code: "# Live inference.py submission",
            filename: event.payload.m2m.split(' | ')[1].trim(),
            agent_role: event.payload.agent,
            mock_mode: isMock
          })
        }).catch(err => console.warn("Backend unavailable:", err));
      }

      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          agent: agentInfo,
          m2m: event.payload.m2m,
          english: M2M_TRANSLATIONS[event.payload.m2m] || event.payload.m2m,
          think: event.payload.think || null,
        },
      });
      break;
    }
    case 'telemetry':
      dispatch({ type: 'UPDATE_TELEMETRY', payload: event.payload });
      break;
    case 'causal':
      dispatch({ type: 'ADD_CAUSAL_NODE', payload: event.payload });
      break;
    case 'preflight':
      dispatch({ type: 'UPDATE_PREFLIGHT', payload: event.payload });
      break;
    case 'disagreement':
      dispatch({ type: 'SET_DISAGREEMENT', payload: event.payload });
      break;
    case 'git':
      // API Integration: RCA Generation
      if (event.payload.files && event.payload.files.includes('RCA_REPORT.md')) {
        fetch(`${getApiBase()}/api/rca`)
          .then(res => res.json())
          .then(data => dispatch({ type: 'SET_RCA_DOCUMENT', payload: data.rca }))
          .catch(() => console.warn("Using static RCA"));
      }
      dispatch({ type: 'ADD_GIT_COMMIT', payload: event.payload });
      break;
    case 'counterfactual':
      // API Integration: Counterfactual
      fetch(`${getApiBase()}/api/counterfactual`)
        .then(res => res.json())
        .then(data => {
            dispatch({ type: 'SET_COUNTERFACTUAL', payload: {
                actual: event.payload.actual,
                dead: {
                   time: data.time_elapsed_formatted || data.time_elapsed_seconds + 's',
                   cost: `$${data.projected_cost_usd.toFixed(2)}`,
                   sla: data.sla_breached ? 'BREACHED' : 'SAFE',
                   outcome: data.outcome
                }
            }});
        })
        .catch(() => dispatch({ type: 'SET_COUNTERFACTUAL', payload: event.payload }));
      break;
    case 'reward':
      dispatch({ type: 'ADD_REWARD', payload: event.payload });
      break;
    default:
      break;
  }
}
