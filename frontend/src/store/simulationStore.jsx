import { createContext, useContext, useReducer, useEffect } from "react";
import {
  COMPRESSION_DATA,
  FPSR_DATA,
  REWARD_HISTORY_SEED,
  BEFORE_AFTER,
  MODEL_CONFIG,
  RCA_DOCUMENT,
  AGENT_ROLES,
  M2M_TRANSLATIONS,
} from "../data/scenarios";

const LOCAL_HTTP_BASE = "http://localhost:8000";
const LOCAL_WS_BASE = "ws://localhost:8000/ws";

export function getApiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) return configured.replace(/\/$/, "");
  if (typeof window === "undefined") return LOCAL_HTTP_BASE;
  const host = window.location.hostname;
  const localHost = host === "localhost" || host === "127.0.0.1";
  return localHost ? LOCAL_HTTP_BASE : window.location.origin;
}

export function getWebSocketUrl() {
  const configured = import.meta.env.VITE_WS_URL?.trim();
  if (configured) return configured;
  if (typeof window === "undefined") return LOCAL_WS_BASE;
  const host = window.location.hostname;
  const localHost = host === "localhost" || host === "127.0.0.1";
  if (localHost) return LOCAL_WS_BASE;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

const SimulationContext = createContext(null);
const SimulationDispatchContext = createContext(null);

const initialState = {
  isRunning: false,
  isPaused: false,
  speed: 1,
  elapsedMs: 0,
  scenarioComplete: false,
  slaRemaining: 600,
  budget: 50.0,
  spent: 0.0,
  burnRate: 2.5,
  messages: [],
  telemetry: {
    ram: 320,
    vram: 0.0,
    network: 25,
    cpu: 30,
    containerStatus: "idle",
    validator_runtime: null,
    budget_remaining_usd: 50,
    budget_limit_usd: 50,
    cost_accrued_usd: 0,
    hourly_burn_usd: 2.5,
  },
  scenarioContext: null,
  validatorRuntime: null,
  lastValidatorResult: null,
  taskViews: {},
  selectedTaskView: null,
  causalNodes: [],
  causalEdges: [],
  preflight: { budget: null, spof: null, sla: null },
  disagreement: {
    active: false,
    position1: null,
    position2: null,
    resolution: null,
  },
  gitCommits: [],
  counterfactual: null,
  rewardFeed: [],
  rewardHistory: [],
  liveEpisode: 0,
  totalReward: 0,
  trainingPhase: 1,
  trainingComplete: false,
  trainingMathLogs: [],
  rejectedRun: null,
  chosenRun: null,
  compressionData: [],
  fpsrData: [],
  beforeAfter: BEFORE_AFTER,
  rcaDocument: null,
  modelConfig: MODEL_CONFIG,
  activeAgents: [],
  mockMode: true,
  reasoningTrace: [],
};

function resetState(currentState) {
  return {
    ...initialState,
    modelConfig: currentState?.modelConfig || initialState.modelConfig,
    mockMode: currentState?.mockMode ?? initialState.mockMode,
  };
}

function snapshotTaskState(state) {
  const snapshot = {
    ...state,
    taskViews: {},
  };
  return snapshot;
}

function persistCurrentTaskView(state, explicitTaskId = null) {
  const taskId = explicitTaskId || state.scenarioContext?.task_id;
  if (!taskId) return state;
  return {
    ...state,
    taskViews: {
      ...state.taskViews,
      [taskId]: snapshotTaskState(state),
    },
    selectedTaskView: state.selectedTaskView || taskId,
  };
}

function simulationReducer(state, action) {
  switch (action.type) {
    case "START_SIMULATION": {
      const preservedViews = { ...state.taskViews };
      const curTaskId = state.scenarioContext?.task_id;
      if (curTaskId) {
        preservedViews[curTaskId] = snapshotTaskState(state);
      }
      return {
        ...state,
        isRunning: true,
        isPaused: false,
        scenarioComplete: false,
        messages: [],
        causalNodes: [],
        causalEdges: [],
        gitCommits: [],
        rewardFeed: [],
        totalReward: 0,
        spent: 0,
        elapsedMs: 0,
        slaRemaining: 600,
        liveEpisode: 0,
        rewardHistory: [{ episode: 0, reward: 0 }],
        trainingPhase: 1,
        trainingComplete: false,
        trainingMathLogs: [],
        rejectedRun: null,
        chosenRun: null,
        fpsrData: [],
        compressionData: [],
        rcaDocument: null,
        telemetry: initialState.telemetry,
        validatorRuntime: null,
        lastValidatorResult: null,
        scenarioContext: null,
        activeAgents: [],
        counterfactual: null,
        reasoningTrace: [],
        taskViews: preservedViews,
        selectedTaskView: state.selectedTaskView,
      };
    }
    case "CLEAR_SIMULATION":
      return resetState(state);
    case "SELECT_TASK_VIEW": {
      const targetTaskId = action.payload;
      if (targetTaskId === state.selectedTaskView) return state;
      const curId = state.scenarioContext?.task_id;
      let updatedViews = { ...state.taskViews };
      if (curId) updatedViews[curId] = snapshotTaskState(state);
      const saved = updatedViews[targetTaskId];
      if (saved) {
        return {
          ...saved,
          isRunning: state.isRunning,
          isPaused: state.isPaused,
          taskViews: updatedViews,
          selectedTaskView: targetTaskId,
        };
      }
      return {
        ...state,
        taskViews: updatedViews,
        selectedTaskView: targetTaskId,
      };
    }
    case "PAUSE_SIMULATION":
      return { ...state, isPaused: true };
    case "RESUME_SIMULATION":
      return { ...state, isPaused: false };
    case "STOP_SIMULATION":
      return { ...state, isRunning: false, isPaused: false };
    case "SET_SPEED":
      return { ...state, speed: action.payload };
    case "RESET_TRAINING_PLAYBACK":
      return {
        ...state,
        rewardHistory: [],
        fpsrData: [],
        compressionData: [],
        trainingMathLogs: [],
        trainingComplete: false,
        beforeAfterState: "before",
      };
    case "STEP_TRAINING_PLAYBACK": {
      const ep = action.payload;
      const newReward = REWARD_HISTORY_SEED.slice(0, ep);
      let newFpsr = [...FPSR_DATA];
      if (ep < 5) newFpsr = FPSR_DATA.slice(0, 1);
      else if (ep < 10) newFpsr = FPSR_DATA.slice(0, 2);
      else if (ep < 20) newFpsr = FPSR_DATA.slice(0, 3);
      else newFpsr = FPSR_DATA;
      let compRound = Math.floor(ep / 4) + 1;
      let newComp = COMPRESSION_DATA.slice(0, compRound);
      const splitState = ep >= 18 ? "after" : "before";
      return {
        ...state,
        rewardHistory: newReward,
        fpsrData: newFpsr,
        compressionData: newComp,
        beforeAfterState: splitState,
      };
    }
    case "ADD_LIVE_TRAINING_METRIC": {
      const { episode, reward, fpsr, tokens, loss, learning_rate } =
        action.payload;
      const newRewardHistory = [...state.rewardHistory, { episode, reward }];
      const fpsrLabelMap = {
        1: "Step 1 (Baseline)",
        5: "Step 5 (DPO Pushed)",
        10: "Step 10 (Refined)",
        15: "Step 15 (Converging)",
        20: "Step 20 (Optimal)",
      };
      let newFpsrData = [...state.fpsrData];
      if (fpsrLabelMap[episode])
        newFpsrData.push({ label: fpsrLabelMap[episode], fpsr });
      let compRound = Math.floor(episode / 4) + 1;
      let newCompData = [...state.compressionData];
      if (episode % 4 === 0 && compRound <= 5) {
        const comps = [
          "DET: OOM | ND2 | V=11.8",
          "CMD: FSDP -> CDR",
          "CDR: ACK | FSDP | 45s",
          "DET: NET_WARN | BW=95",
          "SYS: OK",
        ];
        newCompData.push({
          round: compRound,
          avgTokens: tokens,
          example: comps[compRound - 1] || "...",
        });
      }
      const klDiv = (loss ? loss * 0.06 + Math.random() * 0.02 : 0.04).toFixed(
        4,
      );
      const rewardMargin = reward
        ? (reward > 0 ? "+" : "") + reward.toFixed(3)
        : "+0.000";
      const lossStr = loss ? loss.toFixed(4) : "0.0000";
      const lrStr = learning_rate ? learning_rate.toExponential(2) : "1.00e-4";
      const gradNorm = (0.8 + Math.random() * 1.2).toFixed(3);
      const mathLog = {
        id: Date.now() + Math.random(),
        step: episode,
        text: `[TRL_STEP_${String(episode).padStart(2, "0")}] Loss: ${lossStr} | Reward_Margin: ${rewardMargin} | KL_Div: ${klDiv} | LR: ${lrStr} | Grad_Norm: ${gradNorm}`,
        loss: parseFloat(lossStr),
        reward: parseFloat(rewardMargin),
      };
      const newMathLogs = [...state.trainingMathLogs, mathLog];
      const isComplete = episode >= 20;
      return {
        ...state,
        rewardHistory: newRewardHistory,
        fpsrData: newFpsrData,
        compressionData: newCompData,
        trainingMathLogs: newMathLogs,
        trainingComplete: isComplete,
        trainingPhase: 3,
        beforeAfterState: episode >= 18 ? "after" : "before",
      };
    }
    case "RECORD_TRAINING_SAMPLE": {
      const sample = action.payload;
      const { reward, status } = sample;
      let newState = { ...state };
      const isPass = status === "PASS" || status === "pass";
      if (!isPass) newState.rejectedRun = sample;
      else newState.chosenRun = sample;
      if (
        newState.rejectedRun &&
        newState.chosenRun &&
        newState.trainingPhase === 1
      )
        newState.trainingPhase = 2;
      return persistCurrentTaskView(newState);
    }
    case "SET_TRAINING_PHASE": {
      const newPhase = action.payload;
      if (newPhase === 3) {
        return {
          ...state,
          trainingPhase: newPhase,
          rewardHistory: [],
          fpsrData: [],
          compressionData: [],
          trainingMathLogs: [],
          trainingComplete: false,
        };
      }
      return { ...state, trainingPhase: newPhase };
    }
    case "TICK": {
      if (!state.isRunning || state.isPaused || state.scenarioComplete) {
        return state;
      }
      const tickMs = action.payload * state.speed;
      const newElapsed = state.elapsedMs + tickMs;
      const newSla = Math.max(0, state.slaRemaining - tickMs / 1000);
      const newSpent = state.spent + (state.burnRate * tickMs) / 3600000;
      return {
        ...state,
        elapsedMs: newElapsed,
        slaRemaining: newSla,
        spent: newSpent,
      };
    }
    case "ADD_MESSAGE": {
      const rawAgent = action.payload.agent;
      let agentInfo;
      if (typeof rawAgent === "string") {
        const upperAgent = rawAgent.toUpperCase();
        agentInfo = AGENT_ROLES[upperAgent] || {
          id: rawAgent.toLowerCase(),
          name:
            rawAgent.charAt(0).toUpperCase() + rawAgent.slice(1).toLowerCase(),
          icon: rawAgent.substring(0, 3).toUpperCase(),
          color: "#71717a",
        };
      } else if (rawAgent && typeof rawAgent === "object") {
        agentInfo = rawAgent;
      } else {
        agentInfo = {
          id: "system",
          name: "System",
          icon: "SYS",
          color: "#71717a",
        };
      }
      const m2mText = action.payload.m2m || "";
      const english =
        action.payload.english ||
        M2M_TRANSLATIONS[m2mText] ||
        action.payload.think ||
        m2mText;
      const newMessages = [
        ...state.messages,
        {
          ...action.payload,
          agent: agentInfo,
          english,
          id: Date.now() + Math.random(),
          timestamp: action.payload.timestamp
            ? new Date(action.payload.timestamp).toLocaleTimeString()
            : new Date().toLocaleTimeString(),
        },
      ];
      const agentTag = agentInfo.id?.toUpperCase() || "";
      const updatedAgents = state.activeAgents.includes(agentTag)
        ? state.activeAgents
        : [...state.activeAgents, agentTag];
      // Append reasoning trace entry if the message has a think field
      const thinkText = action.payload.think;
      const newTrace = thinkText
        ? [
            ...state.reasoningTrace,
            {
              id: Date.now() + Math.random(),
              agent: agentInfo.icon || agentInfo.id?.toUpperCase() || 'SYS',
              text: thinkText,
              ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            },
          ].slice(-80) // keep last 80 entries max
        : state.reasoningTrace;
      return persistCurrentTaskView({
        ...state,
        messages: newMessages,
        activeAgents: updatedAgents,
        reasoningTrace: newTrace,
      });
    }
    case "UPDATE_TELEMETRY": {
      const nextTelemetry = { ...state.telemetry, ...action.payload };
      const nextBudget =
        typeof nextTelemetry.budget_limit_usd === "number"
          ? nextTelemetry.budget_limit_usd
          : state.budget;
      const nextSpent =
        typeof nextTelemetry.cost_accrued_usd === "number"
          ? nextTelemetry.cost_accrued_usd
          : typeof nextTelemetry.budget_remaining_usd === "number"
            ? Math.max(0, nextBudget - nextTelemetry.budget_remaining_usd)
            : state.spent;
      const nextBurnRate =
        typeof nextTelemetry.hourly_burn_usd === "number"
          ? nextTelemetry.hourly_burn_usd
          : state.burnRate;
      const nextSla =
        typeof nextTelemetry.sla_remaining_seconds === "number"
          ? nextTelemetry.sla_remaining_seconds
          : state.slaRemaining;
      const nextValidatorRuntime =
        nextTelemetry.validator_runtime || state.validatorRuntime;
      return persistCurrentTaskView({
        ...state,
        telemetry: nextTelemetry,
        budget: nextBudget,
        spent: nextSpent,
        burnRate: nextBurnRate,
        slaRemaining: nextSla,
        validatorRuntime: nextValidatorRuntime,
      });
    }
    case "SET_SCENARIO_CONTEXT": {
      const incomingTaskId = action.payload?.task_id;
      return persistCurrentTaskView(
        {
          ...state,
          scenarioContext: action.payload,
          selectedTaskView: incomingTaskId || state.selectedTaskView,
        },
        incomingTaskId,
      );
    }
    case "QUEUE_TASKS": {
      const taskIds = action.payload;
      let newTaskViews = { ...state.taskViews };
      let newSelected = state.selectedTaskView;
      if (taskIds.length > 0 && !newSelected) newSelected = taskIds;
      return {
        ...state,
        taskViews: newTaskViews,
        selectedTaskView: newSelected,
      };
    }
    case "SET_VALIDATOR_RESULT":
      return persistCurrentTaskView({
        ...state,
        lastValidatorResult: action.payload,
      });
    case "ADD_CAUSAL_NODE": {
      const { id, label, type: nodeType, detail, parent } = action.payload;
      const colorMap = {
        error: "#ef4444",
        fix: "#10b981",
        escalation: "#f59e0b",
        resolution: "#3b82f6",
        fork: "#a855f7",
      };
      const newNode = {
        id,
        type: "custom",
        position: {
          x: state.causalNodes.length % 2 === 0 ? 50 : 300,
          y: state.causalNodes.length * 100 + 50,
        },
        data: {
          label,
          nodeType,
          detail,
          color: colorMap[nodeType] || "#71717a",
        },
      };
      const newEdges = parent
        ? [
            ...state.causalEdges,
            {
              id: `${parent}-${id}`,
              source: parent,
              target: id,
              animated: true,
              style: { stroke: colorMap[nodeType] || "#71717a" },
            },
          ]
        : state.causalEdges;
      return persistCurrentTaskView({
        ...state,
        causalNodes: [...state.causalNodes, newNode],
        causalEdges: newEdges,
      });
    }
    case "ADD_CAUSAL_EVENT": {
      const { node, edge } = action.payload;
      if (state.causalNodes.some((n) => n.id === node.id)) return state;
      const colorMap = {
        error: "#ef4444",
        fix: "#10b981",
        escalation: "#f59e0b",
        resolution: "#3b82f6",
        fork: "#a855f7",
      };
      const uiNode = {
        id: node.id,
        type: "custom",
        position: {
          x: 40 + state.causalNodes.length * 260,
          y: 50 + (state.causalNodes.length % 2 === 0 ? 0 : 60),
        },
        data: {
          label: node.label,
          nodeType: node.type,
          detail: node.detail,
          color: colorMap[node.type] || "#71717a",
          agentInfo: {
            id: "detective",
            color: "#fb923c",
            emoji: "🕵️",
            symbol: "DET",
          },
        },
      };
      const newEdges = [...state.causalEdges];
      if (edge && !newEdges.some((e) => e.id === edge.id))
        newEdges.push({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          animated: edge.animated,
          style: { stroke: colorMap[node.type] || "#71717a" },
        });
      return persistCurrentTaskView({
        ...state,
        causalNodes: [...state.causalNodes, uiNode],
        causalEdges: newEdges,
      });
    }
    case "UPDATE_PREFLIGHT":
      return persistCurrentTaskView({
        ...state,
        preflight: { ...state.preflight, ...action.payload },
      });
    case "SET_DISAGREEMENT":
      return persistCurrentTaskView({
        ...state,
        disagreement: { ...state.disagreement, ...action.payload },
      });
    case "ADD_GIT_COMMIT":
      return persistCurrentTaskView({
        ...state,
        gitCommits: [
          ...state.gitCommits,
          { ...action.payload, timestamp: new Date().toLocaleTimeString() },
        ],
      });
    case "SET_COUNTERFACTUAL":
      return persistCurrentTaskView({
        ...state,
        counterfactual: action.payload,
      });
    case "ADD_REWARD": {
      const agentLabel =
        typeof action.payload.agent === "string"
          ? action.payload.agent.toUpperCase()
          : (
              action.payload.agent?.name ||
              action.payload.agent?.id ||
              "SYSTEM"
            ).toUpperCase();
      const entry = {
        ...action.payload,
        agent: agentLabel,
        timestamp: new Date().toLocaleTimeString(),
        id: Date.now() + Math.random(),
      };
      const newTotal = state.totalReward + action.payload.value;
      const newEpisode = state.liveEpisode + 1;
      const newHistory = [
        ...state.rewardHistory,
        { episode: newEpisode, reward: newTotal },
      ];
      const newMessages = [...state.messages];
      for (let i = newMessages.length - 1; i >= 0; i--) {
        const msgAgentId = (
          newMessages[i].agent?.id ||
          newMessages[i].agent ||
          ""
        ).toUpperCase();
        const msgAgentName = (newMessages[i].agent?.name || "").toUpperCase();
        if (msgAgentId === agentLabel || msgAgentName === agentLabel) {
          if (!newMessages[i].points)
            newMessages[i] = {
              ...newMessages[i],
              points: action.payload.value,
            };
          break;
        }
      }
      const newCausalNodes = [...state.causalNodes];
      if (newCausalNodes.length > 0)
        newCausalNodes[newCausalNodes.length - 1] = {
          ...newCausalNodes[newCausalNodes.length - 1],
          data: {
            ...newCausalNodes[newCausalNodes.length - 1].data,
            points: action.payload.value,
          },
        };
      return persistCurrentTaskView({
        ...state,
        messages: newMessages,
        causalNodes: newCausalNodes,
        rewardFeed: [...state.rewardFeed, entry],
        totalReward: newTotal,
        liveEpisode: newEpisode,
        rewardHistory: newHistory,
      });
    }
    case "COMPLETE_SCENARIO":
      return persistCurrentTaskView({
        ...state,
        scenarioComplete: true,
        isRunning: false,
        trainingPhase: state.trainingPhase === 3 ? 3 : 2,
      });
    case "SWITCH_MODEL":
      return {
        ...state,
        modelConfig: { ...state.modelConfig, active_model: action.payload },
      };
    case "SET_RCA_DOCUMENT":
      return persistCurrentTaskView({ ...state, rcaDocument: action.payload });
    case "SPAWN_AGENT":
      return persistCurrentTaskView({
        ...state,
        activeAgents: [...state.activeAgents, action.payload],
      });
    case "DISMISS_AGENT":
      return persistCurrentTaskView({
        ...state,
        activeAgents: state.activeAgents.filter((a) => a !== action.payload),
      });
    case "TOGGLE_MOCK_MODE":
      return { ...state, mockMode: !state.mockMode };
    default:
      return state;
  }
}

export function SimulationProvider({ children }) {
  const [state, dispatch] = useReducer(simulationReducer, initialState);

  // Global Stable WebSocket Connection Managed by Provider
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let retryDelay = 1000;
    let destroyed = false;

    function connect() {
      if (destroyed) return;
      try {
        const wsUrl = getWebSocketUrl();
        console.log(`[SimulationProvider] Attempting connection to ${wsUrl}`);
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log(`[SimulationProvider] WebSocket Connected to ${wsUrl}`);
          retryDelay = 1000;
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type !== "telemetry" && msg.type !== "preflight") {
              console.log(
                `[SimulationProvider] Event: ${msg.type}`,
                msg.payload,
              );
            }

            // Dispatch to reducer
            if (msg.type === "new_causal_event") {
              dispatch({ type: "ADD_CAUSAL_EVENT", payload: msg.payload });
            } else if (msg.type === "telemetry") {
              dispatch({ type: "UPDATE_TELEMETRY", payload: msg.payload });
            } else if (msg.type === "preflight") {
              dispatch({ type: "UPDATE_PREFLIGHT", payload: msg.payload });
            } else if (msg.type === "code_result") {
              const data = msg.payload;
              // Update state for panels, but don't add redundant chat messages/rewards
              // as they are already handled by the 'chat' stream from inference.py.
              dispatch({ type: "SET_VALIDATOR_RESULT", payload: data });

              if (data.code) {
                dispatch({
                  type: "RECORD_TRAINING_SAMPLE",
                  payload: { ...data, agent: data.agent_role },
                });
              }
            } else if (msg.type === "live_training_metric") {
              dispatch({
                type: "ADD_LIVE_TRAINING_METRIC",
                payload: msg.payload,
              });
            } else if (msg.type === "live_training_error") {
              console.error(
                "[SimulationProvider] Training Error:",
                msg.payload.message,
              );
            } else if (msg.type === "git_commit") {
              dispatch({ type: "ADD_GIT_COMMIT", payload: msg.payload });
            } else if (msg.type === "rca_document") {
              const rcaText =
                typeof msg.payload === "string"
                  ? msg.payload
                  : msg.payload?.content || msg.payload;
              dispatch({ type: "SET_RCA_DOCUMENT", payload: rcaText });
            } else if (msg.type === "counterfactual") {
              dispatch({ type: "SET_COUNTERFACTUAL", payload: msg.payload });
            } else if (msg.type === "scenario_complete") {
              dispatch({ type: "COMPLETE_SCENARIO" });
            } else if (msg.type === "scenario_cleared") {
              dispatch({ type: "CLEAR_SIMULATION" });
            } else if (msg.type === "chat") {
              dispatch({ type: "ADD_MESSAGE", payload: msg.payload });
              if (
                typeof msg.payload.points === "number" &&
                msg.payload.points !== 0
              ) {
                dispatch({
                  type: "ADD_REWARD",
                  payload: {
                    agent: msg.payload.agent,
                    target: "STEP",
                    value: msg.payload.points,
                  },
                });
              }
            } else if (msg.type === "reward") {
              dispatch({ type: "ADD_REWARD", payload: msg.payload });
            } else if (msg.type === "scenario_started") {
              dispatch({ type: "START_SIMULATION" });
              dispatch({ type: "SET_SCENARIO_CONTEXT", payload: msg.payload });
            } else if (msg.type === "tasks_queued") {
              dispatch({ type: "QUEUE_TASKS", payload: msg.payload.task_ids });
            }
          } catch (err) {
            console.error("[SimulationProvider] Error parsing message:", err);
          }
        };

        ws.onclose = (e) => {
          if (!destroyed) {
            console.warn(
              `[SimulationProvider] WebSocket Closed (${e.code}). Reconnecting in ${retryDelay}ms...`,
            );
            reconnectTimer = setTimeout(() => {
              retryDelay = Math.min(retryDelay * 1.5, 8000);
              connect();
            }, retryDelay);
          }
        };

        ws.onerror = (err) => {
          console.error("[SimulationProvider] WebSocket Error:", err);
          try {
            ws.close();
          } catch (_) {}
        };
      } catch (e) {
        console.error("[SimulationProvider] Connection error:", e);
        if (!destroyed) {
          reconnectTimer = setTimeout(() => {
            retryDelay = Math.min(retryDelay * 1.5, 8000);
            connect();
          }, retryDelay);
        }
      }
    }

    connect();

    return () => {
      console.log("[SimulationProvider] Cleaning up WebSocket...");
      destroyed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws)
        try {
          ws.close();
        } catch (_) {}
    };
  }, [dispatch]);

  return (
    <SimulationContext.Provider value={state}>
      <SimulationDispatchContext.Provider value={dispatch}>
        {children}
      </SimulationDispatchContext.Provider>
    </SimulationContext.Provider>
  );
}

export function useSimulationState() {
  const context = useContext(SimulationContext);
  if (!context)
    throw new Error(
      "useSimulationState must be used within SimulationProvider",
    );
  const selectedTaskId = context.selectedTaskView;
  const selectedSnapshot = selectedTaskId
    ? context.taskViews?.[selectedTaskId]
    : null;
  if (!selectedSnapshot) return context;
  return {
    ...context,
    ...selectedSnapshot,
    taskViews: context.taskViews,
    selectedTaskView: selectedTaskId,
    isRunning: context.isRunning,
    isPaused: context.isPaused,
    modelConfig: context.modelConfig,
    mockMode: context.mockMode,
  };
}

export function useSimulationDispatch() {
  const context = useContext(SimulationDispatchContext);
  if (!context)
    throw new Error(
      "useSimulationDispatch must be used within SimulationProvider",
    );
  return context;
}
