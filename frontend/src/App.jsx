import {
  Activity,
  AlertTriangle,
  Check,
  ClipboardCheck,
  Gauge,
  History,
  RefreshCcw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const initialVehicle = {
  brand: "Ford",
  model: "Utility Police Interceptor Base",
  model_year: 2013,
  milage: "51,000 mi.",
  fuel_type: "E85 Flex Fuel",
  engine: "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability",
  transmission: "6-Speed A/T",
  ext_col: "Black",
  int_col: "Black",
  accident: "At least 1 accident or damage reported",
  clean_title: "Yes",
};

const cleanVehicle = {
  brand: "Lexus",
  model: "RX 350 RX 350",
  model_year: 2022,
  milage: "22,372 mi.",
  fuel_type: "Gasoline",
  engine: "3.5 Liter DOHC",
  transmission: "Automatic",
  ext_col: "Blue",
  int_col: "Black",
  accident: "None reported",
  clean_title: "Yes",
};

const emptyDecision = {
  final_offer: "",
  reason: "",
  user_id: "pricing_user_01",
};

function currency(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function Badge({ tone = "neutral", children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Field({ label, children, wide = false }) {
  return (
    <label className={`field ${wide ? "field-wide" : ""}`}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function App() {
  const [vehicle, setVehicle] = useState(initialVehicle);
  const [pricingContext, setPricingContext] = useState({
    reconditioning_cost: 1200,
    target_margin_pct: 0.12,
  });
  const [recommendation, setRecommendation] = useState(null);
  const [decisionForm, setDecisionForm] = useState(emptyDecision);
  const [decisions, setDecisions] = useState([]);
  const [modelHealth, setModelHealth] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("workbench");

  const riskTone = useMemo(() => {
    if (!recommendation) return "neutral";
    if (recommendation.risk_level === "High") return "danger";
    if (recommendation.risk_level === "Medium") return "warning";
    return "success";
  }, [recommendation]);

  useEffect(() => {
    refreshDecisions();
    refreshModelHealth();
  }, []);

  function updateVehicle(field, value) {
    setVehicle((current) => ({
      ...current,
      [field]: field === "model_year" ? Number(value) : value,
    }));
  }

  function updatePricingContext(field, value) {
    setPricingContext((current) => ({
      ...current,
      [field]: Number(value),
    }));
  }

  async function requestJson(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `Request failed with status ${response.status}`);
    }
    return response.json();
  }

  async function getRecommendation(event) {
    event?.preventDefault();
    setStatus("Generating recommendation");
    setError("");

    try {
      const result = await requestJson("/api/v1/recommendations", {
        method: "POST",
        body: JSON.stringify({ vehicle, pricing_context: pricingContext }),
      });
      setRecommendation(result);
      setDecisionForm({
        ...emptyDecision,
        final_offer: result.recommended_offer,
      });
      setActiveView("workbench");
      setStatus("Recommendation ready");
    } catch (err) {
      setError(err.message);
      setStatus("Backend request failed");
    }
  }

  async function submitDecision(decision) {
    if (!recommendation) return;
    setStatus(`Recording ${decision}`);
    setError("");

    try {
      await requestJson("/api/v1/pricing-decisions", {
        method: "POST",
        body: JSON.stringify({
          quote_id: recommendation.quote_id,
          user_id: decisionForm.user_id,
          decision,
          recommended_offer: recommendation.recommended_offer,
          final_offer: Number(decisionForm.final_offer || recommendation.recommended_offer),
          reason: decisionForm.reason || `${decision} from pricing console`,
        }),
      });
      setStatus("Decision recorded");
      await refreshDecisions();
      setActiveView("history");
    } catch (err) {
      setError(err.message);
      setStatus("Decision request failed");
    }
  }

  async function refreshDecisions() {
    try {
      const result = await requestJson("/api/v1/pricing-decisions");
      setDecisions(result);
    } catch {
      setDecisions([]);
    }
  }

  async function refreshModelHealth() {
    try {
      const result = await requestJson("/api/v1/model-health");
      setModelHealth(result);
    } catch {
      setModelHealth(null);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">PW</div>
          <div>
            <h1>PriceWise</h1>
            <p>C2B Pricing Console</p>
          </div>
        </div>

        <nav className="nav-list" aria-label="Pricing console navigation">
          <button
            className={activeView === "workbench" ? "active" : ""}
            onClick={() => setActiveView("workbench")}
          >
            <ClipboardCheck size={18} /> Workbench
          </button>
          <button
            className={activeView === "scenario" ? "active" : ""}
            onClick={() => setActiveView("scenario")}
          >
            <SlidersHorizontal size={18} /> Simulator
          </button>
          <button
            className={activeView === "history" ? "active" : ""}
            onClick={() => {
              refreshDecisions();
              setActiveView("history");
            }}
          >
            <History size={18} /> Decisions
          </button>
          <button
            className={activeView === "health" ? "active" : ""}
            onClick={() => {
              refreshModelHealth();
              setActiveView("health");
            }}
          >
            <Activity size={18} /> Model Health
          </button>
        </nav>

        <div className="system-card">
          <div className="system-row">
            <span>API</span>
            <Badge tone={error ? "danger" : "success"}>{error ? "Check" : "Online"}</Badge>
          </div>
          <p>{status}</p>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Internal pricing operations</p>
            <h2>{viewTitle(activeView)}</h2>
          </div>
          <div className="topbar-actions">
            <button className="ghost-button" onClick={() => setVehicle(initialVehicle)}>
              <RefreshCcw size={16} /> Risk sample
            </button>
            <button className="ghost-button" onClick={() => setVehicle(cleanVehicle)}>
              <ShieldCheck size={16} /> Clean sample
            </button>
          </div>
        </header>

        {error && (
          <div className="alert-row">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {activeView === "workbench" && (
          <section className="content-grid">
            <VehicleForm
              vehicle={vehicle}
              pricingContext={pricingContext}
              updateVehicle={updateVehicle}
              updatePricingContext={updatePricingContext}
              getRecommendation={getRecommendation}
            />
            <RecommendationPanel
              recommendation={recommendation}
              riskTone={riskTone}
              decisionForm={decisionForm}
              setDecisionForm={setDecisionForm}
              submitDecision={submitDecision}
            />
          </section>
        )}

        {activeView === "scenario" && (
          <ScenarioSimulator
            pricingContext={pricingContext}
            updatePricingContext={updatePricingContext}
            getRecommendation={getRecommendation}
            recommendation={recommendation}
          />
        )}

        {activeView === "history" && (
          <DecisionHistory decisions={decisions} refreshDecisions={refreshDecisions} />
        )}

        {activeView === "health" && <ModelHealth modelHealth={modelHealth} />}
      </section>
    </main>
  );
}

function viewTitle(view) {
  const labels = {
    workbench: "Quote Workbench",
    scenario: "Scenario Simulator",
    history: "Decision History",
    health: "Model Health",
  };
  return labels[view] || "Quote Workbench";
}

function VehicleForm({
  vehicle,
  pricingContext,
  updateVehicle,
  updatePricingContext,
  getRecommendation,
}) {
  return (
    <form className="panel form-panel" onSubmit={getRecommendation}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">Vehicle intake</p>
          <h3>Offer request</h3>
        </div>
        <Search size={20} />
      </div>

      <div className="form-grid">
        <Field label="Brand">
          <input value={vehicle.brand} onChange={(event) => updateVehicle("brand", event.target.value)} />
        </Field>
        <Field label="Model" wide>
          <input value={vehicle.model} onChange={(event) => updateVehicle("model", event.target.value)} />
        </Field>
        <Field label="Model year">
          <input
            type="number"
            value={vehicle.model_year}
            onChange={(event) => updateVehicle("model_year", event.target.value)}
          />
        </Field>
        <Field label="Mileage">
          <input value={vehicle.milage} onChange={(event) => updateVehicle("milage", event.target.value)} />
        </Field>
        <Field label="Fuel type">
          <select value={vehicle.fuel_type} onChange={(event) => updateVehicle("fuel_type", event.target.value)}>
            <option>Gasoline</option>
            <option>Hybrid</option>
            <option>Diesel</option>
            <option>E85 Flex Fuel</option>
            <option>Plug-In Hybrid</option>
          </select>
        </Field>
        <Field label="Transmission">
          <input
            value={vehicle.transmission}
            onChange={(event) => updateVehicle("transmission", event.target.value)}
          />
        </Field>
        <Field label="Engine" wide>
          <input value={vehicle.engine} onChange={(event) => updateVehicle("engine", event.target.value)} />
        </Field>
        <Field label="Exterior color">
          <input value={vehicle.ext_col} onChange={(event) => updateVehicle("ext_col", event.target.value)} />
        </Field>
        <Field label="Interior color">
          <input value={vehicle.int_col} onChange={(event) => updateVehicle("int_col", event.target.value)} />
        </Field>
        <Field label="Accident history" wide>
          <select value={vehicle.accident} onChange={(event) => updateVehicle("accident", event.target.value)}>
            <option>None reported</option>
            <option>At least 1 accident or damage reported</option>
          </select>
        </Field>
        <Field label="Clean title">
          <select value={vehicle.clean_title} onChange={(event) => updateVehicle("clean_title", event.target.value)}>
            <option>Yes</option>
            <option>No</option>
          </select>
        </Field>
        <Field label="Reconditioning cost">
          <input
            type="number"
            value={pricingContext.reconditioning_cost}
            onChange={(event) => updatePricingContext("reconditioning_cost", event.target.value)}
          />
        </Field>
        <Field label="Target margin">
          <input
            type="number"
            min="0"
            max="0.5"
            step="0.01"
            value={pricingContext.target_margin_pct}
            onChange={(event) => updatePricingContext("target_margin_pct", event.target.value)}
          />
        </Field>
      </div>

      <button className="primary-button" type="submit">
        <Gauge size={18} /> Get Recommendation
      </button>
    </form>
  );
}

function RecommendationPanel({
  recommendation,
  riskTone,
  decisionForm,
  setDecisionForm,
  submitDecision,
}) {
  if (!recommendation) {
    return (
      <section className="panel empty-panel">
        <Gauge size={36} />
        <h3>No recommendation yet</h3>
        <p>Submit a vehicle request to generate market value, offer guidance, approval status, and pricing rationale.</p>
      </section>
    );
  }

  return (
    <section className="panel recommendation-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Quote {recommendation.quote_id}</p>
          <h3>{recommendation.recommended_action}</h3>
        </div>
        <Badge tone={riskTone}>{recommendation.risk_level} Risk</Badge>
      </div>

      <div className="metric-grid">
        <Metric label="Market value" value={currency(recommendation.market_value)} />
        <Metric label="Recommended offer" value={currency(recommendation.recommended_offer)} emphasis />
        <Metric label="Expected margin" value={currency(recommendation.expected_margin)} />
        <Metric label="Margin rate" value={percent(recommendation.expected_margin_pct)} />
      </div>

      <div className="approval-row">
        <div>
          <span>Approval status</span>
          <strong>{recommendation.approval_required ? "Manager review required" : "Auto-approval allowed"}</strong>
        </div>
        <Badge tone={recommendation.approval_required ? "warning" : "success"}>
          {recommendation.approval_required ? "Review" : "Clear"}
        </Badge>
      </div>

      <div className="section-block">
        <h4>Pricing rationale</h4>
        <ul className="plain-list">
          {recommendation.explanation.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <div className="section-block">
        <h4>Candidate offers</h4>
        <div className="candidate-list">
          {recommendation.candidate_offers.map((offer) => (
            <button
              type="button"
              className="candidate-card"
              key={offer.discount_pct}
              onClick={() =>
                setDecisionForm((current) => ({
                  ...current,
                  final_offer: offer.offer,
                }))
              }
            >
              <span>{percent(offer.discount_pct)} discount</span>
              <strong>{currency(offer.offer)}</strong>
              <small>{percent(offer.expected_margin_pct)} margin</small>
            </button>
          ))}
        </div>
      </div>

      <div className="decision-box">
        <Field label="User ID">
          <input
            value={decisionForm.user_id}
            onChange={(event) =>
              setDecisionForm((current) => ({ ...current, user_id: event.target.value }))
            }
          />
        </Field>
        <Field label="Final offer">
          <input
            type="number"
            value={decisionForm.final_offer}
            onChange={(event) =>
              setDecisionForm((current) => ({ ...current, final_offer: event.target.value }))
            }
          />
        </Field>
        <Field label="Decision reason">
          <textarea
            rows="3"
            value={decisionForm.reason}
            onChange={(event) =>
              setDecisionForm((current) => ({ ...current, reason: event.target.value }))
            }
            placeholder="Add reason for acceptance, rejection, or override"
          />
        </Field>
        <div className="decision-actions">
          <button type="button" className="success-button" onClick={() => submitDecision("accepted")}>
            <Check size={16} /> Accept
          </button>
          <button type="button" className="warning-button" onClick={() => submitDecision("override")}>
            <SlidersHorizontal size={16} /> Override
          </button>
          <button type="button" className="danger-button" onClick={() => submitDecision("rejected")}>
            <X size={16} /> Reject
          </button>
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value, emphasis = false }) {
  return (
    <div className={`metric-card ${emphasis ? "metric-emphasis" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ScenarioSimulator({ pricingContext, updatePricingContext, getRecommendation, recommendation }) {
  return (
    <section className="scenario-layout">
      <div className="panel scenario-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Pricing controls</p>
            <h3>Scenario Simulator</h3>
          </div>
          <SlidersHorizontal size={20} />
        </div>

        <div className="slider-stack">
          <label>
            <span>Reconditioning cost: {currency(pricingContext.reconditioning_cost)}</span>
            <input
              type="range"
              min="0"
              max="5000"
              step="100"
              value={pricingContext.reconditioning_cost}
              onChange={(event) => updatePricingContext("reconditioning_cost", event.target.value)}
            />
          </label>
          <label>
            <span>Target margin: {percent(pricingContext.target_margin_pct)}</span>
            <input
              type="range"
              min="0.04"
              max="0.30"
              step="0.01"
              value={pricingContext.target_margin_pct}
              onChange={(event) => updatePricingContext("target_margin_pct", event.target.value)}
            />
          </label>
        </div>

        <button className="primary-button" type="button" onClick={getRecommendation}>
          <Gauge size={18} /> Recalculate
        </button>
      </div>

      <div className="panel scenario-summary">
        <p className="eyebrow">Latest output</p>
        {recommendation ? (
          <div className="metric-grid">
            <Metric label="Recommended offer" value={currency(recommendation.recommended_offer)} emphasis />
            <Metric label="Expected margin" value={currency(recommendation.expected_margin)} />
            <Metric label="Margin rate" value={percent(recommendation.expected_margin_pct)} />
            <Metric label="Risk score" value={recommendation.risk_score} />
          </div>
        ) : (
          <p className="muted">Run a recommendation from the workbench to populate scenario output.</p>
        )}
      </div>
    </section>
  );
}

function DecisionHistory({ decisions, refreshDecisions }) {
  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Operational audit</p>
          <h3>Pricing decisions</h3>
        </div>
        <button className="ghost-button" type="button" onClick={refreshDecisions}>
          <RefreshCcw size={16} /> Refresh
        </button>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Quote</th>
              <th>Decision</th>
              <th>Recommended</th>
              <th>Final</th>
              <th>User</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {decisions.length === 0 ? (
              <tr>
                <td colSpan="6" className="empty-cell">No decisions recorded yet.</td>
              </tr>
            ) : (
              decisions.map((decision) => (
                <tr key={decision.id}>
                  <td>{decision.quote_id}</td>
                  <td><Badge tone={decision.decision === "accepted" ? "success" : decision.decision === "rejected" ? "danger" : "warning"}>{decision.decision}</Badge></td>
                  <td>{currency(decision.recommended_offer)}</td>
                  <td>{currency(decision.final_offer)}</td>
                  <td>{decision.user_id}</td>
                  <td>{decision.reason || "-"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelHealth({ modelHealth }) {
  return (
    <section className="panel health-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Serving status</p>
          <h3>Model Health</h3>
        </div>
        <Activity size={20} />
      </div>

      {modelHealth ? (
        <div className="health-grid">
          <Metric label="Model type" value={modelHealth.model_type} />
          <Metric label="Target" value={modelHealth.prediction_target} />
          <Metric label="Status" value={modelHealth.status} />
          <div className="artifact-list">
            <span>Serving artifacts</span>
            {modelHealth.serving_artifacts.map((artifact) => (
              <code key={artifact}>{artifact}</code>
            ))}
          </div>
        </div>
      ) : (
        <p className="muted">Model health is unavailable. Confirm the FastAPI service is running.</p>
      )}
    </section>
  );
}

export default App;
