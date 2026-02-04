# Literature Review: VPP Optimization & Explainability

**Date:** 2026-02-04
**Focus:** Imitation Learning for Optimization & XAI in Energy Systems (2022-2025)

## Executive Summary
Recent research (2022-2025) strongly supports the "Teacher-Student" approach (Imitation Learning) for solving complex energy optimization problems. As MILP solvers struggle with real-time requirements in decentralized grids, Neural Networks trained on offline optimal trajectories offer a powerful alternative—achieving near-optimal performance with 10x-100x faster execution. Concurrently, explainable AI (XAI), particularly SHAP (SHapley Additive exPlanations), has become the standard for validating these "black-box" models in safety-critical energy domains.

---

## 1. Imitation Learning for Optimization
**Goal:** Approximating computationally expensive solvers (MILP/MINLP) with fast Neural Networks.

### Key Papers & Findings

*   **"Imitation Learning for Real-Time Demand Response" (2024)** [1]
    *   **Contribution:** Proposed a DNN-based imitation learning framework trained on optimal operation schedules from an optimization problem.
    *   **Relevance:** Demonstrated that NN agents could replicate complex dispatch decisions significantly faster than solving the optimization problem directly, enabling real-time responsiveness.
    *   **Key Insight:** The "Teacher" does not need to run online; offline generation of high-quality labels is sufficient for robust student performance.

*   **"Machine Learning to Enhance MILP in Energy Optimization" (2025)** [2]
    *   **Contribution:** Explored Deep Reinforcement Learning (DRL) and Behavior Cloning to approximate MILP solutions in multi-energy communities.
    *   **Relevance:** Showed that learning-based agents achieved **near-optimal outcomes (within 1-2% gap)** while reducing computation time by orders of magnitude.
    *   **Key Insight:** Behavior Cloning (Supervised Learning on expert trajectories) is often more stable and sample-efficient than pure RL for these problems.

*   **"Learning to Optimize for Mixed-Integer Problems" (2024)** [5]
    *   **Contribution:** Investigated replacing traditional branch-and-bound heuristics with learned policies.
    *   **Relevance:** highlights that "imitation" isn't just about output matching but can learn the underlying structure of the constraints.

---

## 2. XAI for Energy & Battery Systems
**Goal:** Interpreting "Black-Box" energy decisions to ensure safety and trust.

### Key Papers & Findings

*   **"Optimizing Lithium-Ion Battery Performance: Integrating ML and XAI" (2024)** [7]
    *   **Contribution:** Applied SHAP values to identify critical features (Temperature, Voltage, Cycle Index) influencing battery capacity predictions.
    *   **Relevance:** Establishes SHAP as a viable method for analyzing battery systems.
    *   **Key Insight:** SHAP dependency plots revealed non-linear thresholds in battery behavior that linear models missed.

*   **"SPXAI: Solar Power Generation with Explainable AI" (2024)** [8]
    *   **Contribution:** Used LIME and SHAP to explain solar power output predictions.
    *   **Relevance:** Demonstrates XAI integration in renewable energy pipelines, similar to our VPP context.

*   **"Explainable AI Framework for EV Battery Degradation" (Upcoming 2025)** [1, 2]
    *   **Contribution:** Proposes a framework specifically for "transparent insights" into energy storage dynamics.
    *   **Relevance:** Confirms that XAI is not just a "nice-to-have" but a requirement for future energy management systems (EMS) to meet regulatory and operational transparency standards.

---

## 3. Similar Work & Best Practices

**Synthesized Best Practices from Literature:**
1.  **Teacher Quality:** The Student is only as good as the Teacher. Relying on simple heuristic teachers leads to suboptimal students. **Action:** We must use a global, multi-period MILP (Perfect Foresight) as the teacher.
2.  **Feature selection:** Successful models include not just current state ($S_t$) but also "Lookahead" features (e.g., average price next 4 hours) to give the Student context about future constraints.
3.  **Validation:** Simply measuring MSE is insufficient. Metrics must include "Constraint Violation Rate" and "Profit Optimality Gap" to be meaningful for energy systems.

## References
[1] Imitation Learning for Real-Time Demand Response (2024)
[2] Machine Learning to Enhance MILP in Energy Optimization (MDPI, 2025)
[5] Learning to Optimize for Mixed-Integer Problems (arXiv, 2024)
[7] Optimizing Lithium-Ion Battery Performance with XAI (MDPI, 2024)
[8] SPXAI: Solar Power Generation with Explainable AI (2024)
