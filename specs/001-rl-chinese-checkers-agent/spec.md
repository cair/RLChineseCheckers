# Feature Specification: RL Chinese Checkers Agent

**Feature Branch**: `001-rl-chinese-checkers-agent`  
**Created**: 2026-04-27  
**Status**: Draft  
**Input**: User description: "Build an RL Chinese Checkers agent for the existing RLChineseCheckers repository. The goal is to create a tournament-ready reinforcement learning agent that can train through self-play, evaluate against baseline opponents, and plug into the provided multi-system Chinese Checkers client/server setup. The project is for an IKT 460 reinforcement learning course where grading depends on tournament placement, having a running agent that can play a human, and a 5-page report. The agent must preserve the provided project files, integrate with the provided tournament flow, train locally, evaluate against baselines, support curriculum/self-play, and expose report-friendly metrics. Update: the main objective is to maximize tournament performance, not merely provide a working baseline. The fallback heuristic is only a safety net. The intended deliverable is a trained agent that should outperform simple random and greedy baseline policies significantly and becomes competitive with other student implementations. Further the deliverable is intended to support both short practice-tournament runs and longer continued training, compare candidate saved policies, and prioritize strong move selection including goal progress, long jump-chain setup, avoiding regressions, efficient finishing, and robustness across opponent counts. Update: the implementation should also read like a clear student project, with modular and direct organization, named reward components, report-friendly outputs, concise explanations of non-obvious choices, and limited abstraction so the algorithm and results are easy to explain in the final report."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train a Competitive Tournament Candidate Locally (Priority: P1)

As a student with limited project time, I want to train a Chinese Checkers agent locally without running the tournament server, so that I can rapidly improve tournament performance before each practice or final submission.

**Why this priority**: Tournament placement is the largest grading component, so training must aim for competitive play rather than stopping at a merely functional agent.

**Independent Test**: Can be fully tested by starting short and extended training runs from a clean checkout and confirming that training episodes complete, legal moves are always chosen, reusable candidates are saved, and later runs can continue improving from earlier candidates.

**Acceptance Scenarios**:

1. **Given** the repository is available locally, **When** the student starts a training session, **Then** the agent completes multiple Chinese Checkers games without requiring the external tournament server.
2. **Given** the agent is training, **When** legal moves are available, **Then** every selected move is accepted by the game rules.
3. **Given** a short practice-tournament deadline, **When** the student starts a short training run, **Then** the run produces a usable candidate and enough metrics to decide whether it is better than the safety fallback.
4. **Given** more training time is available, **When** the student resumes from a saved candidate, **Then** training continues from that candidate rather than starting over.
5. **Given** training progresses over multiple games, **When** metrics are reviewed, **Then** the student can see movement progress toward target zones, pins reaching goals, move counts, timing, reward trends, and baseline comparison trends.

---

### User Story 2 - Select the Best Tournament Candidate (Priority: P2)

As a student preparing for a round-robin tournament, I want to compare multiple saved agent candidates against baseline opponents across different player counts, so that I can select the strongest candidate for tournament play.

**Why this priority**: Performance may vary by candidate, opponent mix, and number of players, so selection must be based on measured tournament-relevant outcomes rather than the latest saved candidate.

**Independent Test**: Can be fully tested by running evaluation games for two or more candidates against random and goal-directed baseline opponents with different numbers of total players, then confirming that aggregate metrics and a selected best candidate are produced.

**Acceptance Scenarios**:

1. **Given** multiple agent candidates are available, **When** evaluation is run, **Then** results rank candidates using tournament-relevant metrics.
2. **Given** an agent candidate is available, **When** evaluation is run against random opponents, **Then** results include average score, average reward, average move count, pins in goal, distance progress, and timing statistics.
3. **Given** an agent candidate is available, **When** evaluation is run against simple goal-directed opponents, **Then** the same metrics are reported separately from random-opponent results.
4. **Given** evaluation is configured for different total player counts, **When** games complete, **Then** results are grouped so the student can compare two-player through six-player performance.
5. **Given** one candidate performs best under the configured evaluation, **When** selection completes, **Then** that candidate is identified as the tournament candidate.

---

### User Story 3 - Play Through the Provided Tournament Interface (Priority: P3)

As a student submitting a tournament entry, I want the trained agent to join the provided game flow and choose moves automatically, so that it can play against humans or tournament opponents without manual move selection.

**Why this priority**: Tournament and human-play compatibility is a grading requirement, but it depends on having a valid policy and evaluation loop first.

**Independent Test**: Can be fully tested by starting the provided game manager, joining a game with the agent player, and confirming that the agent submits valid moves within the allowed turn time.

**Acceptance Scenarios**:

1. **Given** the provided game manager is running and a game is available, **When** the agent player joins, **Then** it can participate without manual move entry during its turns.
2. **Given** it is the agent's turn, **When** legal moves are provided by the game, **Then** the agent selects and submits one legal move well before the turn timeout.
3. **Given** no trained candidate is available, **When** the agent player must choose a move, **Then** it uses a deterministic fallback policy rather than failing or waiting for manual input.
4. **Given** a selected tournament candidate exists, **When** the agent player starts, **Then** it uses the selected candidate by default instead of an arbitrary saved candidate.

---

### User Story 4 - Collect Report Evidence (Priority: P4)

As a student writing the final report, I want training and evaluation outputs that summarize algorithm behavior and game outcomes, so that I can support claims about reward design, training process, timing, and performance.

**Why this priority**: Report evidence affects grading and should be collected continuously rather than reconstructed after development.

**Independent Test**: Can be fully tested by running training and evaluation and confirming that report-ready metrics are saved or displayed in a clear, repeatable format.

**Acceptance Scenarios**:

1. **Given** training has run, **When** the student reviews outputs, **Then** average reward, average score, average moves per game, and goal-progress statistics are available.
2. **Given** tournament-style evaluation has run, **When** the student reviews outputs, **Then** average time per move and average total game duration are available.
3. **Given** multiple evaluation configurations have run, **When** the student compares results, **Then** the differences between opponent types and player counts are visible.

---

### User Story 5 - Understand and Explain the Implementation (Priority: P5)

As a student who must deliver a written report and understand the submitted work, I want the agent implementation to be organized and explainable, so that I can describe the algorithm, reward design, training process, and results without relying on hidden or opaque behavior.

**Why this priority**: Report quality depends on being able to explain the implementation clearly, and understandable code reduces the risk of delivering a system the student cannot defend.

**Independent Test**: Can be fully tested by reviewing the project after implementation and confirming that the major concepts are easy to locate, reward components are named in outputs, and non-obvious reinforcement learning or game-specific choices have concise explanations.

**Acceptance Scenarios**:

1. **Given** the student opens the project, **When** they look for state representation, legal action handling, action selection, reward calculation, training, checkpointing, evaluation, or tournament play, **Then** each concept is easy to locate in a clearly named module or section.
2. **Given** training or evaluation has run, **When** the student reviews the output, **Then** reward components and performance metrics are named clearly enough to use in report tables or figures.
3. **Given** a non-obvious reinforcement learning or game-specific choice exists, **When** the student reads the nearby explanation, **Then** the reason for that choice is understandable without reverse-engineering the entire project.
4. **Given** the implementation is reviewed for the final report, **When** the student traces a move decision from game state to submitted move, **Then** the path is direct and explainable.

### Edge Cases

- The agent has no trained candidate available when asked to play.
- Several saved candidates exist and the most recent one is not the strongest.
- The current player has no legal moves.
- The game includes any supported total player count from two through six.
- Opponents block common jump routes or create unusual board congestion.
- The agent receives game state for colors or turn orders different from previous training examples.
- Training or evaluation is interrupted before completion.
- A move candidate becomes invalid because the board state changed before submission.
- The game ends by timeout, win condition, or draw condition.
- A greedy immediate jump would reduce long-term progress compared with setting up a stronger future jump chain.
- A complex abstraction or external framework would hide the algorithmic choices that must be explained in the report.
- Training metrics exist but are too vague to support report figures or tables.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a local Chinese Checkers training experience that does not require the external game manager to be running.
- **FR-002**: The system MUST preserve the provided project files and add new project artifacts for the agent work.
- **FR-003**: The system MUST use the same game rules as the provided Chinese Checkers implementation, including single-step moves, chained hops, no captures, unrestricted colored-region entry, win conditions, draw conditions, and tournament scoring concepts.
- **FR-004**: The system MUST support training with one controlled learning agent and between one and five opponents.
- **FR-005**: The system MUST support at least one movement-focused curriculum stage before full opponent-play training.
- **FR-006**: The system MUST support later training or practice against random, goal-directed, and self-play opponent behavior.
- **FR-007**: The system MUST ensure the agent only selects legal moves for the current board state.
- **FR-008**: The system MUST reward or measure progress toward moving all controlled pins into the opposite target zone.
- **FR-009**: The system MUST reward or measure effective use of long movement opportunities, including chained jumps and setup moves that enable stronger later jumps.
- **FR-010**: The system MUST avoid overvaluing greedy immediate jumps when they reduce long-term goal progress or efficient finishing.
- **FR-011**: The system MUST discourage wasted movement and regressions that do not improve tournament-relevant outcomes.
- **FR-012**: The system MUST support both short practice-tournament training runs and longer continued training runs through the same user workflow.
- **FR-013**: The system MUST allow training to continue from a saved candidate.
- **FR-014**: The system MUST save multiple agent candidates or policy states that can be reused for later evaluation or play.
- **FR-015**: The system MUST provide a deterministic fallback policy when no saved candidate is available, while treating that fallback as a safety net rather than the target deliverable.
- **FR-016**: The system MUST evaluate the agent against random opponents.
- **FR-017**: The system MUST evaluate the agent against simple goal-directed opponents.
- **FR-018**: The system MUST compare trained candidates against random and goal-directed baseline behavior.
- **FR-019**: The system MUST identify or export a selected tournament candidate from evaluation results.
- **FR-020**: The system MUST evaluate the agent across multiple supported player counts.
- **FR-021**: The system MUST report average reward, average score, average move count, average time per move, average game duration, pins in goal, and distance-to-goal progress.
- **FR-022**: The system MUST provide a tournament-compatible player experience that can join the provided game flow and submit moves automatically.
- **FR-023**: The tournament-compatible player MUST submit moves within the configured turn limit during normal operation.
- **FR-024**: The system MUST expose enough training and evaluation evidence to support the final course report.
- **FR-025**: The system MUST make failures understandable to the student, including missing saved candidates, unavailable game manager, invalid game state, lack of legal moves, and interrupted training.
- **FR-026**: The system MUST organize core concepts so state representation, legal action handling, action selection, reward calculation, training loop, checkpoint save/load, evaluation, and tournament-compatible play are easy to locate.
- **FR-027**: The system MUST name and report reward components clearly enough for the student to describe reward design in the final report.
- **FR-028**: The system MUST provide training and evaluation outputs that can be converted into report tables or figures without manual reconstruction from raw logs.
- **FR-029**: The system MUST include concise explanations for non-obvious reinforcement learning and game-specific choices.
- **FR-030**: The system MUST avoid unnecessary abstraction, excessive indirection, or opaque framework behavior unless it clearly improves tournament performance or reliability.

### Key Entities *(include if feature involves data)*

- **Game State**: The current board, active player, player colors, pin locations, turn order, move count, status, and latest move information needed to choose valid moves.
- **Legal Move Set**: The available move choices for the active player, represented by pin identity and destination location.
- **Agent Candidate**: A reusable learned or fallback decision policy with associated metadata such as creation time, training configuration, continuation source, and evaluation summary.
- **Selected Tournament Candidate**: The saved candidate identified by evaluation as the current best choice for tournament-compatible play.
- **Training Session**: A sequence of simulated games with curriculum stage, opponent setup, reward outcomes, timing, and aggregate progress metrics.
- **Evaluation Run**: A repeatable benchmark against specified opponent types and player counts, producing aggregate performance and timing metrics.
- **Opponent Profile**: A baseline or self-play behavior used during training or evaluation.
- **Report Metrics**: Aggregated values needed for the course report, including reward, score breakdown, move counts, timing, pins in goal, and distance progress.
- **Reward Component**: A named contribution to training feedback, such as goal progress, jump-chain opportunity, regression penalty, finishing progress, or timing-related behavior.
- **Explanation Note**: A concise description of a non-obvious design choice that helps the student explain the implementation in the report.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A student can complete a local short practice training run of at least 100 games without starting the external game manager.
- **SC-002**: Across training, evaluation, and tournament-compatible play, 100% of submitted moves are legal for the board state used to choose them.
- **SC-003**: During tournament-compatible play, the agent submits a move in under 2 seconds for at least 95% of its turns on a normal local machine.
- **SC-004**: Evaluation reports include all required report metrics for each tested opponent type and player count.
- **SC-005**: The agent can complete evaluation games with every supported total player count from two through six.
- **SC-006**: If no trained candidate is available, the fallback policy can still complete a game without manual move selection.
- **SC-007**: The provided original project files remain unchanged after adding this feature.
- **SC-008**: After evaluation, the student can identify whether trained candidates perform better than random baseline behavior on average score and distance-to-goal progress.
- **SC-009**: After evaluation, the student can identify whether at least one trained candidate performs better than a simple goal-directed baseline on at least one tournament-relevant metric.
- **SC-010**: The student can compare at least two saved candidates and identify one selected tournament candidate.
- **SC-011**: The student can resume training from a saved candidate and produce a later candidate without discarding earlier evaluation results.
- **SC-012**: A student can locate each major concept needed for the report within 2 minutes of opening the project.
- **SC-013**: Training and evaluation outputs include named reward components and metrics suitable for at least one report table or figure.
- **SC-014**: A student can trace a tournament move decision from received game state to submitted legal move without following more than three major conceptual steps.

## Assumptions

- The provided Chinese Checkers rules and scoring behavior are the authority for tournament compatibility.
- The expected tournament uses the same move submission shape as the provided player flow: choose one controlled pin and one destination from legal moves.
- Training time is limited, so the first complete version should produce a minimally competitive candidate quickly, then allow stronger candidates to be produced through continued training.
- The fallback policy is a safety mechanism and baseline reference, not the intended final tournament strategy.
- Training rewards should balance immediate progress with longer-term positional advantages, especially moves that set up useful jump chains.
- Human play means the agent can participate in a game where at least one other player may be controlled manually or by another client.
- Report-ready metrics may be displayed, saved, or both, as long as they are repeatable and easy to collect.
- Specific programming language, library, file naming, and algorithm choices are planning decisions and are intentionally deferred to the implementation plan.
- Implementation understandability is a project goal alongside tournament strength; overly generic or opaque designs are out of scope unless they provide a clear performance or reliability benefit.
