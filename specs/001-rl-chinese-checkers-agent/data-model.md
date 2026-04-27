# Data Model: RL Chinese Checkers Agent

## GameState

Represents one local or tournament-observed game state.

- `status`: available, playing, finished, win, draw, or timeout state
- `current_colour`: colour whose turn it is
- `player_colours`: colours participating in the game
- `pins_by_colour`: map from colour to ordered board indices for its 10 pins
- `turn_order`: ordered colours for turn rotation
- `move_count`: total moves made
- `last_move`: previous move summary if available

Validation:
- Active colour must be one of `player_colours`.
- Pin positions must refer to valid board cells.
- No two pins may occupy the same board cell.

## LegalMove

One valid action for the active player.

- `pin_id`: controlled pin identifier
- `from_index`: current board index
- `to_index`: destination board index
- `is_step`: whether move is adjacent
- `is_jump`: whether move is reachable by jump
- `distance_delta`: progress toward target zone
- `enters_goal`: whether destination enters target zone
- `future_jump_options`: number or score of jump opportunities after the move

Validation:
- `to_index` must be in the legal moves for `pin_id`.
- Tournament client may only submit moves derived from server-provided legal moves.

## ActionMapping

Fixed mapping used by RL wrappers to convert between discrete action ids and game moves.

- `action_id`: stable integer in the Gymnasium discrete action space
- `pin_id`: controlled pin identifier
- `to_index`: destination board index
- `is_valid_for_state`: derived from the current legal move set

Validation:
- Every legal `(pin_id, to_index)` pair for the active player maps to exactly one action id.
- Every true value in an action mask maps back to a legal move.
- Invalid action ids must not be submitted to the simulator or tournament server.

## ActionMask

Boolean vector used by MaskablePPO and tests to identify legal actions in the current state.

- `mask`: fixed-length boolean vector
- `legal_action_ids`: action ids currently allowed
- `source_move_count`: game move count when mask was produced

Validation:
- Mask length equals the fixed action-space size.
- Mask contains at least one legal action unless the active player has no moves.
- Mask must be regenerated after every state transition.

## GymObservation

RL-facing observation produced by the Gymnasium adapter.

- `board_features`: compact numeric encoding of pin occupancy, active colour, goals, and progress signals
- `player_features`: active player and opponent count/context features
- `mask`: legal action mask supplied through the MaskablePPO-compatible interface

Validation:
- Observation shape is stable across games and player counts.
- Observation values are numeric and deterministic for a given state.

## RewardBreakdown

Named reward components for one move or episode.

- `distance_progress`
- `goal_entry`
- `goal_occupancy`
- `regression_penalty`
- `jump_progress`
- `future_jump_setup`
- `finish_bonus`
- `invalid_or_timeout_penalty`
- `total`

Validation:
- `total` equals the sum of named components.
- Components are logged separately for report analysis.

## AgentCandidate

Reusable policy candidate.

- `candidate_id`
- `created_at`
- `source`: heuristic, maskable_ppo, custom_move_ranker, resumed, or selected
- `checkpoint_path`
- `training_config`
- `parent_candidate_id`
- `summary_metrics`

Validation:
- Candidate metadata must identify whether it is a fallback or trained policy.
- Selected candidate must refer to an existing candidate or fallback policy.

## TrainingSession

One training run.

- `session_id`
- `started_at`
- `curriculum_stage`
- `episodes_requested`
- `episodes_completed`
- `opponent_profile`
- `player_count_range`
- `resume_from_candidate`
- `output_candidates`
- `episode_metrics`
- `training_method`: heuristic, maskable_ppo, or custom_move_ranker

Validation:
- Interrupted sessions keep completed episode metrics.
- Resume sessions reference an existing candidate.

## EvaluationRun

Benchmark comparing one or more candidates.

- `run_id`
- `candidate_ids`
- `opponent_profiles`
- `player_counts`
- `games_per_setting`
- `aggregate_metrics`
- `selected_candidate_id`

Validation:
- Results are grouped by candidate, opponent profile, and player count.
- Selection criteria are recorded with the selected candidate.

## OpponentProfile

Baseline or self-play behavior.

- `name`: random, greedy, heuristic, self-play snapshot
- `description`
- `configuration`

Validation:
- Evaluation output must identify which opponent profile was used.

## ReportMetric

Report-ready aggregate.

- `metric_name`
- `value`
- `unit`
- `scope`: training, evaluation, candidate, opponent profile, or player count
- `source_file`

Validation:
- Metrics must be reproducible from saved training/evaluation outputs.
