# Metrics Contract

Metrics are saved in report-friendly CSV and/or JSON files.

## Episode Metrics

Required fields:
- `session_id`
- `episode`
- `candidate_id`
- `player_count`
- `opponent_profile`
- `training_method`
- `total_reward`
- `final_score`
- `moves`
- `pins_in_goal`
- `total_distance`
- `distance_progress`
- `avg_time_per_move_sec`
- `game_duration_sec`
- `won`
- `draw`
- named reward components
- `mask_legal_count`
- `invalid_action_attempts`

## Evaluation Metrics

Required fields:
- `run_id`
- `candidate_id`
- `opponent_profile`
- `candidate_source`
- `player_count`
- `games`
- `avg_reward`
- `avg_score`
- `avg_moves`
- `avg_pins_in_goal`
- `avg_total_distance`
- `avg_distance_progress`
- `avg_time_per_move_sec`
- `avg_game_duration_sec`
- `legal_move_rate`
- `action_mask_match_rate`
- `selected_candidate`

## Candidate Metadata

Required fields:
- `candidate_id`
- `created_at`
- `source`
- `checkpoint_path`
- `parent_candidate_id`
- `training_profile`
- `training_method`
- `episodes_completed`
- `summary_metrics`

## Report Expectations

Saved metrics must support at least:
- a table comparing heuristic, random, greedy, and trained candidates
- a table comparing heuristic, custom-trained, and MaskablePPO candidates when available
- a table or figure of reward components over training
- a table or figure of action-mask validation and legal-move rate
- a table of timing and move-count statistics
- a table comparing performance across player counts
