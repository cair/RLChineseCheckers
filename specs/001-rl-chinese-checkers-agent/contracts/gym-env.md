# Gymnasium Environment Contract

The Gymnasium adapter wraps the project-owned local simulator. It must not contain independent game rules that can drift from `rl_agent.environment`.

## Environment Identity

Suggested id:

```text
RLChineseCheckers-v0
```

## Observation

Observation must be stable numeric data suitable for RL training.

Required properties:
- Same shape across 2-6 player games.
- Encodes active player perspective.
- Encodes board occupancy and goal-progress signals.
- Does not hide reward logic or legal move generation inside opaque library callbacks.

## Action Space

The action space is fixed and discrete.

Action id maps to:

```text
(pin_id, to_index)
```

Required behavior:
- Every legal simulator move for the current player maps to one action id.
- Invalid action ids are rejected by the simulator wrapper.
- The tournament wrapper can convert selected action ids back into `(pin_id, to_index)`.

## Action Mask

The environment must expose a MaskablePPO-compatible mask.

Required behavior:
- Mask length equals action-space size.
- `True` means the action id is legal in the current state.
- `False` means the action id is illegal in the current state.
- Mask is regenerated after every `reset()` and `step()`.
- Tests must compare the mask against the simulator's direct legal move set.

## Step Result

Each step returns:
- observation
- scalar reward
- terminated flag
- truncated flag
- info dictionary

Required `info` keys:
- `reward_breakdown`
- `legal_move_count`
- `selected_move`
- `score`
- `pins_in_goal`
- `total_distance`
- `move_count`

## Optional Dependency Behavior

If Gymnasium is unavailable:
- direct simulator, heuristic policies, evaluation, and tournament wrapper must still work.
- the Gym wrapper should fail with a clear dependency message.

If Stable-Baselines3 or sb3-contrib is unavailable:
- MaskablePPO training should fail with a clear dependency message.
- heuristic and custom fallback paths should remain available.
