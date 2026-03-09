# RSA (Rational Speech Acts) Agent for Hanabi

This directory contains the implementation of a Rational Speech Acts (RSA) agent for the Hanabi card game, based on pragmatic reasoning for cooperative communication.

## Overview

The RSA agent uses pragmatic reasoning to:
- **Generate clues** (as a pragmatic speaker S₁) that maximize informativity
- **Interpret clues** (as a pragmatic listener L₁) by reasoning about why the speaker chose that particular clue
- **Make decisions** based on belief distributions over possible card configurations

## Files

### Core Implementation

1. **`rsa_listener.jl`** - Listener models
   - `literal_listener()` - L₀ model: filters beliefs based on clue truthfulness
   - `pragmatic_listener()` - L₁ model: updates beliefs by reasoning about speaker intent
   - Helper functions for hand enumeration and probability computation

2. **`rsa_speaker.jl`** - Speaker model and QUD functions
   - `pragmatic_speaker()` - S₁ model: selects clues using softmax over informativity scores
   - `qud_play_score()` - QUD for identifying playable cards
   - `qud_discard_score()` - QUD for identifying safe discards
   - `choose_clue_s1()` - Main clue selection function

3. **`rsa_agent.jl`** - RSA agent implementation
   - `RSAHanabiAgent` - Main agent struct with configurable parameters
   - `choose_action()` - Action selection following priority order
   - `update_beliefs_hint!()` - Uses L₁ for clue interpretation
   - `update_beliefs_action!()` - Literal updates for play/discard observations

4. **`rsa_play_game.jl`** - Simulation and testing
   - `play_rsa_game()` - Play a single game with RSA agents
   - `run_rsa_simulations()` - Run multiple games with configurable parameters
   - `compare_agents()` - Compare RSA vs baseline greedy agents
   - `test_parameter_sensitivity()` - Test different parameter settings

## Dependencies

The RSA implementation depends on:
- `../hanabi_game_state.jl` - Game state, cards, actions (from baseline)
- `../agent.jl` - Agent interface, belief structures (from baseline)
- Julia packages: `Statistics`, `Random`

The baseline game state requires the `Gen` package for probabilistic deck shuffling:
```julia
import Pkg
Pkg.add("Gen")
```

## RSA Model Architecture

```
┌─────────────────────────────────────────────────┐
│                  RSA Agent                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  When Receiving Clue:                          │
│  ┌─────────────┐                               │
│  │ L₀ Literal  │ → Filter beliefs by truth     │
│  └──────┬──────┘                               │
│         │                                       │
│         ↓                                       │
│  ┌─────────────┐                               │
│  │ S₁ Speaker  │ → What would speaker say      │
│  │   Model     │    for each possible hand?    │
│  └──────┬──────┘                               │
│         │                                       │
│         ↓                                       │
│  ┌─────────────┐                               │
│  │ L₁ Pragmatic│ → Update beliefs by speaker   │
│  │  Listener   │    likelihood                 │
│  └─────────────┘                               │
│                                                 │
│  When Giving Clue:                             │
│  ┌─────────────┐                               │
│  │ QUD Select  │ → Choose goal (play/discard)  │
│  └──────┬──────┘                               │
│         │                                       │
│         ↓                                       │
│  ┌─────────────┐                               │
│  │ S₁ Speaker  │ → Score clues by              │
│  │   Model     │    informativity + QUD        │
│  └──────┬──────┘                               │
│         │                                       │
│         ↓                                       │
│  ┌─────────────┐                               │
│  │  Softmax    │ → Choose best/sample clue     │
│  └─────────────┘                               │
└─────────────────────────────────────────────────┘
```

## Configurable Parameters

The `RSAHanabiAgent` accepts the following parameters:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Speaker rationality | `α` | 1.0 | Controls how deterministically S₁ chooses clues (higher = more deterministic) |
| Play threshold | `θ_play` | 0.85 | Minimum probability to play a card |
| Discard threshold | `θ_discard` | 0.70 | Minimum probability to discard a card |
| QUD mode | `qud_mode` | `:dynamic` | Question Under Discussion (`:play`, `:discard`, or `:dynamic`) |
| Clue threshold | `clue_threshold` | 0.6 | Minimum S₁ score to prefer giving a clue |

### QUD Selection

- **`:play`** - Focus on identifying playable cards (maximize progress)
- **`:discard`** - Focus on identifying safe discards (recover info tokens)
- **`:dynamic`** - Automatically choose based on game state:
  - Use `:discard` when info tokens ≤ 2
  - Use `:play` otherwise (default)

### Action Priority Order

The RSA agent follows this **modified** decision hierarchy (optimized for aggressive play):

1. **Play card** if any card has P(playable) ≥ `θ_play` ⭐ **PRIORITIZED**
2. **Give clue** if info tokens > 0 AND best clue score ≥ `clue_threshold`
3. **Discard card** if info tokens < 8 AND any card has P(safe) ≥ `θ_discard`
4. **Give any clue** if info tokens > 0 (fallback)
5. **Discard least risky card** (ultimate fallback)

> **Key Change**: Playing cards is now prioritized over giving clues. This encourages agents to:
> - Make progress immediately when confident
> - Capitalize on received hints quickly
> - Avoid over-communication and under-playing
> - Score points more aggressively

## Usage Examples

### Basic Usage

```julia
include("rsa_play_game.jl")

# Run 100 simulations with default parameters (logs saved automatically)
results = run_rsa_simulations(
    100,        # num_simulations
    5,          # num_players
    5;          # cards_per_player
    α=1.0,
    θ_play=0.85,
    θ_discard=0.70,
    qud_mode=:dynamic,
    verbose=false
)

println("Mean score: ", results["mean"])
println("Victories: ", results["victories"])

# Logs are saved to scripts/rsa/logs/ directory
```

### Compare with Baseline

```julia
# Compare RSA agents vs greedy baseline (with logging)
compare_agents(100, 5, 5)

# Compare without saving logs
compare_agents(100, 5, 5; save_logs=false)
```

### Parameter Sensitivity

```julia
# Test different parameter settings (with logging)
test_parameter_sensitivity()

# Test without saving logs
test_parameter_sensitivity(save_logs=false)
```

### Custom Agent Configuration

```julia
# Create a custom RSA agent
game = init_game(5, 5)
agent = RSAHanabiAgent(
    1,                                  # player_id
    init_player_knowledge(game, 1),     # knowledge
    2.0,                                # α (more deterministic)
    0.90,                              # θ_play (more conservative)
    0.60,                              # θ_discard (more aggressive)
    :play,                             # qud_mode (always focus on playable cards)
    0.7                                # clue_threshold
)
```

### Custom Logging

```julia
# Run simulations with custom log file name
results = run_rsa_simulations(
    100, 5, 5;
    α=1.5,
    θ_play=0.90,
    qud_mode=:play,
    log_file="custom_rsa_test.txt"
)

# Run without logging
results = run_rsa_simulations(
    100, 5, 5;
    log_file=nothing  # No log file created
)
```

## Simulation Logs

All simulation results are automatically saved to the `logs/` directory with timestamps.

### Log File Types

1. **RSA Agent Logs** (`rsa_agent_YYYYMMDD_HHMMSS.txt`)
   - Detailed RSA agent performance
   - Parameter configuration
   - Score distribution and statistics

2. **Baseline Logs** (`baseline_greedy_YYYYMMDD_HHMMSS.txt`)
   - Baseline greedy agent performance
   - For comparison with RSA

3. **Comparison Logs** (`comparison_YYYYMMDD_HHMMSS.txt`)
   - Side-by-side comparison
   - Improvement analysis
   - Performance metrics

4. **Parameter Sensitivity Logs**
   - Individual: `param_[type]_[value]_YYYYMMDD_HHMMSS.txt`
   - Summary: `param_sensitivity_summary_YYYYMMDD_HHMMSS.txt`

### Log Contents

Each log includes:
- Configuration and parameters
- Mean/std/min/max scores
- Victory/loss statistics
- Score distribution histogram
- Individual game scores
- Quartile analysis

See [`logs/README.md`](logs/README.md) for detailed information about log formats.

## Implementation Details

### Belief Representation

- Reuses `CardBelief` structure from baseline agent
- Each agent maintains `Vector{CardBelief}` for own hand
- Each `CardBelief` stores:
  - `probs::Dict{Card, Float64}` - Probability distribution over possible cards
  - `known_color::Union{Symbol, Nothing}` - Known color (if any)
  - `known_number::Union{Int, Nothing}` - Known number (if any)
  - `known::Union{Card, Bool}` - Exact card (if fully determined)

### RSA Depth

Implements **L₀→S₁→L₁** (depth-1 recursion):
- **L₀ (Literal Listener)**: Filters hands consistent with clue
- **S₁ (Pragmatic Speaker)**: Chooses clues via softmax over informativity + QUD scores
- **L₁ (Pragmatic Listener)**: Weights hand configurations by P(clue | hand) from S₁

### Belief Updates

- **Clue reception**: Uses pragmatic listener (L₁) for RSA inference
- **Play/Discard observation**: Uses literal belief update (consistency filter)
- **Theory of mind**: Maintains beliefs about other players' beliefs (literal updates for simplicity)

## Key Differences from Baseline Agent

| Aspect | Baseline (Greedy) | RSA Agent |
|--------|-------------------|-----------|
| Clue selection | Heuristic scoring (playability + informativity) | S₁ pragmatic speaker model with QUD |
| Clue interpretation | Literal update only | L₁ pragmatic listener reasoning |
| Decision making | Fixed threshold-based | Configurable with RSA parameters |
| Communication model | None (greedy heuristic) | Explicit pragmatic reasoning |

## Expected Performance

- RSA agent should score **higher than random** agents (sanity check)
- RSA agent should perform **comparably to or better than** greedy baseline
- Benefits most visible when:
  - Clues need to be highly informative (few info tokens)
  - Disambiguating between similar cards
  - Coordinating on urgent plays

## Limitations & Extensions

### Current Limitations

1. **Computational cost**: Pragmatic listener enumerates hand configurations (can be slow)
2. **Theory of mind**: Uses literal updates for other players (not full RSA)
3. **No pragmatic inference on play/discard**: Actions only update beliefs literally

### Possible Extensions (Phase 2)

1. **Pragmatic action inference**: Treat play/discard as implicit clues
2. **Higher RSA depth**: Implement L₂, S₂ for deeper reasoning
3. **Optimized belief enumeration**: Use more efficient sampling/inference
4. **Multi-agent RSA**: All agents reason about each other's RSA models
5. **Learned QUD selection**: Adapt QUD choice based on game history

## Testing

Run syntax check:
```julia
julia scripts/rsa/test_syntax.jl
```

Run full simulation:
```julia
julia scripts/rsa/rsa_play_game.jl
```

This will:
1. Run 100 games with RSA agents
2. Run 100 games with baseline greedy agents
3. Compare performance statistics
4. Report mean scores, victory rates, and improvement percentage

## References

- RSA implementation based on [`docs/rsa_implementation.md`](../../docs/rsa_implementation.md)
- Baseline implementation in [`scripts/agent.jl`](../agent.jl)
- Game state definition in [`scripts/hanabi_game_state.jl`](../hanabi_game_state.jl)

## Authors

Implementation follows the RSA framework specification for cooperative Hanabi agents with pragmatic communication.
