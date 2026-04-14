include("hanabi_game_state.jl")

# ============================================================================
# RSA INFRASTRUCTURE 
# ============================================================================
# This module contains utility functions and constants for RSA-based
# hint selection and belief updating in Hanabi. Functions are adapted
# from rsa_agent.jl and tailored for the rsa_v2 agent architecture.
# ============================================================================

# RSA Constants
const RSA_RATIONALITY_DEFAULT = 3.0  # α parameter for softmax (higher = more deterministic)
const UTILITY_PLAYABLE = 3.0         # Weight for playable cards (highest priority)
const UTILITY_CRITICAL = 2.0         # Weight for critical cards (5s, last copies)
const UTILITY_DISPENSABLE = 1.0      # Weight for dispensable cards (safe to discard)
const UTILITY_INFO = 0.5             # Weight for general information gain

# ============================================================================
# CARD VALUE ASSESSMENT FUNCTIONS
# ============================================================================

"""
    is_critical_card(card::Card, played_stacks::Dict{Symbol, Int}) -> Bool

Returns true if the card is critical and should be saved for later play.

A card is critical if:
- It is a 5 (only one per color, completes the stack)
- It is the next card needed for a color (number = current_stack + 1)
  AND that number appears only once or twice in deck
"""
function is_critical_card(card::Card, played_stacks::Dict{Symbol, Int})
    # All 5s are critical (only one per color)
    if card.number == 5
        return true
    end
    
    # Rainbow cards are not critical in this variant
    if card.color == :rainbow
        return false
    end
    
    # Check if this is the next card needed for the color
    current_stack = get(played_stacks, card.color, 0)
    if card.number == current_stack + 1
        # This is the next card for this color
        # Count how many cards are in the deck with this number
        copies = get(DECK_COMPOSITION, card, 0)
        
        # Critical if only 1 or 2 copies exist (rare cards)
        return copies <= 2
    end
    
    return false
end

"""
    is_dispensable_card(card::Card, played_stacks::Dict{Symbol, Int}) -> Bool

Returns true if the card is dispensable and safe to discard.

A card is dispensable if its number is less than or equal to the current
stack level for its color. This guarantees the card cannot be played,
as all cards of lower numbers have already been played.

IMPORTANT: We do NOT infer dispensability from discard pile inspection.
Observing a card in the discard pile does not guarantee other copies are
safe to discard—the first copy might have been discarded by mistake.
"""
function is_dispensable_card(card::Card, played_stacks::Dict{Symbol, Int})
    # Rainbow cards can be played on any stack
    if card.color == :rainbow
        # Dispensable if the minimum stack level has reached this card's number
        min_stack = minimum(values(played_stacks))
        return card.number <= min_stack
    else
        # Regular card is dispensable if already played (number ≤ current stack)
        current_stack = get(played_stacks, card.color, 0)
        return card.number <= current_stack
    end
end

"""
    speaker_utility(card::Card, is_playable::Bool, is_critical::Bool, 
                   is_dispensable::Bool) -> Float64

Map a card's category to its utility weight for the pragmatic speaker.

Returns one of: 3.0 (playable), 2.0 (critical), 1.0 (dispensable), 0.5 (info)

This implements the utility hierarchy: 
  Playable > Critical > Disposable > General Info
"""
function speaker_utility(card::Card, is_playable::Bool, is_critical::Bool, 
                        is_dispensable::Bool)::Float64
    if is_playable
        return UTILITY_PLAYABLE
    elseif is_critical
        return UTILITY_CRITICAL
    elseif is_dispensable
        return UTILITY_DISPENSABLE
    else
        return UTILITY_INFO
    end
end

# ============================================================================
# NOTE ON DECK COMPOSITION
# ============================================================================
# compute_remaining_cards() is defined in hanabi_game_state.jl and should
# be used from there. The DECK_COMPOSITION constant is also defined in
# hanabi_game_state.jl and is available here via the include() statement.
# We do not duplicate these here to avoid maintenance burden and version drift.

# ============================================================================
# EXPORT
# ============================================================================
# These functions are intended to be used by RSAHanabiAgentV2.
# Note: compute_remaining_cards() should be imported from hanabi_game_state.jl

export is_critical_card,
       is_dispensable_card,
       speaker_utility,
       RSA_RATIONALITY_DEFAULT,
       UTILITY_PLAYABLE,
       UTILITY_CRITICAL,
       UTILITY_DISPENSABLE,
       UTILITY_INFO
