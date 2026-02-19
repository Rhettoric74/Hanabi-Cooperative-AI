using Gen
using Random
import Random: shuffle!

# ============================================================================
# CARD REPRESENTATION
# ============================================================================

struct Card
    color::Symbol
    number::Int
end

Base.show(io::IO, c::Card) = print(io, "$(c.color) $(c.number)")
Base.:(==)(a::Card, b::Card) = a.color == b.color && a.number == b.number
Base.hash(c::Card, h::UInt) = hash(c.color, hash(c.number, h))

# ============================================================================
# DECK MODELING
# ============================================================================

# Complete deck composition (60 cards)
const DECK_COMPOSITION = Dict{Card, Int}()
colors = [:red, :white, :green, :blue, :yellow, :rainbow]

for color in colors
    # Three 1s
    DECK_COMPOSITION[Card(color, 1)] = 3
    
    # Two each of 2,3,4
    for num in 2:4
        DECK_COMPOSITION[Card(color, num)] = 2
    end
    
    # One 5
    DECK_COMPOSITION[Card(color, 5)] = 1
end

# Create a full deck as a vector
function create_full_deck()
    deck = Card[]
    for (card, count) in DECK_COMPOSITION
        for _ in 1:count
            push!(deck, card)
        end
    end
    return deck
end

# Gen model for shuffled deck
@gen function shuffled_deck_model()
    deck = create_full_deck()
    n = length(deck)
    
    # Fisher-Yates shuffle with trace addresses
    for i in 1:n-1
        j = {Symbol("swap_$i")} ~ uniform_discrete(i, n)
        deck[i], deck[j] = deck[j], deck[i]
    end
    
    return deck
end

@gen function deal_hands_model(num_players::Int, cards_per_player::Int)
    deck = {:deck} ~ shuffled_deck_model()
    
    player_hands = [Card[] for _ in 1:num_players]
    card_idx = 1
    
    for player in 1:num_players
        for _ in 1:cards_per_player
            if card_idx <= length(deck)
                push!(player_hands[player], deck[card_idx])
                card_idx += 1
            end
        end
    end
    
    remaining_deck = deck[card_idx:end]
    
    return player_hands, remaining_deck
end

# ============================================================================
# PUBLIC GAME STATE
# ============================================================================

# Fix the PublicGameState constructor
mutable struct PublicGameState
    played_stacks::Dict{Symbol, Int}  # :red, :white, :green, :blue, :yellow only
    info_tokens::Int
    explosion_tokens::Int
    discard_pile::Vector{Card}
    deck_size::Int
    
    # Inner constructor with positional arguments
    function PublicGameState(
        played_stacks::Dict{Symbol, Int},
        info_tokens::Int,
        explosion_tokens::Int,
        discard_pile::Vector{Card},
        deck_size::Int
    )
        new(played_stacks, info_tokens, explosion_tokens, discard_pile, deck_size)
    end
end

# Outer constructor with defaults (positional, not keyword)
function PublicGameState(
    played_stacks=Dict(:red=>0, :white=>0, :green=>0, :blue=>0, :yellow=>0),
    info_tokens=8,
    explosion_tokens=0,
    discard_pile=Card[],
    deck_size=60
)
    return PublicGameState(played_stacks, info_tokens, explosion_tokens, discard_pile, deck_size)
end

const MAX_SCORE = 25  # 5 colors × 5 numbers
const WIN_CONDITION = Dict(:red=>5, :white=>5, :green=>5, :blue=>5, :yellow=>5)

# Check if a card can be played
function can_play_card(state::PublicGameState, card::Card)
    if card.color == :rainbow
        # Rainbow can be played as any missing number
        return any(current < 5 && card.number == current + 1 
                   for (_, current) in state.played_stacks)
    else
        current = get(state.played_stacks, card.color, 0)
        return card.number == current + 1
    end
end

# Get valid rainbow placement options
function valid_rainbow_placements(state::PublicGameState, rainbow_number::Int)
    return [color for (color, current) in state.played_stacks 
            if current == rainbow_number - 1]
end

# Play a card successfully (rainbow version with target)
function play_rainbow_card!(state::PublicGameState, card::Card, target_color::Symbol)
    if card.color != :rainbow
        error("Can only use this function for rainbow cards")
    end
    
    if target_color ∉ valid_rainbow_placements(state, card.number)
        error("Invalid target color $target_color for rainbow $(card.number)")
    end
    
    state.played_stacks[target_color] = card.number
    
    if card.number == 5 && state.info_tokens < 8
        state.info_tokens += 1
    end
    
    return state
end

# Play a regular card successfully
function play_card!(state::PublicGameState, card::Card)
    if card.color == :rainbow
        error("Use play_rainbow_card! for rainbow cards")
    end
    
    if !can_play_card(state, card)
        error("Cannot play card $card")
    end
    
    state.played_stacks[card.color] = card.number
    
    if card.number == 5 && state.info_tokens < 8
        state.info_tokens += 1
    end
    
    return state
end

# Discard a card
function discard_card!(state::PublicGameState, card::Card)
    push!(state.discard_pile, card)
    return state
end

# Fail to play a card (explosion)
function fail_play!(state::PublicGameState, card::Card)
    state.explosion_tokens += 1
    push!(state.discard_pile, card)
    return state
end

# Use an information token
function use_info_token!(state::PublicGameState)
    if state.info_tokens <= 0
        error("No information tokens available")
    end
    state.info_tokens -= 1
    return state
end

# Check game over conditions
function is_game_over(state::PublicGameState)
    if state.explosion_tokens >= 4
        return true, :explosion_loss
    end
    
    if all(state.played_stacks[color] == 5 for color in keys(WIN_CONDITION))
        return true, :victory
    end
    
    return false, :ongoing
end

# Current score
function current_score(state::PublicGameState)
    return sum(values(state.played_stacks))
end

# ============================================================================
# FULL GAME STATE (with hidden info)
# ============================================================================

mutable struct FullGameState
    public::PublicGameState
    player_hands::Vector{Vector{Card}}
    deck::Vector{Card}
    current_player::Int
    history::Vector{String}
    
    function FullGameState(
        public=PublicGameState(),
        player_hands=Vector{Card}[],
        deck=Card[],
        current_player=1,
        history=String[]
    )
        new(public, player_hands, deck, current_player, history)
    end
end

# Initialize a new game
function init_game(num_players::Int=3, cards_per_player::Int=5)
    trace, = generate(deal_hands_model, (num_players, cards_per_player))
    player_hands, remaining_deck = get_retval(trace)
    
    # Use positional arguments, not keyword
    public = PublicGameState(
        Dict(:red=>0, :white=>0, :green=>0, :blue=>0, :yellow=>0),  # played_stacks
        8,                                                           # info_tokens
        0,                                                           # explosion_tokens
        Card[],                                                      # discard_pile
        length(remaining_deck)                                       # deck_size
    )
    
    return FullGameState(public, player_hands, remaining_deck, 1, 
                        ["Game started with $num_players players"])
end
# ============================================================================
# PLAYER KNOWLEDGE STATE
# ============================================================================

# A player's belief about a card slot
struct CardBelief
    probs::Dict{Card, Float64}
    known::Bool
    known_color::Union{Symbol,Nothing}
    known_number::Union{Int,Nothing}
end

# Player's knowledge state (what they know about the game)
struct PlayerKnowledge
    own_hand::Vector{CardBelief}
    other_hands::Dict{Int, Vector{Card}}  # Direct observation
    discard_pile::Vector{Card}
    played_stacks::Dict{Symbol,Int}
    theory_of_mind::Dict{Int, Vector{CardBelief}}  # Beliefs about others' beliefs
    info_tokens::Int
    explosion_tokens::Int
    deck_size::Int
end

# Compute remaining cards based on observed cards
function compute_remaining_cards(observed_cards::Vector{Card})
    remaining = copy(DECK_COMPOSITION)
    
    for card in observed_cards
        remaining[card] -= 1
        if remaining[card] < 0
            error("Observed more copies of $card than exist in deck")
        end
    end
    
    return remaining
end

# Create belief based on observed cards
function create_informed_belief(observed_cards::Vector{Card})
    remaining = compute_remaining_cards(observed_cards)
    total_remaining = sum(values(remaining))
    
    probs = Dict{Card, Float64}()
    for (card, count) in remaining
        if count > 0
            probs[card] = count / total_remaining
        end
    end
    
    return CardBelief(probs, false, nothing, nothing)
end

# Initialize a player's knowledge from the full game state
function init_player_knowledge(game::FullGameState, player_id::Int)
    n_players = length(game.player_hands)
    
    # Cards visible to this player (all cards not in their own hand)
    visible_cards = Card[]
    for (i, hand) in enumerate(game.player_hands)
        if i != player_id
            append!(visible_cards, hand)
        end
    end
    append!(visible_cards, game.public.discard_pile)
    for (color, level) in game.public.played_stacks
        for num in 1:level
            push!(visible_cards, Card(color, num))
        end
    end
    
    # Own hand beliefs
    own_hand_size = length(game.player_hands[player_id])
    own_beliefs = [create_informed_belief(visible_cards) for _ in 1:own_hand_size]
    
    # Other players' hands - directly observable
    other_hands = Dict(
        p => game.player_hands[p] for p in 1:n_players if p != player_id
    )
    
    # Theory of mind for each other player
    theory = Dict{Int, Vector{CardBelief}}()
    for p in 1:n_players
        if p != player_id
            # What player p can see: all cards except player p's hand
            p_visible = Card[]
            for (i, hand) in enumerate(game.player_hands)
                if i != p
                    append!(p_visible, hand)
                end
            end
            append!(p_visible, game.public.discard_pile)
            for (color, level) in game.public.played_stacks
                for num in 1:level
                    push!(p_visible, Card(color, num))
                end
            end
            
            p_hand_size = length(game.player_hands[p])
            theory[p] = [create_informed_belief(p_visible) for _ in 1:p_hand_size]
        end
    end
    
    return PlayerKnowledge(
        own_beliefs,
        other_hands,
        copy(game.public.discard_pile),
        copy(game.public.played_stacks),
        theory,
        game.public.info_tokens,
        game.public.explosion_tokens,
        game.public.deck_size
    )
end

# ============================================================================
# GAME ACTIONS
# ============================================================================

# Action types
abstract type Action end
struct PlayCard <: Action
    player::Int
    card_index::Int
end
struct DiscardCard <: Action
    player::Int
    card_index::Int
end
struct GiveHint <: Action
    giver::Int
    receiver::Int
    hint_type::Union{Symbol, Int}  # :color or number
    hint_value::Union{Symbol, Int}
end
# TODO: make this more sophisticated, usually it's a player's choice which stack a rainbow card get's played on
# Player's don't need to specify which stack until after they play the card.
function rainbow_choice(options, game)
    return rand(options)
end

# Execute a play action
function execute_action!(game::FullGameState, action::PlayCard)
    player = action.player
    card = game.player_hands[player][action.card_index]
    
    push!(game.history, "Player $player attempts to play $card")
    
    if can_play_card(game.public, card)
        if card.color == :rainbow
            stack = rainbow_choice(valid_rainbow_placements(game, card.color), game)
            play_rainbow_card!(game, card, stack)
        else
            play_card!(game.public, card)
        end
        push!(game.history, "  ✓ Successful")
        
        # Remove card from hand
        deleteat!(game.player_hands[player], action.card_index)
        
        # Draw new card if available
        if !isempty(game.deck)
            new_card = popfirst!(game.deck)
            push!(game.player_hands[player], new_card)
            game.public.deck_size = length(game.deck)
        end
    else
        fail_play!(game.public, card)
        push!(game.history, "  ✗ Failed (explosion)")
        
        deleteat!(game.player_hands[player], action.card_index)
        
        if !isempty(game.deck)
            new_card = popfirst!(game.deck)
            push!(game.player_hands[player], new_card)
            game.public.deck_size = length(game.deck)
        end
    end
    
    game.current_player = mod1(game.current_player + 1, length(game.player_hands))
    return game
end

# Execute a discard action
function execute_action!(game::FullGameState, action::DiscardCard)
    player = action.player
    card = game.player_hands[player][action.card_index]
    
    if game.public.info_tokens >= 8
        error("Cannot discard when info tokens are maxed")
    end
    
    push!(game.history, "Player $player discards $card")
    
    discard_card!(game.public, card)
    deleteat!(game.player_hands[player], action.card_index)
    game.public.info_tokens += 1
    
    if !isempty(game.deck)
        new_card = popfirst!(game.deck)
        push!(game.player_hands[player], new_card)
        game.public.deck_size = length(game.deck)
    end
    
    game.current_player = mod1(game.current_player + 1, length(game.player_hands))
    return game
end

