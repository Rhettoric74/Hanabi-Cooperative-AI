include("hanabi_game_state.jl")

# ============================================================================
# ABSTRACT AGENT INTERFACE
# ============================================================================

"""
    AbstractHanabiAgent

Abstract type for Hanabi players. All concrete agents must implement the
methods below. Agents are expected to maintain their own internal state
(e.g., beliefs), but they receive the full `FullGameState` in each method
to allow recomputing permissible knowledge if needed.
"""
abstract type AbstractHanabiAgent end

"""
    choose_action(agent::AbstractHanabiAgent, game::FullGameState) -> Action

Return the action the agent wishes to take, given the current game state.
The agent must not use hidden information (e.g., its own cards' identities)
unless it has deduced them legally from hints.
"""
function choose_action end

"""
    update_beliefs_hint!(agent::AbstractHanabiAgent, hint::CardHint, game::FullGameState)

Update the agent's internal beliefs after a hint is given. The `hint`
describes who gave the hint, who received it, which attribute was hinted,
and which card indices in the receiver's hand match the attribute.
"""
function update_beliefs_hint! end

"""
    update_beliefs_action!(agent::AbstractHanabiAgent, action::Action,
                           acting_player::Int, game::FullGameState)

Update the agent's internal beliefs after observing an action taken by
`acting_player` (which may be the agent itself or another player).
"""
function update_beliefs_action! end


"""
    RandomHanabiAgent <: AbstractHanabiAgent

An agent that picks uniformly at random from all currently legal actions.
It does not maintain persistent beliefs; instead it recomputes its knowledge
from the game state each time it needs to decide. This agent is useful for
testing and as a baseline.
"""
mutable struct RandomHanabiAgent <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
end

function label_hinted_cards!(card_beliefs::Vector{CardBelief}, indices::Vector{Int}, attribute::Union{Symbol, Int})
    for index in indices
        card_belief = card_beliefs[index]
        if attribute isa Int
            card_belief.known_number = attribute
        else
            card_belief.known_color = attribute
        end
        if !isnothing(card_belief.known_number) && !isnothing(card_belief.known_color)
            card_belief.known = true
        end
    end
end

function literal_belief_update!(card_beliefs::Vector{CardBelief}, observed_cards::Vector{Card})
    remaining = compute_remaining_cards(observed_cards)
    
    for card_belief in card_beliefs
        probs = Dict{Card, Float64}()
        for (card, count) in remaining
            total_remaining = 0
            if !isnothing(card_belief.known_color) && card.color == card_belief.known_color
                total_remaining = sum([count for (card, count) in remaining if card.color == card_belief.known_color])
            elseif !isnothing(card_belief.known_number) && card.number == card_belief.known_number
                total_remaining = sum([count for (card, count) in remaining if card.number == card_belief.known_color])
            elseif isnothing(card_belief.known_color) && isnothing(card_belief.known_number)
                total_remaining = sum(values(remaining))
            end
            if total_remaining == 0
                probs[card] = 0
            else
                probs[card] = count / total_remaining
            end
            if card_belief.known
                if card != card_belief.known
                    probs[card] = 0
                else
                    probs[card] = 1
                end
            end
        end
    end
end

# Constructor: create a random agent for a given player in an initial game.
function RandomHanabiAgent(player_id::Int, knowledge::PlayerKnowledge)
    return RandomHanabiAgent(player_id, knowledge)
end

function choose_action(agent::RandomHanabiAgent, game::FullGameState)
    # Build the agent's current knowledge from the full game state.
    # This automatically respects information constraints (own hand is unknown,
    # other hands are fully observed).
    knowledge = init_player_knowledge(game, agent.player_id)
    public = game.public

    actions = Action[]

    # 1. Hint actions (only if info tokens are available)
    if public.info_tokens > 0
        for (receiver, hand) in knowledge.other_hands
            # Determine all colors and numbers present in the receiver's hand
            colors = unique(c.color for c in hand)
            numbers = unique(c.number for c in hand)
            for col in colors
                push!(actions, GiveHint(agent.player_id, receiver, col))
            end
            for num in numbers
                push!(actions, GiveHint(agent.player_id, receiver, num))
            end
        end
    end

    # 2. Play actions – one for each card slot in the agent's own hand
    for idx in 1:length(knowledge.own_hand)
        push!(actions, PlayCard(agent.player_id, idx))
    end

    # 3. Discard actions – only allowed if info tokens are not already maxed
    if public.info_tokens < 8
        for idx in 1:length(knowledge.own_hand)
            push!(actions, DiscardCard(agent.player_id, idx))
        end
    end

    # Safety check – should never happen in a normal game
    isempty(actions) && error("No legal actions available for player $(agent.player_id)")

    return rand(actions)
end

function update_beliefs_hint!(agent::RandomHanabiAgent, hint::CardHint, game::FullGameState)
    # simply label cards according to the hint
    label_hinted_cards!(agent.knowledge.own_hand, hint.indices, hint.attribute)
    return agent
end

function update_beliefs_action!(agent::RandomHanabiAgent, action::Action,
                                acting_player::Int, game::FullGameState)
    literal_belief_update!(agent.player_knowledge.own_hand, get_visible_cards(game, acting_player))
    for player in 1:game.player_hands.length
        if player != acting_player
            literal_belief_update!(agent.player_knowledge.theory_of_mind[player], get_visible_cards(game, [player, acting_player]))
        end
    end
    return agent
end