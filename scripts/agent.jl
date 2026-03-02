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
        
        # Calculate total remaining cards that match the current knowledge
        total_remaining = 0
        for (card, count) in remaining
            # Skip cards with zero count
            count == 0 && continue
            
            # Check if card matches known constraints
            color_match = isnothing(card_belief.known_color) || card.color == card_belief.known_color
            number_match = isnothing(card_belief.known_number) || card.number == card_belief.known_number
            
            if color_match && number_match
                total_remaining += count
            end
        end
        
        # Handle the case where no cards match (shouldn't happen in a valid game)
        if total_remaining == 0
            # If no cards match constraints, reset beliefs to uniform over all remaining cards
            total_remaining = sum(values(remaining))
            if total_remaining == 0
                # If no cards remain at all, can't update
                continue
            end
        end
        
        # Calculate probabilities
        for (card, count) in remaining
            if count == 0
                probs[card] = 0.0
                continue
            end
            
            # Check if card matches known constraints
            color_match = isnothing(card_belief.known_color) || card.color == card_belief.known_color
            number_match = isnothing(card_belief.known_number) || card.number == card_belief.known_number
            
            if color_match && number_match
                probs[card] = count / total_remaining
            else
                probs[card] = 0.0
            end
        end
        
        # Override if card is fully known
        if card_belief.known isa Card
            for card in keys(probs)
                probs[card] = (card == card_belief.known) ? 1.0 : 0.0
            end
        end
        
        card_belief.probs = probs
    end
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
    label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
    return agent
end

function update_beliefs_action!(agent::RandomHanabiAgent, action::Action,
                                acting_player::Int, game::FullGameState)
    
    # Update beliefs about own hand
    visible_cards = get_visible_cards(game, agent.player_id)
    literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    
    # Update theory of mind for other players
    for player in 1:length(game.player_hands)
        if player != agent.player_id
            if haskey(agent.player_knowledge.theory_of_mind, player)
                player_visible = get_visible_cards(game, [player, agent.player_id])
                literal_belief_update!(agent.player_knowledge.theory_of_mind[player], player_visible)
            else
                println("Warning: No theory_of_mind entry for player $player")
                # Initialize it
                player_visible = get_visible_cards(game, [player, agent.player_id])
                hand_size = length(game.player_hands[player])
                agent.player_knowledge.theory_of_mind[player] = 
                    [create_informed_belief(player_visible) for _ in 1:hand_size]
            end
        end
    end
    
    return agent
end

"""
    GreedyHanabiAgent <: AbstractHanabiAgent

A greedy agent that:
- Uses literal belief updates
- Plays cards when probability of being playable ≥ threshold
- Gives hints about unknown attributes, prioritizing playable cards
- Discards cards least likely to be playable
"""
mutable struct GreedyHanabiAgent <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
    play_threshold::Float64  # Probability threshold for playing
    # Constructor with default threshold
    function GreedyHanabiAgent(player_id::Int, knowledge::PlayerKnowledge, play_threshold::Float64 = 0.6)
        return new(player_id, knowledge, play_threshold)
    end
end

# Constructor for easy creation (matches RandomHanabiAgent interface)
function GreedyHanabiAgent(player_id::Int, knowledge::PlayerKnowledge)
    return GreedyHanabiAgent(player_id, knowledge, 0.6)
end

function choose_action(agent::GreedyHanabiAgent, game::FullGameState)
    knowledge = agent.player_knowledge
    public = game.public
    
    # Calculate play probabilities for each card in hand
    play_probs = [calculate_play_probability(knowledge.own_hand[i], public) 
                  for i in 1:length(knowledge.own_hand)]
    
    # 1. Check if any card meets play threshold
    for (idx, prob) in enumerate(play_probs)
        if prob ≥ agent.play_threshold
            return PlayCard(agent.player_id, idx)
        end
    end
    
    # 2. If info tokens available, consider giving hints
    if public.info_tokens > 0
        hint_action = choose_hint_action(agent, game)
        if !isnothing(hint_action)
            return hint_action
        end
    end
    
    # 3. If discarding is allowed, discard least playable card
    if public.info_tokens < 8
        # Find card with lowest play probability
        min_prob_idx = argmin(play_probs)
        return DiscardCard(agent.player_id, min_prob_idx)
    end
    
    # 4. Fallback: if can't discard (max tokens) and no playable cards, must hint
    # (This should only happen in edge cases)
    if public.info_tokens > 0
        # Try to give any hint, even if not optimal
        fallback_hint = choose_any_hint(agent, game)
        if !isnothing(fallback_hint)
            return fallback_hint
        end
    end
    
    # 5. Ultimate fallback: play the card with highest probability
    best_idx = argmax(play_probs)
    return PlayCard(agent.player_id, best_idx)
end

"""
    calculate_play_probability(belief::CardBelief, public::PublicGameState) -> Float64

Calculate probability that a card is playable based on current beliefs.
"""
function calculate_play_probability(belief::CardBelief, public::PublicGameState)
    prob = 0.0
    
    for (card, p) in belief.probs
        if p > 0 && can_play_card(public, card)
            prob += p
        end
    end
    
    return prob
end

"""
    choose_hint_action(agent::GreedyHanabiAgent, game::FullGameState) -> Union{GiveHint, Nothing}

Choose the best hint to give based on:
1. Prioritize revealing playable cards
2. Prefer hints about unknown attributes
3. Prefer hints that give the most new information
"""
function choose_hint_action(agent::GreedyHanabiAgent, game::FullGameState)
    best_hint = nothing
    best_score = -1.0
    
    # Consider each other player as a potential hint receiver
    for receiver in 1:length(game.player_hands)
        receiver == agent.player_id && continue
        
        # Get receiver's actual hand (for determining playable cards)
        receiver_hand = game.player_hands[receiver]
        
        # Get what receiver knows about their own hand (from agent's theory of mind)
        receiver_knowledge = agent.player_knowledge.theory_of_mind[receiver]
        
        # Find playable cards in receiver's hand
        playable_indices = Int[]
        for (i, card) in enumerate(receiver_hand)
            if can_play_card(game.public, card)
                push!(playable_indices, i)
            end
        end
        
        # Consider color hints
        colors = [:red, :white, :green, :blue, :yellow, :rainbow]
        for color in colors
            # Find indices with this color
            indices = [i for (i, card) in enumerate(receiver_hand) if card.color == color]
            isempty(indices) && continue
            
            # Check if receiver already knows this color for these cards
            already_known = all(i -> !isnothing(receiver_knowledge[i].known_color) && 
                                   receiver_knowledge[i].known_color == color, indices)
            already_known && continue
            
            # Score this hint
            score = score_hint(indices, playable_indices, receiver_knowledge)
            
            if score > best_score
                best_score = score
                best_hint = GiveHint(agent.player_id, receiver, color)
            end
        end
        
        # Consider number hints
        for number in 1:5
            indices = [i for (i, card) in enumerate(receiver_hand) if card.number == number]
            isempty(indices) && continue
            
            # Check if receiver already knows this number for these cards
            already_known = all(i -> !isnothing(receiver_knowledge[i].known_number) && 
                                   receiver_knowledge[i].known_number == number, indices)
            already_known && continue
            
            # Score this hint
            score = score_hint(indices, playable_indices, receiver_knowledge)
            
            if score > best_score
                best_score = score
                best_hint = GiveHint(agent.player_id, receiver, number)
            end
        end
    end
    
    return best_hint
end

"""
    score_hint(hint_indices::Vector{Int}, playable_indices::Vector{Int}, 
               receiver_knowledge::Vector{CardBelief}) -> Float64

Score a potential hint based on:
- How many cards it gives new information about
- Whether it reveals playable cards
- How uncertain the receiver was about those cards
"""
function score_hint(hint_indices::Vector{Int}, playable_indices::Vector{Int}, 
                    receiver_knowledge::Vector{CardBelief})
    score = 0.0
    
    for idx in hint_indices
        belief = receiver_knowledge[idx]
        
        # Base score for giving any new information
        if isnothing(belief.known_color) || isnothing(belief.known_number)
            score += 1.0
        end
        
        # Bonus for cards that are playable
        if idx in playable_indices
            score += 3.0
        end
        
        # Bonus for cards that were highly uncertain
        # (measure uncertainty by number of possible cards)
        n_possible = sum(belief.probs[card] > 0 for card in keys(belief.probs))
        if n_possible > 1
            score += 0.5 * (n_possible / 10)  # Higher uncertainty = higher bonus
        end
    end
    
    # Bonus for hinting about multiple cards
    score += 0.5 * length(hint_indices)
    
    return score
end

"""
    choose_any_hint(agent::GreedyHanabiAgent, game::FullGameState) -> Union{GiveHint, Nothing}

Fallback: give any legal hint when no good hints are found.
"""
function choose_any_hint(agent::GreedyHanabiAgent, game::FullGameState)
    for receiver in 1:length(game.player_hands)
        receiver == agent.player_id && continue
        hand = game.player_hands[receiver]
        
        # Try colors first
        colors = unique(c.color for c in hand)
        if !isempty(colors)
            return GiveHint(agent.player_id, receiver, colors[1])
        end
        
        # Then numbers
        numbers = unique(c.number for c in hand)
        if !isempty(numbers)
            return GiveHint(agent.player_id, receiver, numbers[1])
        end
    end
    return nothing
end

# Reuse the same belief update functions from RandomHanabiAgent
function update_beliefs_hint!(agent::GreedyHanabiAgent, hint::CardHint, game::FullGameState)
    if hint.reciever == agent.player_id
        # Hint was given to this agent
        label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
    else
        # Hint was given to someone else - update theory of mind
        if haskey(agent.player_knowledge.theory_of_mind, hint.reciever)
            label_hinted_cards!(agent.player_knowledge.theory_of_mind[hint.reciever], 
                               hint.indices, hint.attribute)
        end
    end
    return agent
end

function update_beliefs_action!(agent::GreedyHanabiAgent, action::Action,
                                acting_player::Int, game::FullGameState)
    # Update beliefs about own hand
    visible_cards = get_visible_cards(game, agent.player_id)
    literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    
    # Update theory of mind for other players
    for player in 1:length(game.player_hands)
        if player != agent.player_id
            if haskey(agent.player_knowledge.theory_of_mind, player)
                player_visible = get_visible_cards(game, [player, agent.player_id])
                literal_belief_update!(agent.player_knowledge.theory_of_mind[player], player_visible)
            end
        end
    end
    
    # Update public information in knowledge
    agent.player_knowledge.info_tokens = game.public.info_tokens
    agent.player_knowledge.explosion_tokens = game.public.explosion_tokens
    agent.player_knowledge.deck_size = game.public.deck_size
    agent.player_knowledge.discard_pile = copy(game.public.discard_pile)
    agent.player_knowledge.played_stacks = copy(game.public.played_stacks)
    
    return agent
end

# Helper function to find index of minimum value
function argmin(v::Vector{Float64})
    return findmin(v)[2]
end

# Helper function to find index of maximum value
function argmax(v::Vector{Float64})
    return findmax(v)[2]
end