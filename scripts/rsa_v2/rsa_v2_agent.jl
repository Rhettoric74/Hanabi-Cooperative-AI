include("hanabi_game_state.jl")
include("rsa_utilities.jl")

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
        
        # If both color and number are now known, we know the exact card
        if !isnothing(card_belief.known_number) && !isnothing(card_belief.known_color)
            # Find the actual card that matches both
            for (card, _) in card_belief.probs
                if card.color == card_belief.known_color && card.number == card_belief.known_number
                    card_belief.known = card  # ← Store the Card, not true
                    break
                end
            end
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
        
        # Handle the case where no cards match
        # should only happen for theory of mind updates when a player
        # that another player has been told about a card, but can see
        # the only remaining card of that color/number in the player's hand
        if total_remaining == 0
            total_remaining = sum(values(remaining))
            if total_remaining == 0
                probs[card] = 0.0
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


# function choose_action(agent::RandomHanabiAgent, game::FullGameState)
#     # Build the agent's current knowledge from the full game state.
#     # This automatically respects information constraints (own hand is unknown,
#     # other hands are fully observed).
#     knowledge = init_player_knowledge(game, agent.player_id)
#     public = game.public

#     actions = Action[]

#     # 1. Hint actions (only if info tokens are available)
#     if public.info_tokens > 0
#         for (receiver, hand) in knowledge.other_hands
#             # Determine all colors and numbers present in the receiver's hand
#             colors = unique(c.color for c in hand)
#             numbers = unique(c.number for c in hand)
#             for col in colors
#                 push!(actions, GiveHint(agent.player_id, receiver, col))
#             end
#             for num in numbers
#                 push!(actions, GiveHint(agent.player_id, receiver, num))
#             end
#         end
#     end

#     # 2. Play actions – one for each card slot in the agent's own hand
#     for idx in 1:length(knowledge.own_hand)
#         push!(actions, PlayCard(agent.player_id, idx))
#     end

#     # 3. Discard actions – only allowed if info tokens are not already maxed
#     if public.info_tokens < 8
#         for idx in 1:length(knowledge.own_hand)
#             push!(actions, DiscardCard(agent.player_id, idx))
#         end
#     end

#     # Safety check – should never happen in a normal game
#     isempty(actions) && error("No legal actions available for player $(agent.player_id)")
#     action = rand(actions)
#     if action isa PlayCard || action isa DiscardCard
#         if !isempty(game.deck)
#             agent.player_knowledge.own_hand[action.card_index].known = false
#             agent.player_knowledge.own_hand[action.card_index].known_color = nothing
#             agent.player_knowledge.own_hand[action.card_index].known_number = nothing
#         else
#             # remove beliefs about the card if there's no new cards to draw
#             deleteat!(agent.player_knowledge.own_hand, action.card_idx)
#         end
#     end

#     return action
# end

# function update_beliefs_hint!(agent::RandomHanabiAgent, hint::CardHint, game::FullGameState)
#     # simply label cards according to the hint
#     label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
#     return agent
# end

# function update_beliefs_action!(agent::RandomHanabiAgent, action::Action,
#                                 acting_player::Int, game::FullGameState)
    
#     # Update beliefs about own hand
#     visible_cards = get_visible_cards(game, agent.player_id)
#     literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    
#     # Update theory of mind for other players
#     for player in 1:length(game.player_hands)
#         if player != agent.player_id
#             if haskey(agent.player_knowledge.theory_of_mind, player)
#                 player_visible = get_visible_cards(game, [player, agent.player_id])
#                 literal_belief_update!(agent.player_knowledge.theory_of_mind[player], player_visible)
#             else
#                 println("Warning: No theory_of_mind entry for player $player")
#                 # Initialize it
#                 player_visible = get_visible_cards(game, [player, agent.player_id])
#                 hand_size = length(game.player_hands[player])
#                 agent.player_knowledge.theory_of_mind[player] = 
#                     [create_informed_belief(player_visible) for _ in 1:hand_size]
#             end
#         end
#     end
    
#     return agent
# end

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

# function choose_action(agent::GreedyHanabiAgent, game::FullGameState)
#     knowledge = agent.player_knowledge
#     public = game.public
    
#     # Calculate play probabilities for each card in hand
#     play_probs = [calculate_play_probability(knowledge.own_hand[i], public) 
#                   for i in 1:length(knowledge.own_hand)]
#     action = nothing
#     # 1. Check if any card meets play threshold
#     for (idx, prob) in enumerate(play_probs)
#         if prob ≥ agent.play_threshold
#             println(prob)
#             action = PlayCard(agent.player_id, idx)
#         end
#     end
#     # 2. If info tokens available, consider giving hints
#     if isnothing(action) && public.info_tokens > 0
#         hint_action = choose_hint_action(agent, game)
#         if !isnothing(hint_action)
#             action = hint_action
#         end
#     end
    
#     # 3. If discarding is allowed, discard least playable card
#     if isnothing(action) && public.info_tokens < 8
#         # Find card with lowest play probability
#         min_prob_idx = argmin(play_probs)
#         action = DiscardCard(agent.player_id, min_prob_idx)
#     end
    
#     # 4. Fallback: if can't discard (max tokens) and no playable cards, must hint
#     # (This should only happen in edge cases)
#     if isnothing(action) && public.info_tokens > 0
#         # Try to give any hint, even if not optimal
#         fallback_hint = choose_any_hint(agent, game)
#         if !isnothing(fallback_hint)
#             action = fallback_hint
#         end
#     end
    
#     # 5. Ultimate fallback: discard the card with the lowest probability
#     if isnothing(action)
#         worst_idx = argmin(play_probs)
#         action = DiscardCard(agent.player_id, worst_idx)
#     end
#     if action isa PlayCard || action isa DiscardCard
#         if !isempty(game.deck)
#             agent.player_knowledge.own_hand[action.card_index].known = false
#             agent.player_knowledge.own_hand[action.card_index].known_color = nothing
#             agent.player_knowledge.own_hand[action.card_index].known_number = nothing
#         else
#             deleteat!(agent.player_knowledge.own_hand, action.card_index)
#         end
#     end
#     return action
# end

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
# function update_beliefs_hint!(agent::GreedyHanabiAgent, hint::CardHint, game::FullGameState)
#     if hint.reciever == agent.player_id
#         # Hint was given to this agent
#         label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
#     else
#         # Hint was given to someone else - update theory of mind
#         if haskey(agent.player_knowledge.theory_of_mind, hint.reciever)
#             label_hinted_cards!(agent.player_knowledge.theory_of_mind[hint.reciever], 
#                                hint.indices, hint.attribute)
#         end
#     end
#     return agent
# end

# function update_beliefs_action!(agent::GreedyHanabiAgent, action::Action,
#                                 acting_player::Int, game::FullGameState)
#     # Update beliefs about own hand
#     visible_cards = get_visible_cards(game, agent.player_id)
#     literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    
#     # Update theory of mind for other players
#     for player in 1:length(game.player_hands)
#         if player != agent.player_id
#             if haskey(agent.player_knowledge.theory_of_mind, player)
#                 player_visible = get_visible_cards(game, [player, agent.player_id])
#                 literal_belief_update!(agent.player_knowledge.theory_of_mind[player], player_visible)
#             end
#         end
#     end
    
#     # Update public information in knowledge
#     agent.player_knowledge.info_tokens = game.public.info_tokens
#     agent.player_knowledge.explosion_tokens = game.public.explosion_tokens
#     agent.player_knowledge.deck_size = game.public.deck_size
#     agent.player_knowledge.discard_pile = copy(game.public.discard_pile)
#     agent.player_knowledge.played_stacks = copy(game.public.played_stacks)
    
#     return agent
# end

# Helper function to find index of minimum value
function argmin(v::Vector{Float64})
    return findmin(v)[2]
end

# Helper function to find index of maximum value
function argmax(v::Vector{Float64})
    return findmax(v)[2]
end

# ============================================================================
# RSA-BASED PRAGMATIC SPEAKER AGENT
# ============================================================================

"""
    RSAHanabiAgentV2 <: AbstractHanabiAgent

An RSA-based pragmatic speaker agent that:
- Uses literal belief updates (like GreedyHanabiAgent)
- Chooses hints using RSA pragmatic speaker reasoning (S1)
- Plays cards when probability ≥ threshold
- Discards low-probability cards

The agent differs from GreedyHanabiAgent in hint selection: instead of greedily
scoring hints by how many playable cards they reveal, it uses utility-weighted
softmax reasoning to select hints that a rational listener would interpret as
indicating important cards.
"""
mutable struct RSAHanabiAgentV2 <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
    threshold::Float64  # Probability threshold for playing
    rationality::Float64  # α parameter for softmax (higher = more deterministic)
    
    function RSAHanabiAgentV2(player_id::Int, knowledge::PlayerKnowledge,
                             threshold::Float64 = 0.6, rationality::Float64 = 1.0)
        return new(player_id, knowledge, threshold, rationality)
    end
end

"""
    compute_speaker_likelihood(receivers_hand::Vector{Card}, hint_indices::Vector{Int},
                               agent::RSAHanabiAgentV2, public::PublicGameState) -> Float64

Compute likelihood P_S(hint | cards) that this hint would be chosen to describe this set of cards.

For each card that matches the hint, compute exp(α * U(card)) where U is the speaker utility
determined by whether the card is playable, critical, or dispensable. Sum these contributions
to get the total likelihood for this hint.

The higher the utility of the hinted cards, the higher the likelihood that a rational speaker
would choose this hint. This implements the RSA S1 speaker model.
"""
function compute_speaker_likelihood(receivers_hand::Vector{Card}, hint_indices::Vector{Int},
                                    agent::RSAHanabiAgentV2, public::PublicGameState)::Float64
    score = 0.0
    played_stacks = public.played_stacks
    
    for idx in hint_indices
        card = receivers_hand[idx]
        
        # Determine card properties
        is_playable = can_play_card(public, card)
        is_critical = is_critical_card(card, played_stacks)
        is_dispensable = is_dispensable_card(card, played_stacks)
        
        # Compute utility using RSA speaker utility function
        utility = speaker_utility(card, is_playable, is_critical, is_dispensable)
        
        # Add exp(α * U) to score (higher utilities contribute more exponentially)
        score += exp(agent.rationality * utility)
    end
    
    return score
end

"""
    simulate_l0_listener(receiver_beliefs::Vector{CardBelief}, hint_attribute::Union{Symbol, Int},
                         hint_indices::Vector{Int}, public::PublicGameState)::Vector{CardBelief}

Simulate what an L0 literal listener would infer from a hint. This applies only literal belief
updates (label hinted cards + belief refresh), without pragmatic reasoning.

Used during speaker planning to estimate how much a hint helps the receiver.
Returns updated belief state for comparison with current state.
"""
function simulate_l0_listener(receiver_beliefs::Vector{CardBelief}, 
                              hint_attribute::Union{Symbol, Int},
                              hint_indices::Vector{Int}, 
                              public::PublicGameState)::Vector{CardBelief}
    # Create a copy of beliefs to simulate on (don't modify original)
    simulated_beliefs = deepcopy(receiver_beliefs)
    
    # Apply literal labeling
    label_hinted_cards!(simulated_beliefs, hint_indices, hint_attribute)
    
    # Note: We don't refresh deck composition here since we're just estimating immediate inference
    return simulated_beliefs
end

"""
    compute_playability_improvement(old_belief::CardBelief, new_belief::CardBelief,
                                    public::PublicGameState)::Float64

Compute how much a belief update improved the receiver's ability to identify playable cards.

Returns a score representing the change in probability for important cards (playable > critical > dispensable).
This is used to measure information gain from a hint.
"""
function compute_playability_improvement(old_belief::CardBelief, new_belief::CardBelief,
                                         public::PublicGameState)::Float64
    improvement = 0.0
    
    # For each possible card, compute utility-weighted probability change
    for card in keys(new_belief.probs)
        old_prob = get(old_belief.probs, card, 0.0)
        new_prob = new_belief.probs[card]
        prob_increase = max(0.0, new_prob - old_prob)  # Only count increases in confidence
        
        if prob_increase > 0.01  # Only significant changes
            # Weight by card utility so playable cards matter most
            is_playable = can_play_card(public, card)
            is_critical = is_critical_card(card, public.played_stacks)
            is_dispensable = is_dispensable_card(card, public.played_stacks)
            
            utility = speaker_utility(card, is_playable, is_critical, is_dispensable)
            
            # Information gain: probability increase weighted by utility
            improvement += prob_increase * utility
        end
    end
    
    return improvement
end

"""
    compute_information_gain(receiver_beliefs::Vector{CardBelief}, hint_indices::Vector{Int},
                             hint_attribute::Union{Symbol, Int}, 
                             public::PublicGameState)::Float64

Estimate the information gain for the receiver from this hint.

Simulates L0 listener inference, computes playability improvements for hinted cards,
and returns a bonus score. Higher bonus means the hint is more informative and helpful
for the receiver.

This bonus is added to the pragmatic speaker score to make hint selection strategic:
"Choose hints that not only highlight important cards, but significantly help the receiver
understand their importance."
"""
function compute_information_gain(receiver_beliefs::Vector{CardBelief}, 
                                   hint_indices::Vector{Int},
                                   hint_attribute::Union{Symbol, Int},
                                   public::PublicGameState)::Float64
    total_gain = 0.0
    
    # Simulate what receiver would infer (L0 literal listener)
    simulated_beliefs = simulate_l0_listener(receiver_beliefs, hint_attribute, hint_indices, public)
    
    # For each hinted card, compute improvement in playability inference
    for idx in hint_indices
        if idx <= length(receiver_beliefs)
            gain = compute_playability_improvement(receiver_beliefs[idx], simulated_beliefs[idx], public)
            total_gain += gain
        end
    end
    
    return total_gain
end

"""
    choose_hint_action_rsa_v2(agent::RSAHanabiAgentV2, game::FullGameState) -> Union{GiveHint, Nothing}

Choose a hint action using RSA-based pragmatic speaker reasoning (S1) + Theory-of-Mind.

For each receiver and each possible hint (colors and numbers):
  1. Compute base score using RSA speaker likelihood: S1_score = sum(exp(α·U(card)))
  2. Compute information gain bonus using receiver's theory-of-mind beliefs (Phase 4):
     bonus = information_gain based on how much hint helps receiver infer playability
  3. Total score = S1_score + bonus_weight * information_gain

This makes hint selection two-fold strategic:
  - S1 score: "This hint highlights important cards for me to identify"
  - Bonus: "This hint significantly helps the receiver learn about their cards"

Returns hint with highest total score. Unlike GreedyHanabiAgent which uses hard max hint
scoring, this uses utility weighting with exponential preference for impactful hints.
"""
function choose_hint_action_rsa_v2(agent::RSAHanabiAgentV2, game::FullGameState)::Union{GiveHint, Nothing}
    best_hint = nothing
    best_score = -1000.0 # Start with a very low score to ensure any valid hint will be better
    public = game.public
    
    # Weight for information gain bonus (Phase 4 parameter)
    # Higher = more weight on helping receiver; lower = more weight on speaker utility
    bonus_weight = 0.5
    
    # Consider each other player as a potential receiver
    for receiver in 1:length(game.player_hands)
        receiver == agent.player_id && continue
        
        receiver_hand = game.player_hands[receiver]
        receiver_knowledge = agent.player_knowledge.theory_of_mind[receiver]
        
        # Try all color hints
        colors = unique(c.color for c in receiver_hand)
        for color in colors
            hint_indices = [i for (i, card) in enumerate(receiver_hand) if card.color == color]
            
            # Skip if receiver already knows this color for these cards
            already_known = all(i -> !isnothing(receiver_knowledge[i].known_color) &&
                                    receiver_knowledge[i].known_color == color, hint_indices)
            already_known && continue
            
            # Compute base score using RSA speaker likelihood (S1)
            s1_score = compute_speaker_likelihood(receiver_hand, hint_indices, agent, public)
            
            # Compute information gain bonus from theory-of-mind
            # This estimates how much the hint helps the receiver infer playability
            info_gain = compute_information_gain(receiver_knowledge, hint_indices, color, public)
            
            # Combined score: pragmatic speaker (S1) + theory-of-mind bonus
            total_score = s1_score + (bonus_weight * info_gain)
            
            if total_score > best_score
                best_score = total_score
                best_hint = GiveHint(agent.player_id, receiver, color)
            end
        end
        
        # Try all number hints
        for number in 1:5
            hint_indices = [i for (i, card) in enumerate(receiver_hand) if card.number == number]
            isempty(hint_indices) && continue
            
            # Skip if receiver already knows this number for these cards
            already_known = all(i -> !isnothing(receiver_knowledge[i].known_number) &&
                                    receiver_knowledge[i].known_number == number, hint_indices)
            already_known && continue
            
            # Compute base score using RSA speaker likelihood (S1)
            s1_score = compute_speaker_likelihood(receiver_hand, hint_indices, agent, public)
            
            # Compute information gain bonus from theory-of-mind
            # This estimates how much the hint helps the receiver infer playability
            info_gain = compute_information_gain(receiver_knowledge, hint_indices, number, public)
            
            # Combined score: pragmatic speaker (S1) + theory-of-mind bonus
            total_score = s1_score + (bonus_weight * info_gain)
            
            if total_score > best_score
                best_score = total_score
                best_hint = GiveHint(agent.player_id, receiver, number)
            end
        end
    end
    
    return best_hint
end

function choose_action(agent::RSAHanabiAgentV2, game::FullGameState)
    knowledge = agent.player_knowledge
    public = game.public
    
    # Calculate play probabilities for each card in hand
    play_probs = [calculate_play_probability(knowledge.own_hand[i], public)
                  for i in 1:length(knowledge.own_hand)]
    action = nothing
    
    # 1. Check if any card meets play threshold
    for (idx, prob) in enumerate(play_probs)
        if prob ≥ agent.threshold
            action = PlayCard(agent.player_id, idx)
            break
        end
    end
    
    # 2. If info tokens available, consider giving hints (using RSA selection)
    if isnothing(action) && public.info_tokens > 0
        hint_action = choose_hint_action_rsa_v2(agent, game)
        if !isnothing(hint_action)
            action = hint_action
        end
    end
    
    # 3. If discarding is allowed, discard least playable card
    if isnothing(action) && public.info_tokens < 8
        min_prob_idx = argmin(play_probs)
        action = DiscardCard(agent.player_id, min_prob_idx)
    end
    
    # 4. Fallback: if can't discard (max tokens) and no playable cards, must hint
    if isnothing(action) && public.info_tokens > 0
        fallback_hint = choose_any_hint(agent, game)
        if !isnothing(fallback_hint)
            action = fallback_hint
        end
    end
    
    # 5. Ultimate fallback: discard the card with the lowest probability
    if isnothing(action)
        worst_idx = argmin(play_probs)
        action = DiscardCard(agent.player_id, worst_idx)
    end
    
    # Update hand beliefs after action
    if action isa PlayCard || action isa DiscardCard
        if !isempty(game.deck)
            agent.player_knowledge.own_hand[action.card_index].known = false
            agent.player_knowledge.own_hand[action.card_index].known_color = nothing
            agent.player_knowledge.own_hand[action.card_index].known_number = nothing
        else
            deleteat!(agent.player_knowledge.own_hand, action.card_index)
        end
    end
    
    return action
end

# ============================================================================
# RSA PRAGMATIC LISTENER 
# ============================================================================

"""
    pragmatic_listener_update!(card_beliefs::Vector{CardBelief}, hint_attribute::Union{Symbol, Int},
                               hint_indices::Vector{Int}, agent::RSAHanabiAgentV2, 
                               public::PublicGameState)

Update card beliefs using pragmatic listener reasoning (L1) layer on top of literal beliefs.

After literal labeling, this function computes Bayesian pragmatic update:
    belief_L1(card) ∝ belief_L0(card) * P_S(hint | card)

For each hinted card, multiply its probability distribution by the speaker likelihood:
    P_S(hint | card) = exp(α * U(card))

where U(card) is the speaker utility (playable=3.0, critical=2.0, dispensable=1.0, etc).

This implements the key RSA insight: "The speaker chose this hint intentionally,
so hinted cards are more likely to have high utility (be playable, critical, etc)."

Non-hinted cards retain their original (literal) probabilities.
After updating all hinted cards, probabilities are normalized.
"""
function pragmatic_listener_update!(card_beliefs::Vector{CardBelief}, 
                                    hint_attribute::Union{Symbol, Int},
                                    hint_indices::Vector{Int}, 
                                    agent::RSAHanabiAgentV2, 
                                    public::PublicGameState)
    # For each hinted card, apply pragmatic weighting
    for idx in hint_indices
        card_belief = card_beliefs[idx]
        
        # Compute pragmatic weight for each possible card value
        for (card, prob) in card_belief.probs
            if prob > 0
                # Check if this card matches the hint attribute
                matches_hint = false
                if hint_attribute isa Symbol  # Color hint
                    matches_hint = (card.color == hint_attribute)
                else  # Number hint
                    matches_hint = (card.number == hint_attribute)
                end
                
                if matches_hint
                    # Compute speaker utility for this card
                    is_playable = can_play_card(public, card)
                    is_critical = is_critical_card(card, public.played_stacks)
                    is_dispensable = is_dispensable_card(card, public.played_stacks)
                    
                    utility = speaker_utility(card, is_playable, is_critical, is_dispensable)
                    
                    # Pragmatic weight: exp(α * U(card))
                    # Higher utility cards get exponentially higher weighting
                    pragmatic_weight = exp(agent.rationality * utility)
                    
                    # Update belief: multiply by pragmatic weight
                    card_belief.probs[card] *= pragmatic_weight
                end
            end
        end
        
        # Normalize probabilities for this card belief
        total_prob = sum(values(card_belief.probs))
        if total_prob > 0
            for card in keys(card_belief.probs)
                card_belief.probs[card] /= total_prob
            end
        end
    end
end

function update_beliefs_hint!(agent::RSAHanabiAgentV2, hint::CardHint, game::FullGameState)
    if hint.reciever == agent.player_id
        # Hint was given to this agent - update own beliefs with literal label + pragmatic reasoning
        label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
        # Apply pragmatic listener update (L1 reasoning layer)
        pragmatic_listener_update!(agent.player_knowledge.own_hand, hint.attribute, 
                                   hint.indices, agent, game.public)
        # Refresh beliefs based on remaining deck composition
        visible_cards = get_visible_cards(game, agent.player_id)
        literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    else
        # Hint was given to someone else - update theory of mind with pragmatic reasoning (PHASE 4)
        if haskey(agent.player_knowledge.theory_of_mind, hint.reciever)
            label_hinted_cards!(agent.player_knowledge.theory_of_mind[hint.reciever],
                               hint.indices, hint.attribute)

            # Apply pragmatic listener update to theory of mind
            # This enables strategic hint selection: the agent remembers that hints are intentional,
            # so the receiver will pragmatically infer hints reveal important cards
            pragmatic_listener_update!(agent.player_knowledge.theory_of_mind[hint.reciever],
                                      hint.attribute, hint.indices, agent, game.public)
            # Refresh theory-of-mind beliefs based on visible cards
            player_visible = get_visible_cards(game, [hint.reciever, agent.player_id])
            literal_belief_update!(agent.player_knowledge.theory_of_mind[hint.reciever], player_visible)
        end
    end
    return agent
end

function update_beliefs_action!(agent::RSAHanabiAgentV2, action::Action,
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