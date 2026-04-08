include("hanabi_game_state.jl")
include("rsa_utilities.jl")
using StatsBase

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


# helper function for beliefs management 
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
    RSAHanabiAgent <: AbstractHanabiAgent

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
mutable struct RSAHanabiAgent <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
    threshold::Float64  # Probability threshold for playing
    rationality::Float64  # α parameter for softmax (higher = more deterministic)
    use_softmax::Bool  # toggle between softmax (probabilistic) and argmax (deterministic)
    beta::Float64  # Cost scaling parameter: cost = beta / info_tokens (higher = stronger token penalty)
    
    function RSAHanabiAgent(player_id::Int, knowledge::PlayerKnowledge,
                             threshold::Float64 = 0.6, rationality::Float64 = 1.0,
                             use_softmax::Bool = true, beta::Float64 = 1.0)
        return new(player_id, knowledge, threshold, rationality, use_softmax, beta)
    end
end

"""
    sample_hint_rsa(scored_hints::Vector{Tuple{GiveHint, Float64}}, rationality::Float64)::GiveHint

Sample a hint from the collection of scored hints using softmax distribution.

This implements the RSA speaker model: S1 chooses utterances (hints) probabilistically,
weighted by their utility scores with temperature parameter 1/α:

    P_S1(hint) ∝ exp(α · score(hint))

Where:
- `scored_hints`: Vector of (hint, total_score) tuples collected from all possible hints
- `rationality`: α parameter controlling the softmax temperature

The softmax sampling respects RSA theory where the speaker is rational but not
deterministic; higher-scoring hints are more likely but lower-scoring hints
can still be chosen with non-zero probability.

"""
function sample_hint_rsa(scored_hints::Vector{Tuple{GiveHint, Float64}}, rationality::Float64)::GiveHint
    if isempty(scored_hints)
        error("Cannot sample from empty hint collection")
    end
    
    # Extract scores
    scores = [score for (_, score) in scored_hints]
    
    # Compute softmax probabilities: P_i = exp(α·score_i) / Σ_j exp(α·score_j)
    # For numerical stability, use log-sum-exp trick:
    # 1. Subtract max score from all scores (doesn't change softmax probabilities)
    # 2. Compute exp(α·(score_i - score_max))
    # 3. Normalize
    
    max_score = maximum(scores)
    
    # Compute exp terms with numerical stability
    exp_terms = exp.(rationality .* (scores .- max_score))
    
    # Normalize to get probabilities
    probabilities = exp_terms ./ sum(exp_terms)
    
    # Sample index according to distribution using StatsBase
    sampled_idx = sample(Weights(probabilities))
    
    return scored_hints[sampled_idx][1]
end

"""
    compute_l0_listener_inference(receivers_hand::Vector{Card}, hint_attribute::Union{Symbol, Int},
                                  hint_indices::Vector{Int})::Dict{Card, Float64}

Compute P_L0(card | hint): the probability that a literal (non-pragmatic) listener would infer
each possible card from the hint, assuming uniform distribution over matching cards.

For cards matching the hint: P_L0(card | hint) = 1 / num_matches
For cards not matching: P_L0(card | hint) = 0

This is the L0 (literal listener) model: simple, non-pragmatic card identification.
Used by the speaker to reason: "How would a literal listener interpret this hint?"
"""
function compute_l0_listener_inference(receivers_hand::Vector{Card}, hint_attribute::Union{Symbol, Int},
                                       hint_indices::Vector{Int})::Dict{Card, Float64}
    l0_probs = Dict{Card, Float64}()
    
    # Initialize with zero probability for all cards
    for card in receivers_hand
        l0_probs[card] = 0.0
    end
    
    # Uniform probability over matching cards
    num_matches = length(hint_indices)
    if num_matches > 0
        uniform_prob = 1.0 / num_matches
        for idx in hint_indices
            l0_probs[receivers_hand[idx]] = uniform_prob
        end
    end
    
    return l0_probs
end

"""
    compute_speaker_likelihood(receivers_hand::Vector{Card}, hint_indices::Vector{Int},
                               agent::RSAHanabiAgent, public::PublicGameState, 
                               hint_attribute::Union{Symbol, Int})::Float64

Compute likelihood P_S(hint | cards) that a rational speaker would choose this hint.

This is the S1 (pragmatic speaker) model. For each card matching the hint:
  1. Compute P_L0(card | hint) - how likely the literal listener is to infer this card
  2. Compute U(card) - the utility (importance) of the card
  3. Combine: exp(α * (log P_L0 + U))
  
Intuition: Speaker prefers hints that communicate high-utility cards to a literal listener.
- If many cards match (low P_L0): hint is ambiguous, less valuable to speaker
- If few cards match (high P_L0): hint is clear, speaker will use to highlight important cards
- Higher utility cards (playable, critical) → speaker uses hints for them more

This implements proper RSA S1 reasoning with L0 foundation.
"""
function compute_speaker_likelihood(receivers_hand::Vector{Card}, hint_indices::Vector{Int},
                                    agent::RSAHanabiAgent, public::PublicGameState,
                                    hint_attribute::Union{Symbol, Int})::Float64
    score = 0.0
    played_stacks = public.played_stacks
    
    # Compute L0 literal listener inference for this hint
    l0_probs = compute_l0_listener_inference(receivers_hand, hint_attribute, hint_indices)
    
    for idx in hint_indices
        card = receivers_hand[idx]
        
        # Get L0 probability for this card (will be 1/num_matches for matching cards)
        l0_prob = l0_probs[card]
        
        # DEBUG XXX
        if l0_prob > 0
            # Convert to log space for numerical stability
            log_l0_prob = log(l0_prob)
            
            # Determine card properties
            is_playable = can_play_card(public, card)
            is_critical = is_critical_card(card, played_stacks)
            is_dispensable = is_dispensable_card(card, played_stacks)
            
            # Compute utility using RSA speaker utility function
            utility = speaker_utility(card, is_playable, is_critical, is_dispensable)
            
            # Compute cost term: C(hint) = beta / info_tokens
            # Higher cost when tokens are low, low cost when tokens are abundant
            # Encourages hinting early, discourages hinting when tokens are scarce
            cost = agent.beta / max(0.0001, Float64(public.info_tokens))
            
            # RSA S1 formula: exp(α * (log(P_L0) + U - C))
            # Combines L0 inference clarity, card importance, and token cost
            utility_with_cost = log_l0_prob + utility - cost
            s1_score = exp(agent.rationality * utility_with_cost)
            
            score += s1_score
        end
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
    choose_hint_action_rsa_v2(agent::RSAHanabiAgent, game::FullGameState) -> Union{GiveHint, Nothing}

Choose a hint action using RSA-based pragmatic speaker reasoning (S1) + Theory-of-Mind.

For each receiver and each possible hint (colors and numbers):
  1. Compute base score using RSA speaker likelihood: S1_score = sum(exp(α·U(card)))
  2. Compute information gain bonus using receiver's theory-of-mind beliefs:
     bonus = information_gain based on how much hint helps receiver infer playability
  3. Total score = S1_score + bonus_weight * information_gain

This makes hint selection two-fold strategic:
  - S1 score: "This hint highlights important cards for me to identify"
  - Bonus: "This hint significantly helps the receiver learn about their cards"

Returns hint with highest total score. Unlike GreedyHanabiAgent which uses hard max hint
scoring, this uses utility weighting with exponential preference for impactful hints.
"""
function choose_hint_action_rsa_v2(agent::RSAHanabiAgent, game::FullGameState)::Union{GiveHint, Nothing}
    # ========================================================================
    # Hint selection with RSA-aligned probability model
    # ========================================================================
    # Collects all valid hints with their RSA scores, then either:
    # - Uses SOFTMAX SAMPLING if use_softmax=true (theory-aligned)
    # - Uses ARGMAX if use_softmax=false (greedy deterministic)
    #
    # Softmax formula: P_S1(hint) ∝ exp(α · score(hint))
    # Higher scores → more likely, but lower scores still possible
    # Aligns with RSA speaker model where rationality α controls preference
    # ========================================================================
    
    scored_hints = Tuple{GiveHint, Float64}[]
    public = game.public
    
    # Weight for information gain bonus 
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
            # Incorporates L0 literal listener model + speaker utility
            s1_score = compute_speaker_likelihood(receiver_hand, hint_indices, agent, public, color)
            
            # Compute information gain bonus from theory-of-mind
            info_gain = compute_information_gain(receiver_knowledge, hint_indices, color, public)
            
            # Combined score: pragmatic speaker (S1) + theory-of-mind bonus
            total_score = s1_score + (bonus_weight * info_gain)
            
            hint = GiveHint(agent.player_id, receiver, color)
            push!(scored_hints, (hint, total_score))
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
            s1_score = compute_speaker_likelihood(receiver_hand, hint_indices, agent, public, number)
            
            # Compute information gain bonus from theory-of-mind
            info_gain = compute_information_gain(receiver_knowledge, hint_indices, number, public)
            
            # Combined score: pragmatic speaker (S1) + theory-of-mind bonus
            total_score = s1_score + (bonus_weight * info_gain)
            
            hint = GiveHint(agent.player_id, receiver, number)
            push!(scored_hints, (hint, total_score))
        end
    end
    
    # Return hint based on use_softmax flag
    if isempty(scored_hints)
        return nothing
    elseif agent.use_softmax
        # PROBABILISTIC: Softmax sampling
        # P_S1(hint) ∝ exp(α · score(hint))
        # Theory-aligned: speaker is rational (prefers high scores) but non-deterministic
        return sample_hint_rsa(scored_hints, agent.rationality)
    else
        # DETERMINISTIC: Argmax (pick best hint) Greedy approximation 
        best_hint, best_score = scored_hints[1]
        for (hint, score) in scored_hints[2:end]
            if score > best_score
                best_score = score
                best_hint = hint
            end
        end
        return best_hint
    end
end

function choose_action(agent::RSAHanabiAgent, game::FullGameState)
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
                               hint_indices::Vector{Int}, agent::RSAHanabiAgent, 
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
                                    agent::RSAHanabiAgent, 
                                    public::PublicGameState)
    # =========================================================================
    # PRAGMATIC LISTENER (L1) WITH PROPER BAYESIAN UPDATE
    # =========================================================================
    # 
    # Apply Bayes' rule: belief_L1(card) ∝ belief_L0(card)cho × P_S1(hint | card)
    # 
    # where P_S1(hint | card) = exp(α · (log(P_L0) + U(card)))
    #
    # Key insight: The listener reasons about WHY the speaker chose this hint.
    # - If speaker wanted to communicate a card, they'd pick hints that match it
    # - Speaker prefers hints that also highlight high-utility cards
    # - Therefore: hinted cards matching the attribute get upweighted
    # - Non-matching cards get zeroed out
    #
    # =========================================================================
    
    # Step 1: Compute L0 baseline
    # L0 literal listener: uniform probability over cards matching the hint
    num_matches = length(hint_indices)
    
    if num_matches == 0
        # No cards match this hint (shouldn't happen in practice)
        return
    end
    
    # Need to verify if correct DEBUG XXX
    # L0 probability for any matching card: uniform distribution
    # In log form: log(P_L0) = -log(num_matches)
    log_l0_prob = -log(num_matches)
    
    # Step 2: For each hinted card slot, apply pragmatic Bayesian update
    for idx in hint_indices
        card_belief = card_beliefs[idx]
        
        # Create new belief dictionary to avoid modifying while iterating
        updated_probs = Dict{Card, Float64}()
        
        # For each card in the belief space, compute updated probability
        for (card, prior_prob) in card_belief.probs
            # Determine if this card matches the hint attribute
            matches_hint = false
            if hint_attribute isa Symbol  # Color hint
                matches_hint = (card.color == hint_attribute)
            else  # Number hint
                matches_hint = (card.number == hint_attribute)
            end
            
            if matches_hint
                # ================================================================
                # MATCHING CARD: Apply speaker likelihood weighting
                # ================================================================
                # Compute speaker utility for this card
                is_playable = can_play_card(public, card)
                is_critical = is_critical_card(card, public.played_stacks)
                is_dispensable = is_dispensable_card(card, public.played_stacks)
                utility = speaker_utility(card, is_playable, is_critical, is_dispensable)
                
                # RSA formula: P_S1(hint | card) ∝ exp(α · (log_L0 + utility))
                # This combines two factors:
                #   1. log_L0: How clearly does this hint identify the card? 
                #              (lower = fewer matches = clearer identification)
                #   2. utility: Does the speaker want to highlight this card?
                #              (higher = more important card)
                s1_likelihood = exp(agent.rationality * (log_l0_prob + utility))
                
                # Bayesian update: new belief ∝ prior × likelihood
                # Prior is the literal L0 belief (already in card_belief.probs)
                # Likelihood is the pragmatic speaker model
                updated_probs[card] = prior_prob * s1_likelihood
                
            else
                # ================================================================
                # NON-MATCHING CARD: Zero out probability
                # ================================================================
                # Card doesn't match the hint attribute, so listener knows it's not 
                # the card the speaker was hinting about. Set probability to 0.
                # This is the literal belief update: the hint provides information
                # that eliminates non-matching possibilities.
                updated_probs[card] = 0.0
            end
        end
        
        # Step 3: Normalize probabilities to sum to 1.0
        total_prob = sum(values(updated_probs))
        
        if total_prob > 0
            # Normalize: divide each probability by total
            for card in keys(updated_probs)
                updated_probs[card] /= total_prob
            end
        else
            # Fallback: if all probabilities zeroed out (shouldn't happen), 
            # reset to uniform over matching cards
            # println("!!Warning: All probabilities zeroed out in pragmatic update. Resetting to uniform over matches.")
            for card in keys(updated_probs)
                if (hint_attribute isa Symbol ? card.color == hint_attribute : card.number == hint_attribute)
                    updated_probs[card] = 1.0 / num_matches
                end
            end
        end
        
        # Update the belief in place
        card_belief.probs = updated_probs
    end
end

function update_beliefs_hint!(agent::RSAHanabiAgent, hint::CardHint, game::FullGameState)
    if hint.reciever == agent.player_id
        # Hint was given to this agent - update own beliefs with literal label + pragmatic reasoning
        label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
        # Apply pragmatic listener update (L1 reasoning layer)
        pragmatic_listener_update!(agent.player_knowledge.own_hand, hint.attribute, 
                                   hint.indices, agent, game.public)
        # Refresh beliefs based on remaining deck composition
        visible_cards = get_visible_cards(game, agent.player_id)
        #literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
    else
        # Hint was given to someone else - update theory of mind with pragmatic reasoning 
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
            #literal_belief_update!(agent.player_knowledge.theory_of_mind[hint.reciever], player_visible)
        end
    end
    return agent
end

function update_beliefs_action!(agent::RSAHanabiAgent, action::Action,
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