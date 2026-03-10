# RSA Speaker Module
# Implements pragmatic speaker (S₁) model with QUD support

include("../hanabi_game_state.jl")
include("../agent.jl")
include("rsa_listener.jl")

# =============================================================================
# ENTROPY-BASED INFORMATIVITY FUNCTIONS
# =============================================================================

"""
    compute_belief_entropy(belief::CardBelief) -> Float64

Compute the Shannon entropy of a single card belief distribution.
H(B) = -Σ p(card) log₂(p(card))
Returns entropy in bits.
"""
function compute_belief_entropy(belief::CardBelief)
    entropy = 0.0
    for (card, prob) in belief.probs
        if prob > 0
            entropy -= prob * log2(prob)
        end
    end
    return entropy
end

"""
    compute_weighted_belief_entropy(belief::CardBelief, public::PublicGameState, qud::Symbol) -> Float64

Compute weighted entropy where playable (or discardable) cards get higher weight.
This implements QUD-weighted informativity: we care more about reducing uncertainty
on cards that matter for the current goal.
"""
function compute_weighted_belief_entropy(belief::CardBelief, public::PublicGameState, qud::Symbol)
    entropy = 0.0
    total_weight = 0.0
    
    for (card, prob) in belief.probs
        if prob > 0
            # Compute weight based on QUD
            weight = 1.0
            if qud == :play && can_play_card(public, card)
                weight = 3.0  # Heavily weight playable cards
            elseif qud == :discard && is_safe_to_discard(card, public)
                weight = 2.0  # Weight safe discards moderately
            end
            
            total_weight += weight * prob
            entropy -= weight * prob * log2(prob)
        end
    end
    
    # Normalize by total weight to keep entropy scale interpretable
    if total_weight > 0
        entropy /= total_weight
    end
    
    return entropy
end

"""
    compute_hand_entropy(beliefs::Vector{CardBelief}, public::PublicGameState, qud::Symbol) -> Float64

Compute total entropy across all cards in hand (with QUD weighting).
"""
function compute_hand_entropy(beliefs::Vector{CardBelief}, public::PublicGameState, qud::Symbol)
    total_entropy = 0.0
    for belief in beliefs
        total_entropy += compute_weighted_belief_entropy(belief, public, qud)
    end
    return total_entropy
end

"""
    compute_informativity_entropy(prior_beliefs::Vector{CardBelief}, 
                                   posterior_beliefs::Vector{CardBelief},
                                   public::PublicGameState,
                                   qud::Symbol) -> Float64

Compute informativity as entropy reduction (information gain).
I(clue) = H(prior) - H(posterior)
Higher values mean the clue provides more information.
Returns information gain in bits.
"""
function compute_informativity_entropy(
    prior_beliefs::Vector{CardBelief},
    posterior_beliefs::Vector{CardBelief},
    public::PublicGameState,
    qud::Symbol
)
    prior_entropy = compute_hand_entropy(prior_beliefs, public, qud)
    posterior_entropy = compute_hand_entropy(posterior_beliefs, public, qud)
    
    # Information gain = reduction in entropy
    informativity = prior_entropy - posterior_entropy
    
    # Ensure non-negative (numerical stability)
    return max(0.0, informativity)
end

"""
    qud_play_score(clue_indices::Vector{Int}, hand::Vector{Card}, public::PublicGameState) -> Float64

QUD for "which card should my partner play?"
Scores how well the clue identifies playable cards.
Higher score = clue better identifies urgent playable cards.
"""
function qud_play_score(
    clue_indices::Vector{Int},
    hand::Vector{Card},
    public::PublicGameState
)
    score = 0.0
    
    # Find playable cards in hand
    playable_indices = Int[]
    for (idx, card) in enumerate(hand)
        if can_play_card(public, card)
            push!(playable_indices, idx)
        end
    end
    
    if isempty(playable_indices)
        # No playable cards - clue has no value for this QUD
        return 0.0
    end
    
    # Count how many playable cards are hinted
    playable_hinted = length(intersect(clue_indices, playable_indices))
    
    # Count how many non-playable cards are hinted
    non_playable_hinted = length(clue_indices) - playable_hinted
    
    # Score: reward identifying playable cards, penalize including non-playable
    score += playable_hinted * 3.0  # High reward for playable cards
    score -= non_playable_hinted * 0.5  # Mild penalty for noise
    
    # Bonus if clue identifies ALL playable cards of a certain type
    all_playable_hinted = playable_hinted == length(playable_indices)
    if all_playable_hinted
        score += 2.0
    end
    
    # Bonus for precision: if clue only points to playable cards
    if non_playable_hinted == 0 && playable_hinted > 0
        score += 1.0
    end
    
    return score
end

"""
    qud_discard_score(clue_indices::Vector{Int}, hand::Vector{Card}, public::PublicGameState) -> Float64

QUD for "which cards are safe to discard?"
Scores how well the clue identifies cards that are safe to discard.
Higher score = clue better identifies low-value/duplicate cards.
"""
function qud_discard_score(
    clue_indices::Vector{Int},
    hand::Vector{Card},
    public::PublicGameState
)
    score = 0.0
    
    # Determine which cards are safe to discard
    safe_indices = Int[]
    for (idx, card) in enumerate(hand)
        is_safe = is_safe_to_discard(card, public)
        if is_safe
            push!(safe_indices, idx)
        end
    end
    
    if isempty(safe_indices)
        # No safe discards - clue has no value for this QUD
        return 0.0
    end
    
    # Count how many safe cards are hinted
    safe_hinted = length(intersect(clue_indices, safe_indices))
    
    # Count how many critical cards are hinted
    critical_hinted = length(clue_indices) - safe_hinted
    
    # Score: reward identifying safe cards, heavily penalize including critical cards
    score += safe_hinted * 2.0
    score -= critical_hinted * 2.0  # Strong penalty for marking critical cards
    
    return score
end

"""
    is_safe_to_discard(card::Card, public::PublicGameState) -> Bool

Determine if a card is safe to discard (already played or not needed).
"""
function is_safe_to_discard(card::Card, public::PublicGameState)
    # Check if card is already played
    if card.color != :rainbow
        stack_height = get(public.played_stacks, card.color, 0)
        if card.number <= stack_height
            return true  # Already played
        end
    end
    
    # Check if card is a 1 and the stack is already past 1
    if card.number == 1
        if card.color == :rainbow
            # Rainbow 1 is safe if all stacks are past 1
            return all(v >= 1 for v in values(public.played_stacks))
        else
            return get(public.played_stacks, card.color, 0) >= 1
        end
    end
    
    # Otherwise, only safe if it's a duplicate (not implemented here - conservative)
    return false
end

"""
    enumerate_legal_clues(hand::Vector{Card}, giver_id::Int, receiver_id::Int) -> Vector{CardHint}

Enumerate all legal clues that can be given for a hand.
A clue is legal if at least one card in the hand matches the attribute.
"""
function enumerate_legal_clues(hand::Vector{Card}, giver_id::Int, receiver_id::Int)
    legal_clues = CardHint[]
    
    # Color clues
    colors = [:red, :white, :green, :blue, :yellow, :rainbow]
    for color in colors
        indices = Int[]
        for (idx, card) in enumerate(hand)
            if card.color == color
                push!(indices, idx)
            end
        end
        if !isempty(indices)
            push!(legal_clues, CardHint(giver_id, receiver_id, color, indices))
        end
    end
    
    # Number clues
    for number in 1:5
        indices = Int[]
        for (idx, card) in enumerate(hand)
            if card.number == number
                push!(indices, idx)
            end
        end
        if !isempty(indices)
            push!(legal_clues, CardHint(giver_id, receiver_id, number, indices))
        end
    end
    
    return legal_clues
end

"""
    pragmatic_speaker(partner_hand::Vector{Card}, partner_beliefs::Vector{CardBelief}, 
                      public::PublicGameState; α::Float64=1.0, qud::Symbol=:play,
                      receiver_id::Int, giver_id::Int, min_informativity::Float64=0.01) -> Dict{CardHint, Float64}

Pragmatic speaker (S₁): Chooses clues by reasoning about how the listener will interpret them.
Uses entropy-based informativity with QUD weighting.

P_S1(u | h*) = softmax(α · [entropy_reduction(u) + QUD_score(u, h*)])

Returns a probability distribution over legal clues that meet the min_informativity threshold.
Clues with information gain below min_informativity are filtered out.
"""
function pragmatic_speaker(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64 = 1.0,
    qud::Symbol = :play,
    receiver_id::Int,
    giver_id::Int,
    min_informativity::Float64 = 0.01
)
    # Enumerate all legal clues
    legal_clues = enumerate_legal_clues(partner_hand, giver_id, receiver_id)
    
    if isempty(legal_clues)
        return Dict{CardHint, Float64}()
    end
    
    # Score each clue
    clue_scores = Float64[]
    valid_clues = CardHint[]
    
    for clue in legal_clues
        # Get visible cards from receiver's perspective
        visible_cards = Vector{Card}()
        # In practice, receiver can see all other players' cards and discard pile
        append!(visible_cards, public.discard_pile)
        # Note: We don't have access to full game state here, so we approximate
        
        # Apply literal listener to see how beliefs would update
        literal_updated = literal_listener(partner_beliefs, clue, visible_cards)
        
        # Compute entropy-based informativity (information gain in bits)
        informativity = compute_informativity_entropy(
            partner_beliefs,
            literal_updated,
            public,
            qud
        )
        
        # Filter out clues with insufficient informativity (redundant hints)
        if informativity < min_informativity
            # This clue provides negligible new information - skip it
            continue
        end
        
        # Compute QUD score
        qud_score = 0.0
        if qud == :play
            qud_score = qud_play_score(clue.indices, partner_hand, public)
        elseif qud == :discard
            qud_score = qud_discard_score(clue.indices, partner_hand, public)
        end
        
        # Combined score: entropy reduction + QUD utility
        # Informativity is already in interpretable units (bits)
        # Scale QUD score to be comparable
        total_score = informativity + 0.5 * qud_score
        
        push!(valid_clues, clue)
        push!(clue_scores, total_score)
    end
    
    # If no clues meet informativity threshold, return empty distribution
    if isempty(valid_clues)
        return Dict{CardHint, Float64}()
    end
    
    # Apply softmax with rationality parameter α
    max_score = maximum(clue_scores)
    exp_scores = exp.(α .* (clue_scores .- max_score))  # Subtract max for numerical stability
    total = sum(exp_scores)
    
    clue_probs = exp_scores ./ total
    
    # Return distribution
    clue_dist = Dict{CardHint, Float64}()
    for (clue, prob) in zip(valid_clues, clue_probs)
        clue_dist[clue] = prob
    end
    
    return clue_dist
end

"""
    choose_clue_s1(partner_hand::Vector{Card}, partner_beliefs::Vector{CardBelief},
                   public::PublicGameState; α::Float64, qud::Symbol,
                   receiver_id::Int, giver_id::Int, 
                   stochastic::Bool=false, min_informativity::Float64=0.01) -> Union{GiveHint, Nothing}

Choose the best clue according to the pragmatic speaker model.
If stochastic=true, samples from P_S1; otherwise returns argmax.
Clues with information gain below min_informativity are filtered out.
"""
function choose_clue_s1(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64,
    qud::Symbol,
    receiver_id::Int,
    giver_id::Int,
    stochastic::Bool = false,
    min_informativity::Float64 = 0.01
)
    # Get speaker distribution
    clue_dist = pragmatic_speaker(
        partner_hand,
        partner_beliefs,
        public;
        α=α,
        qud=qud,
        receiver_id=receiver_id,
        giver_id=giver_id,
        min_informativity=min_informativity
    )
    
    if isempty(clue_dist)
        return nothing
    end
    
    # Choose clue
    if stochastic
        # Sample from distribution
        clues = collect(keys(clue_dist))
        probs = [clue_dist[c] for c in clues]
        
        cumsum_probs = cumsum(probs)
        r = rand()
        idx = findfirst(x -> x >= r, cumsum_probs)
        if isnothing(idx)
            idx = length(clues)
        end
        
        chosen_clue = clues[idx]
    else
        # Choose argmax
        chosen_clue = nothing
        max_prob = -Inf
        for (clue, prob) in clue_dist
            if prob > max_prob
                max_prob = prob
                chosen_clue = clue
            end
        end
    end
    
    if isnothing(chosen_clue)
        return nothing
    end
    
    # Convert CardHint to GiveHint action
    return GiveHint(chosen_clue.giver, chosen_clue.reciever, chosen_clue.attribute)
end

"""
    get_best_clue_score(partner_hand::Vector{Card}, partner_beliefs::Vector{CardBelief},
                        public::PublicGameState; α::Float64, qud::Symbol,
                        receiver_id::Int, giver_id::Int, min_informativity::Float64=0.01) -> Float64

Get the score of the best clue available (used for action selection thresholding).
Returns 0.0 if no clues meet the min_informativity threshold.
"""
function get_best_clue_score(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64,
    qud::Symbol,
    receiver_id::Int,
    giver_id::Int,
    min_informativity::Float64 = 0.01
)
    clue_dist = pragmatic_speaker(
        partner_hand,
        partner_beliefs,
        public;
        α=α,
        qud=qud,
        receiver_id=receiver_id,
        giver_id=giver_id,
        min_informativity=min_informativity
    )
    
    if isempty(clue_dist)
        return 0.0
    end
    
    # Return maximum probability (confidence in best clue)
    return maximum(values(clue_dist))
end
