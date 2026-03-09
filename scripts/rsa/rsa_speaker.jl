# RSA Speaker Module
# Implements pragmatic speaker (S₁) model with QUD support

include("../hanabi_game_state.jl")
include("../agent.jl")
include("rsa_listener.jl")

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
                      receiver_id::Int, giver_id::Int) -> Dict{CardHint, Float64}

Pragmatic speaker (S₁): Chooses clues by reasoning about how the listener will interpret them.
P_S1(u | h*) = softmax(α · [log P_L0(h* | u) + QUD_score(u, h*)])

Returns a probability distribution over legal clues.
"""
function pragmatic_speaker(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64 = 1.0,
    qud::Symbol = :play,
    receiver_id::Int,
    giver_id::Int
)
    # Enumerate all legal clues
    legal_clues = enumerate_legal_clues(partner_hand, giver_id, receiver_id)
    
    if isempty(legal_clues)
        return Dict{CardHint, Float64}()
    end
    
    # Score each clue
    clue_scores = Float64[]
    
    for clue in legal_clues
        # Compute literal listener probability: P_L0(h* | u)
        # The literal listener filters to hands consistent with the clue
        # For the true hand, this is always 1.0 if clue is truthful
        # So we focus on QUD score which measures informativity
        
        # Get visible cards from receiver's perspective
        visible_cards = Vector{Card}()
        # In practice, receiver can see all other players' cards and discard pile
        append!(visible_cards, public.discard_pile)
        # Note: We don't have access to full game state here, so we approximate
        
        # Apply literal listener to see how beliefs would update
        literal_updated = literal_listener(partner_beliefs, clue, visible_cards)
        
        # Compute informativity: how much does this clue narrow down beliefs?
        informativity = 0.0
        for (prior_belief, post_belief) in zip(partner_beliefs, literal_updated)
            # Count number of possible cards before and after
            prior_possible = sum(p > 0 for p in values(prior_belief.probs))
            post_possible = sum(p > 0 for p in values(post_belief.probs))
            
            # Higher score for greater reduction in uncertainty
            if prior_possible > 0
                reduction = (prior_possible - post_possible) / prior_possible
                informativity += reduction
            end
        end
        
        # Compute QUD score
        qud_score = 0.0
        if qud == :play
            qud_score = qud_play_score(clue.indices, partner_hand, public)
        elseif qud == :discard
            qud_score = qud_discard_score(clue.indices, partner_hand, public)
        end
        
        # Combined score: informativity + QUD utility
        total_score = informativity + qud_score
        push!(clue_scores, total_score)
    end
    
    # Apply softmax with rationality parameter α
    max_score = maximum(clue_scores)
    exp_scores = exp.(α .* (clue_scores .- max_score))  # Subtract max for numerical stability
    total = sum(exp_scores)
    
    clue_probs = exp_scores ./ total
    
    # Return distribution
    clue_dist = Dict{CardHint, Float64}()
    for (clue, prob) in zip(legal_clues, clue_probs)
        clue_dist[clue] = prob
    end
    
    return clue_dist
end

"""
    choose_clue_s1(partner_hand::Vector{Card}, partner_beliefs::Vector{CardBelief},
                   public::PublicGameState; α::Float64, qud::Symbol,
                   receiver_id::Int, giver_id::Int, 
                   stochastic::Bool=false) -> Union{GiveHint, Nothing}

Choose the best clue according to the pragmatic speaker model.
If stochastic=true, samples from P_S1; otherwise returns argmax.
"""
function choose_clue_s1(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64,
    qud::Symbol,
    receiver_id::Int,
    giver_id::Int,
    stochastic::Bool = false
)
    # Get speaker distribution
    clue_dist = pragmatic_speaker(
        partner_hand,
        partner_beliefs,
        public;
        α=α,
        qud=qud,
        receiver_id=receiver_id,
        giver_id=giver_id
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
                        receiver_id::Int, giver_id::Int) -> Float64

Get the score of the best clue available (used for action selection thresholding).
"""
function get_best_clue_score(
    partner_hand::Vector{Card},
    partner_beliefs::Vector{CardBelief},
    public::PublicGameState;
    α::Float64,
    qud::Symbol,
    receiver_id::Int,
    giver_id::Int
)
    clue_dist = pragmatic_speaker(
        partner_hand,
        partner_beliefs,
        public;
        α=α,
        qud=qud,
        receiver_id=receiver_id,
        giver_id=giver_id
    )
    
    if isempty(clue_dist)
        return 0.0
    end
    
    # Return maximum probability (confidence in best clue)
    return maximum(values(clue_dist))
end
