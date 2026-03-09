# RSA Listener Module
# Implements literal (L₀) and pragmatic (L₁) listener models

include("../hanabi_game_state.jl")
include("../agent.jl")

"""
    literal_listener(prior_beliefs::Vector{CardBelief}, clue::CardHint, visible_cards::Vector{Card}) -> Vector{CardBelief}

Literal listener (L₀): Updates beliefs by filtering to only cards consistent with the clue.
P_L0(h | u) ∝ B(h) · 𝟙[u is truthful for h]

The clue is truthful if it correctly identifies cards with the given attribute.
"""
function literal_listener(
    prior_beliefs::Vector{CardBelief},
    clue::CardHint,
    visible_cards::Vector{Card}
)
    # Create a copy of beliefs to update
    updated_beliefs = deepcopy(prior_beliefs)
    
    # The clue tells us which indices have the attribute
    # and implicitly which indices DON'T have it
    hinted_indices = Set(clue.indices)
    attribute = clue.attribute
    
    for (idx, belief) in enumerate(updated_beliefs)
        new_probs = Dict{Card, Float64}()
        
        if idx in hinted_indices
            # This position WAS hinted - card must have the attribute
            if attribute isa Int
                # Number hint
                belief.known_number = attribute
                for (card, prob) in belief.probs
                    if card.number == attribute
                        new_probs[card] = prob
                    else
                        new_probs[card] = 0.0
                    end
                end
            else
                # Color hint
                belief.known_color = attribute
                for (card, prob) in belief.probs
                    if card.color == attribute
                        new_probs[card] = prob
                    else
                        new_probs[card] = 0.0
                    end
                end
            end
        else
            # This position was NOT hinted - card must NOT have the attribute
            if attribute isa Int
                # Number hint - this card is NOT this number
                for (card, prob) in belief.probs
                    if card.number != attribute
                        new_probs[card] = prob
                    else
                        new_probs[card] = 0.0
                    end
                end
            else
                # Color hint - this card is NOT this color
                for (card, prob) in belief.probs
                    if card.color != attribute
                        new_probs[card] = prob
                    else
                        new_probs[card] = 0.0
                    end
                end
            end
        end
        
        # Normalize probabilities
        total = sum(values(new_probs))
        if total > 0
            for card in keys(new_probs)
                new_probs[card] /= total
            end
        end
        
        belief.probs = new_probs
        
        # Check if card is now fully known
        if !isnothing(belief.known_number) && !isnothing(belief.known_color)
            for (card, prob) in belief.probs
                if prob > 0 && card.color == belief.known_color && card.number == belief.known_number
                    belief.known = card
                    break
                end
            end
        end
    end
    
    # Also apply literal belief update to account for visible cards
    literal_belief_update!(updated_beliefs, visible_cards)
    
    return updated_beliefs
end

"""
    enumerate_hand_configurations(beliefs::Vector{CardBelief}, max_samples::Int=1000) -> Vector{Vector{Card}}

Enumerate possible hand configurations consistent with current beliefs.
For efficiency, samples up to max_samples configurations.
"""
function enumerate_hand_configurations(beliefs::Vector{CardBelief}, max_samples::Int=1000)
    hand_configs = Vector{Vector{Card}}()
    
    # For each position, get cards with non-zero probability
    possible_cards_per_position = Vector{Vector{Card}}()
    probs_per_position = Vector{Vector{Float64}}()
    
    for belief in beliefs
        cards = Card[]
        probs = Float64[]
        for (card, prob) in belief.probs
            if prob > 0
                push!(cards, card)
                push!(probs, prob)
            end
        end
        push!(possible_cards_per_position, cards)
        push!(probs_per_position, probs)
    end
    
    # Sample configurations (weighted by joint probability)
    for _ in 1:max_samples
        config = Card[]
        for (idx, (cards, probs)) in enumerate(zip(possible_cards_per_position, probs_per_position))
            if isempty(cards)
                # No valid cards for this position - invalid configuration
                config = nothing
                break
            end
            # Sample based on probabilities
            if sum(probs) > 0
                normalized_probs = probs ./ sum(probs)
                sampled_idx = rand(1:length(cards))
                # Use weighted sampling
                cumsum_probs = cumsum(normalized_probs)
                r = rand()
                sampled_idx = findfirst(x -> x >= r, cumsum_probs)
                if isnothing(sampled_idx)
                    sampled_idx = length(cards)
                end
                push!(config, cards[sampled_idx])
            else
                config = nothing
                break
            end
        end
        
        if !isnothing(config) && length(config) == length(beliefs)
            push!(hand_configs, config)
        end
    end
    
    # Remove duplicates
    unique!(hand_configs)
    
    return hand_configs
end

"""
    compute_hand_probability(hand::Vector{Card}, beliefs::Vector{CardBelief}) -> Float64

Compute the probability of a specific hand configuration under current beliefs.
"""
function compute_hand_probability(hand::Vector{Card}, beliefs::Vector{CardBelief})
    prob = 1.0
    for (idx, card) in enumerate(hand)
        if idx <= length(beliefs)
            card_prob = get(beliefs[idx].probs, card, 0.0)
            prob *= card_prob
        end
    end
    return prob
end

"""
    pragmatic_listener(prior_beliefs::Vector{CardBelief}, clue::CardHint, game::FullGameState, 
                       speaker_id::Int; α::Float64=1.0, qud::Symbol=:play, speaker_fn=nothing) -> Vector{CardBelief}

Pragmatic listener (L₁): Updates beliefs by reasoning about why the speaker chose this clue.
B_new(h) ∝ B_old(h) · P_S1(u | h)

This requires computing what the speaker would have said for each possible hand configuration.
"""
function pragmatic_listener(
    prior_beliefs::Vector{CardBelief},
    clue::CardHint,
    game::FullGameState,
    speaker_id::Int;
    α::Float64 = 1.0,
    qud::Symbol = :play,
    speaker_fn = nothing  # Will be set to pragmatic_speaker from rsa_speaker.jl
)
    # First apply literal listener update
    visible_cards = get_visible_cards(game, clue.reciever)
    literal_beliefs = literal_listener(prior_beliefs, clue, visible_cards)
    
    # If no speaker function provided, return literal update
    if isnothing(speaker_fn)
        return literal_beliefs
    end
    
    # Sample possible hand configurations
    hand_configs = enumerate_hand_configurations(literal_beliefs, 500)
    
    if isempty(hand_configs)
        # Fall back to literal update if no valid configurations
        return literal_beliefs
    end
    
    # For each possible hand configuration, compute P_S1(observed_clue | hand)
    config_weights = Float64[]
    
    for hand_config in hand_configs
        # Get speaker's distribution over clues for this hand configuration
        # We need to simulate what the speaker would say if the receiver had this hand
        
        # Compute P_S1(u | h) for the observed clue u
        try
            # Create a mock belief state for this configuration
            mock_beliefs = deepcopy(prior_beliefs)
            
            # Get the speaker's clue distribution
            # speaker_fn returns Dict{CardHint, Float64}
            clue_dist = speaker_fn(
                hand_config,
                mock_beliefs,
                game.public;
                α=α,
                qud=qud,
                receiver_id=clue.reciever,
                giver_id=speaker_id
            )
            
            # Find probability of observed clue
            clue_prob = 0.0
            for (candidate_clue, prob) in clue_dist
                # Match clues by attribute and indices
                if candidate_clue.attribute == clue.attribute && 
                   Set(candidate_clue.indices) == Set(clue.indices)
                    clue_prob = prob
                    break
                end
            end
            
            # Weight this configuration by prior probability and speaker likelihood
            prior_prob = compute_hand_probability(hand_config, prior_beliefs)
            push!(config_weights, prior_prob * clue_prob)
        catch e
            # If speaker computation fails, use prior only
            prior_prob = compute_hand_probability(hand_config, prior_beliefs)
            push!(config_weights, prior_prob * 0.01)  # Small weight for error cases
        end
    end
    
    # Normalize weights
    total_weight = sum(config_weights)
    if total_weight > 0
        config_weights ./= total_weight
    else
        # Fall back to uniform if all weights are zero
        fill!(config_weights, 1.0 / length(config_weights))
    end
    
    # Update beliefs by aggregating over weighted configurations
    pragmatic_beliefs = deepcopy(literal_beliefs)
    
    for belief in pragmatic_beliefs
        new_probs = Dict{Card, Float64}()
        
        # For each possible card at this position
        for (card, _) in belief.probs
            # Sum probability across all configurations where this card appears at this position
            card_prob = 0.0
            for (config_idx, hand_config) in enumerate(hand_configs)
                belief_idx = findfirst(b -> b === belief, pragmatic_beliefs)
                if !isnothing(belief_idx) && belief_idx <= length(hand_config)
                    if hand_config[belief_idx] == card
                        card_prob += config_weights[config_idx]
                    end
                end
            end
            new_probs[card] = card_prob
        end
        
        # Normalize
        total = sum(values(new_probs))
        if total > 0
            for card in keys(new_probs)
                new_probs[card] /= total
            end
        end
        
        belief.probs = new_probs
    end
    
    return pragmatic_beliefs
end
