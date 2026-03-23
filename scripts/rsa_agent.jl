include("agent.jl")

# ============================================================================
# RSA HANABI AGENT - Rational Speech Act Framework
# ============================================================================

"""
    RSAHanabiAgent <: AbstractHanabiAgent

A Hanabi agent that uses the Rational Speech Act (RSA) framework for
pragmatic hint selection (S1 speaker) and hint interpretation (L1 listener).

Uses Bayesian reasoning about communicative intent:
- S1 speaker: Chooses hints by simulating L0's literal interpretation and
  maximizing utility (playable > critical > dispensable > info)
- L1 listener: Interprets hints by reasoning about why the speaker chose
  this hint rather than alternatives

Parameters:
- `player_id`: Agent's player number
- `player_knowledge`: Belief state about the game
- `play_threshold`: Probability threshold for playing cards (same as Greedy)
- `alpha`: Rationality parameter for softmax (higher = more deterministic)
- `w_play`, `w_save`, `w_discard`, `w_info`: Utility weights for different
  card types
"""
mutable struct RSAHanabiAgent <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
    play_threshold::Float64
    alpha::Float64
    # Utility weights
    w_play::Float64      # Weight for playable cards
    w_save::Float64      # Weight for critical cards (5s, last copies)
    w_discard::Float64   # Weight for dispensable cards
    w_info::Float64      # Weight for general information
    
    function RSAHanabiAgent(
        player_id::Int,
        knowledge::PlayerKnowledge,
        play_threshold::Float64 = 0.6,
        alpha::Float64 = 3.0,
        w_play::Float64 = 3.0,
        w_save::Float64 = 2.0,
        w_discard::Float64 = 1.0,
        w_info::Float64 = 0.5
    )
        new(player_id, knowledge, play_threshold, alpha, 
            w_play, w_save, w_discard, w_info)
    end
end

# ============================================================================
# HELPER FUNCTIONS - Card Value Assessment
# ============================================================================

"""
    is_critical_card(card::Card, public::PublicGameState) -> Bool

Returns true if the card is critical (should be saved):
- All 5s are critical (only one per color)
- Cards that are the last remaining copy (others in discard pile)
"""
function is_critical_card(card::Card, public::PublicGameState)
    # All 5s are critical
    if card.number == 5
        return true
    end
    
    # Check if this is the last copy of this card
    # Count how many copies are in the discard pile
    discard_count = count(c -> c == card, public.discard_pile)
    
    # Get total copies that should exist
    total_copies = get(DECK_COMPOSITION, card, 0)
    
    # If all but one copy is discarded, this is critical
    return discard_count >= total_copies - 1
end

"""
    is_dispensable_card(card::Card, public::PublicGameState) -> Bool

Returns true if the card is dispensable (safe to discard):
- Cards whose number is at or below the current stack level for that color
- Rainbow cards below the minimum stack level
"""
function is_dispensable_card(card::Card, public::PublicGameState)
    if card.color == :rainbow
        # Rainbow can be played on any stack
        # Dispensable if below minimum stack level
        min_stack = minimum(values(public.played_stacks))
        return card.number <= min_stack
    else
        # Regular card is dispensable if already played or redundant
        current_stack = get(public.played_stacks, card.color, 0)
        return card.number <= current_stack
    end
end

# ============================================================================
# L0 LITERAL LISTENER - Enhanced with Negative Information
# ============================================================================

"""
    apply_negative_hint!(card_beliefs::Vector{CardBelief}, 
                        all_indices::Vector{Int},
                        hinted_indices::Vector{Int}, 
                        attribute::Union{Symbol, Int})

Apply negative information from a hint: cards at non-hinted positions
cannot have the hinted attribute. Zeroes out probabilities for cards
matching the attribute at non-hinted positions.
"""
function apply_negative_hint!(
    card_beliefs::Vector{CardBelief},
    all_indices::Vector{Int},
    hinted_indices::Vector{Int},
    attribute::Union{Symbol, Int}
)
    unhinted_indices = setdiff(all_indices, hinted_indices)
    
    for idx in unhinted_indices
        belief = card_beliefs[idx]
        new_probs = Dict{Card, Float64}()
        
        # Filter out cards matching the attribute
        for (card, prob) in belief.probs
            if attribute isa Symbol
                # Color hint - exclude cards with this color
                if card.color != attribute
                    new_probs[card] = prob
                else
                    new_probs[card] = 0.0
                end
            else  # Int - number hint
                # Number hint - exclude cards with this number
                if card.number != attribute
                    new_probs[card] = prob
                else
                    new_probs[card] = 0.0
                end
            end
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
end

"""
    simulate_L0_update!(card_beliefs::Vector{CardBelief}, 
                       hint::CardHint,
                       visible_cards::Vector{Card})

Simulate L0 literal listener belief update on a COPY of beliefs.
Applies both positive information (hinted cards) and negative information
(non-hinted cards).
"""
function simulate_L0_update!(
    card_beliefs::Vector{CardBelief},
    hint::CardHint,
    visible_cards::Vector{Card}
)
    # Apply positive information (label hinted cards)
    label_hinted_cards!(card_beliefs, hint.indices, hint.attribute)
    
    # Apply negative information (filter non-hinted cards)
    all_indices = collect(1:length(card_beliefs))
    apply_negative_hint!(card_beliefs, all_indices, hint.indices, hint.attribute)
    
    # Update probabilities based on remaining cards
    literal_belief_update!(card_beliefs, visible_cards)
end

# ============================================================================
# S1 PRAGMATIC SPEAKER - Utility-Based Hint Selection
# ============================================================================

"""
    speaker_utility(hint::GiveHint,
                   true_hand::Vector{Card},
                   l0_beliefs::Vector{CardBelief},
                   public::PublicGameState,
                   agent::RSAHanabiAgent) -> Float64

Compute utility of a hint from the speaker's perspective.
Utility = sum over cards of: (card_weight * info_gain)

Where:
- card_weight depends on card type (playable > critical > dispensable > info)
- info_gain = log(P_L0(true_card)) measures how well L0 identifies the card
"""
function speaker_utility(
    hint::GiveHint,
    true_hand::Vector{Card},
    l0_beliefs::Vector{CardBelief},
    public::PublicGameState,
    agent::RSAHanabiAgent
)
    utility = 0.0
    
    for i in 1:length(true_hand)
        true_card = true_hand[i]
        belief = l0_beliefs[i]
        
        # How well does L0 identify this card?
        p_true = get(belief.probs, true_card, 0.0)
        info_gain = p_true > 0 ? log(p_true) : -10.0  # Avoid log(0)
        
        # Determine card category and weight
        weight = if can_play_card(public, true_card)
            agent.w_play  # Playable cards are most important
        elseif is_critical_card(true_card, public)
            agent.w_save  # Critical cards should be saved
        elseif is_dispensable_card(true_card, public)
            agent.w_discard  # Dispensable cards can be discarded
        else
            agent.w_info  # General information
        end
        
        utility += weight * info_gain
    end
    
    return utility
end

"""
    choose_hint_action_rsa(agent::RSAHanabiAgent, 
                          game::FullGameState) -> Union{GiveHint, Nothing}

S1 pragmatic speaker: Choose best hint by simulating L0's interpretation
and maximizing utility via softmax.

For each possible hint:
1. Simulate L0's belief update
2. Compute utility based on how well L0 identifies important cards
3. Select hint probabilistically via softmax(alpha * utility)
"""
function choose_hint_action_rsa(agent::RSAHanabiAgent, game::FullGameState)
    candidate_hints = Tuple{GiveHint, Float64}[]  # (hint, utility) pairs
    
    # Enumerate all legal hints to all other players
    for receiver in 1:length(game.player_hands)
        receiver == agent.player_id && continue
        
        receiver_hand = game.player_hands[receiver]
        receiver_knowledge = agent.player_knowledge.theory_of_mind[receiver]
        
        # Get visible cards from receiver's perspective
        receiver_visible = get_visible_cards(game, receiver)
        
        # Try all color hints
        colors = [:red, :white, :green, :blue, :yellow, :rainbow]
        for color in colors
            indices = [i for (i, card) in enumerate(receiver_hand) if card.color == color]
            isempty(indices) && continue
            
            hint = GiveHint(agent.player_id, receiver, color)
            hint_struct = CardHint(agent.player_id, receiver, color, indices)
            
            # Simulate L0 update on a COPY of receiver's beliefs
            l0_beliefs = deepcopy(receiver_knowledge)
            simulate_L0_update!(l0_beliefs, hint_struct, receiver_visible)
            
            # Compute utility
            utility = speaker_utility(hint, receiver_hand, l0_beliefs, 
                                     game.public, agent)
            
            push!(candidate_hints, (hint, utility))
        end
        
        # Try all number hints
        for number in 1:5
            indices = [i for (i, card) in enumerate(receiver_hand) if card.number == number]
            isempty(indices) && continue
            
            hint = GiveHint(agent.player_id, receiver, number)
            hint_struct = CardHint(agent.player_id, receiver, number, indices)
            
            # Simulate L0 update on a COPY of receiver's beliefs
            l0_beliefs = deepcopy(receiver_knowledge)
            simulate_L0_update!(l0_beliefs, hint_struct, receiver_visible)
            
            # Compute utility
            utility = speaker_utility(hint, receiver_hand, l0_beliefs, 
                                     game.public, agent)
            
            push!(candidate_hints, (hint, utility))
        end
    end
    
    isempty(candidate_hints) && return nothing
    
    # Softmax selection
    # Compute exp(alpha * utility) for each hint
    scaled_utilities = [agent.alpha * u for (_, u) in candidate_hints]
    
    # Numerical stability: subtract max
    max_util = maximum(scaled_utilities)
    exp_utils = [exp(u - max_util) for u in scaled_utilities]
    total = sum(exp_utils)
    
    # Sample from the distribution
    probs = exp_utils ./ total
    
    # For now, take argmax (deterministic); could sample for stochasticity
    best_idx = argmax(probs)
    return candidate_hints[best_idx][1]
end

# ============================================================================
# L1 PRAGMATIC LISTENER - Reasoning About Speaker Intent
# ============================================================================

"""
    compute_speaker_likelihood(hint::CardHint,
                              slot_i::Int,
                              card_c::Card,
                              agent::RSAHanabiAgent,
                              game::FullGameState) -> Float64

Compute P_S1(hint | card_i = c): the likelihood that the speaker would give
this hint if the listener had card c at position slot_i.

Algorithm:
1. Construct hypothetical hand where slot_i = card_c
2. For this hand, compute utilities of all possible hints
3. Return softmax probability of the observed hint
"""
function compute_speaker_likelihood(
    hint::CardHint,
    slot_i::Int,
    card_c::Card,
    agent::RSAHanabiAgent,
    game::FullGameState
)
    # Construct hypothetical hand
    # Use expected cards from own beliefs for other slots
    hypothetical_hand = Card[]
    for i in 1:length(agent.player_knowledge.own_hand)
        if i == slot_i
            push!(hypothetical_hand, card_c)
        else
            # Use most likely card for this slot
            belief = agent.player_knowledge.own_hand[i]
            # Find card with highest probability
            most_likely_card = nothing
            max_prob = -Inf
            for (card, prob) in belief.probs
                if prob > max_prob
                    max_prob = prob
                    most_likely_card = card
                end
            end
            push!(hypothetical_hand, most_likely_card)
        end
    end
    
    # Compute utilities of all possible hints the speaker could have given
    # to this agent (from hypothetical perspective)
    hint_utilities = Float64[]
    hint_matches_observed = Bool[]
    
    visible_from_listener = get_visible_cards(game, agent.player_id)
    
    # Enumerate all hints speaker could give to this agent
    colors = [:red, :white, :green, :blue, :yellow, :rainbow]
    for color in colors
        indices = [i for (i, card) in enumerate(hypothetical_hand) if card.color == color]
        isempty(indices) && continue
        
        test_hint = CardHint(hint.giver, agent.player_id, color, indices)
        
        # Simulate L0 update
        l0_beliefs = deepcopy(agent.player_knowledge.own_hand)
        simulate_L0_update!(l0_beliefs, test_hint, visible_from_listener)
        
        # Compute utility
        utility = speaker_utility(
            GiveHint(hint.giver, agent.player_id, color),
            hypothetical_hand,
            l0_beliefs,
            game.public,
            agent
        )
        
        push!(hint_utilities, utility)
        push!(hint_matches_observed, 
              (hint.attribute == color && hint.indices == indices))
    end
    
    # Try number hints
    for number in 1:5
        indices = [i for (i, card) in enumerate(hypothetical_hand) if card.number == number]
        isempty(indices) && continue
        
        test_hint = CardHint(hint.giver, agent.player_id, number, indices)
        
        # Simulate L0 update
        l0_beliefs = deepcopy(agent.player_knowledge.own_hand)
        simulate_L0_update!(l0_beliefs, test_hint, visible_from_listener)
        
        # Compute utility
        utility = speaker_utility(
            GiveHint(hint.giver, agent.player_id, number),
            hypothetical_hand,
            l0_beliefs,
            game.public,
            agent
        )
        
        push!(hint_utilities, utility)
        push!(hint_matches_observed, 
              (hint.attribute == number && hint.indices == indices))
    end
    
    # Softmax to get P_S1(hint | hypothetical_hand)
    scaled_utilities = agent.alpha .* hint_utilities
    max_util = maximum(scaled_utilities)
    exp_utils = exp.(scaled_utilities .- max_util)
    probs = exp_utils ./ sum(exp_utils)
    
    # Return probability of observed hint
    observed_idx = findfirst(hint_matches_observed)
    return observed_idx !== nothing ? probs[observed_idx] : 0.0
end

"""
    pragmatic_listener_update!(agent::RSAHanabiAgent,
                              hint::CardHint,
                              game::FullGameState)

L1 pragmatic listener: Update beliefs by reasoning about speaker intent.

OPTIMIZED: Only applies L1 reasoning to hinted slots, uses L0 for non-hinted slots.
For hinted slots: P_L1(card_i = c | hint) ∝ P_prior(c) * P_S1(hint | card_i = c)
"""
function pragmatic_listener_update!(
    agent::RSAHanabiAgent,
    hint::CardHint,
    game::FullGameState
)
    # First apply literal labeling (positive information)
    label_hinted_cards!(agent.player_knowledge.own_hand, hint.indices, hint.attribute)
    
    # Apply negative information to non-hinted slots
    all_indices = collect(1:length(agent.player_knowledge.own_hand))
    apply_negative_hint!(agent.player_knowledge.own_hand, all_indices, 
                        hint.indices, hint.attribute)
    
    # OPTIMIZATION: Only apply L1 pragmatic reasoning to hinted slots
    for i in hint.indices
        belief = agent.player_knowledge.own_hand[i]
        new_probs = Dict{Card, Float64}()
        
        # Only consider cards matching the hint attribute (already filtered by literal update)
        for (card, prior_p) in belief.probs
            prior_p == 0 && continue
            
            # Compute speaker likelihood for this card
            speaker_like = compute_speaker_likelihood(hint, i, card, agent, game)
            
            # Bayesian update
            new_probs[card] = prior_p * speaker_like
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
    
    # Finally, update all beliefs based on remaining cards
    visible_cards = get_visible_cards(game, agent.player_id)
    literal_belief_update!(agent.player_knowledge.own_hand, visible_cards)
end

# ============================================================================
# AGENT INTERFACE IMPLEMENTATION
# ============================================================================

"""
    choose_action(agent::RSAHanabiAgent, game::FullGameState) -> Action

Choose action using RSA for hints, greedy for play/discard.

Priority:
1. Play if any card has P(playable) >= threshold
2. Give RSA hint if info tokens available
3. Discard least playable card if allowed
4. Fallback hint or discard
"""
function choose_action(agent::RSAHanabiAgent, game::FullGameState)
    knowledge = agent.player_knowledge
    public = game.public
    
    # Calculate play probabilities
    play_probs = [calculate_play_probability(knowledge.own_hand[i], public)
                  for i in 1:length(knowledge.own_hand)]
    
    # 1. Check if any card meets play threshold
    for (idx, prob) in enumerate(play_probs)
        if prob >= agent.play_threshold
            action = PlayCard(agent.player_id, idx)
            # Update beliefs about the card that will be replaced
            if !isempty(game.deck)
                agent.player_knowledge.own_hand[idx].known = false
                agent.player_knowledge.own_hand[idx].known_color = nothing
                agent.player_knowledge.own_hand[idx].known_number = nothing
            else
                deleteat!(agent.player_knowledge.own_hand, idx)
            end
            return action
        end
    end
    
    # 2. Consider RSA hint if info tokens available
    if public.info_tokens > 0
        hint_action = choose_hint_action_rsa(agent, game)
        if !isnothing(hint_action)
            return hint_action
        end
    end
    
    # 3. Discard least playable card if allowed
    if public.info_tokens < 8
        min_idx = argmin(play_probs)
        action = DiscardCard(agent.player_id, min_idx)
        if !isempty(game.deck)
            agent.player_knowledge.own_hand[min_idx].known = false
            agent.player_knowledge.own_hand[min_idx].known_color = nothing
            agent.player_knowledge.own_hand[min_idx].known_number = nothing
        else
            deleteat!(agent.player_knowledge.own_hand, min_idx)
        end
        return action
    end
    
    # 4. Fallback: forced to hint (at max tokens)
    if public.info_tokens > 0
        hint_action = choose_hint_action_rsa(agent, game)
        if !isnothing(hint_action)
            return hint_action
        end
    end
    
    # 5. Ultimate fallback: discard worst card (shouldn't happen)
    worst_idx = argmin(play_probs)
    action = DiscardCard(agent.player_id, worst_idx)
    if !isempty(game.deck)
        agent.player_knowledge.own_hand[worst_idx].known = false
        agent.player_knowledge.own_hand[worst_idx].known_color = nothing
        agent.player_knowledge.own_hand[worst_idx].known_number = nothing
    else
        deleteat!(agent.player_knowledge.own_hand, worst_idx)
    end
    return action
end

"""
    update_beliefs_hint!(agent::RSAHanabiAgent, hint::CardHint, game::FullGameState)

Update beliefs after observing a hint.
- If hint is for this agent: Use OPTIMIZED L1 pragmatic listener
- If hint is for another agent: Use literal update on theory of mind

Note: L1 listener is now optimized to only process hinted slots with L1 reasoning,
making it computationally tractable.
"""
function update_beliefs_hint!(agent::RSAHanabiAgent, hint::CardHint, game::FullGameState)
    if hint.reciever == agent.player_id
        # Hint was given to this agent - use OPTIMIZED L1 pragmatic listener
        pragmatic_listener_update!(agent, hint, game)
    else
        # Hint was given to someone else - update theory of mind (literal)
        if haskey(agent.player_knowledge.theory_of_mind, hint.reciever)
            label_hinted_cards!(
                agent.player_knowledge.theory_of_mind[hint.reciever],
                hint.indices,
                hint.attribute
            )
            # Also apply negative information
            all_indices = collect(1:length(agent.player_knowledge.theory_of_mind[hint.reciever]))
            apply_negative_hint!(
                agent.player_knowledge.theory_of_mind[hint.reciever],
                all_indices,
                hint.indices,
                hint.attribute
            )
        end
    end
    return agent
end

"""
    update_beliefs_action!(agent::RSAHanabiAgent, action::Action,
                          acting_player::Int, game::FullGameState)

Update beliefs after any action (literal belief updates).
"""
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
                literal_belief_update!(
                    agent.player_knowledge.theory_of_mind[player],
                    player_visible
                )
            end
        end
    end
    
    # Update public information
    agent.player_knowledge.info_tokens = game.public.info_tokens
    agent.player_knowledge.explosion_tokens = game.public.explosion_tokens
    agent.player_knowledge.deck_size = game.public.deck_size
    agent.player_knowledge.discard_pile = copy(game.public.discard_pile)
    agent.player_knowledge.played_stacks = copy(game.public.played_stacks)
    
    return agent
end
