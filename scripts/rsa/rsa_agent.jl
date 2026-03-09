# RSA Agent Module
# Implements RSAHanabiAgent with RSA-based action selection

include("../hanabi_game_state.jl")
include("../agent.jl")
include("rsa_listener.jl")
include("rsa_speaker.jl")

"""
    RSAHanabiAgent <: AbstractHanabiAgent

Agent that uses Rational Speech Acts framework for clue interpretation and generation.
- Uses pragmatic listener (L₁) for interpreting received clues
- Uses pragmatic speaker (S₁) for selecting clues to give
- Configurable RSA parameters and decision thresholds
"""
mutable struct RSAHanabiAgent <: AbstractHanabiAgent
    player_id::Int
    player_knowledge::PlayerKnowledge
    
    # RSA parameters
    α::Float64              # Speaker rationality (default 1.0)
    θ_play::Float64        # Play threshold (default 0.85)
    θ_discard::Float64     # Discard threshold (default 0.70)
    qud_mode::Symbol       # :dynamic, :play, or :discard
    clue_threshold::Float64 # Min S1 score to prefer clue (default 0.6)
    
    # Constructor with defaults
    function RSAHanabiAgent(
        player_id::Int,
        player_knowledge::PlayerKnowledge,
        α::Float64 = 1.0,
        θ_play::Float64 = 0.85,
        θ_discard::Float64 = 0.70,
        qud_mode::Symbol = :dynamic,
        clue_threshold::Float64 = 0.6
    )
        new(player_id, player_knowledge, α, θ_play, θ_discard, qud_mode, clue_threshold)
    end
end

"""
    select_qud(agent::RSAHanabiAgent, game::FullGameState) -> Symbol

Select the Question Under Discussion based on game state.
- :play when focusing on helping partner play cards
- :discard when focusing on clearing low-value cards (low info tokens)
"""
function select_qud(agent::RSAHanabiAgent, game::FullGameState)
    if agent.qud_mode != :dynamic
        return agent.qud_mode
    end
    
    # Dynamic QUD selection based on game state
    public = game.public
    
    # Use discard QUD when info tokens are critically low
    if public.info_tokens <= 2
        return :discard
    end
    
    # Check if there are urgent playable cards in partner hands
    has_urgent_plays = false
    for (receiver, hand) in agent.player_knowledge.other_hands
        for card in hand
            if can_play_card(public, card)
                has_urgent_plays = true
                break
            end
        end
        if has_urgent_plays
            break
        end
    end
    
    # Default to play QUD, especially if there are playable cards
    return :play
end

"""
    calculate_discard_safety(belief::CardBelief, public::PublicGameState) -> Float64

Calculate probability that a card is safe to discard.
"""
function calculate_discard_safety(belief::CardBelief, public::PublicGameState)
    prob = 0.0
    
    for (card, p) in belief.probs
        if p > 0 && is_safe_to_discard(card, public)
            prob += p
        end
    end
    
    return prob
end

"""
    choose_action(agent::RSAHanabiAgent, game::FullGameState) -> Action

Choose an action following MODIFIED RSA decision logic:
1. If any card has P_playable ≥ θ_play → Play card (PRIORITIZED)
2. Else if info tokens > 0 AND high-value clue exists → Give clue
3. Else if info tokens < 8 AND safe discard exists → Discard
4. Else if info tokens > 0 → Give any clue (fallback)
5. Else → Discard least dangerous card

Note: This modified priority encourages aggressive play when confident,
ensuring agents capitalize on received hints and make progress.
"""
function choose_action(agent::RSAHanabiAgent, game::FullGameState)
    knowledge = agent.player_knowledge
    public = game.public
    
    # Calculate play probabilities for own hand
    play_probs = [calculate_play_probability(knowledge.own_hand[i], public) 
                  for i in 1:length(knowledge.own_hand)]
    
    # Calculate discard safety for own hand
    discard_safeties = [calculate_discard_safety(knowledge.own_hand[i], public)
                        for i in 1:length(knowledge.own_hand)]
    
    action = nothing
    best_clue = nothing
    best_clue_score = 0.0
    
    # Get current QUD
    current_qud = select_qud(agent, game)
    
    # PRE-COMPUTE: Evaluate clue options (but don't commit yet)
    if public.info_tokens > 0
        # Try to find best clue for each partner
        for (receiver, hand) in knowledge.other_hands
            if receiver != agent.player_id
                receiver_beliefs = get(knowledge.theory_of_mind, receiver, nothing)
                if !isnothing(receiver_beliefs)
                    clue_score = get_best_clue_score(
                        hand,
                        receiver_beliefs,
                        public;
                        α=agent.α,
                        qud=current_qud,
                        receiver_id=receiver,
                        giver_id=agent.player_id
                    )
                    
                    if clue_score > best_clue_score
                        best_clue_score = clue_score
                        best_clue = choose_clue_s1(
                            hand,
                            receiver_beliefs,
                            public;
                            α=agent.α,
                            qud=current_qud,
                            receiver_id=receiver,
                            giver_id=agent.player_id,
                            stochastic=false
                        )
                    end
                end
            end
        end
    end
    
    # 1. PRIORITY: Play card if confident (MOVED TO FIRST)
    best_play_idx = nothing
    best_play_prob = agent.θ_play
    
    for (idx, prob) in enumerate(play_probs)
        if prob >= best_play_prob
            best_play_prob = prob
            best_play_idx = idx
        end
    end
    
    if !isnothing(best_play_idx)
        action = PlayCard(agent.player_id, best_play_idx)
    end
    
    # 2. If no confident play, give informative clue
    if isnothing(action) && public.info_tokens > 0
        if !isnothing(best_clue) && best_clue_score >= agent.clue_threshold
            action = best_clue
        end
    end
    
    # 3. If can't play or give good clue, consider discarding
    if isnothing(action) && public.info_tokens < 8
        best_discard_idx = nothing
        best_discard_safety = agent.θ_discard
        
        for (idx, safety) in enumerate(discard_safeties)
            if safety >= best_discard_safety
                best_discard_safety = safety
                best_discard_idx = idx
            end
        end
        
        if !isnothing(best_discard_idx)
            action = DiscardCard(agent.player_id, best_discard_idx)
        end
    end
    
    # 4. Fallback: give any clue if tokens available
    if isnothing(action) && public.info_tokens > 0 && !isnothing(best_clue)
        action = best_clue
    end
    
    # 5. Ultimate fallback: discard least playable card
    if isnothing(action)
        # Choose card with lowest play probability
        worst_idx = argmin(play_probs)
        action = DiscardCard(agent.player_id, worst_idx)
    end
    
    # Update beliefs about cards that will be replaced
    if action isa PlayCard || action isa DiscardCard
        if !isempty(game.deck)
            # Card will be replaced - reset beliefs for that position
            agent.player_knowledge.own_hand[action.card_index].known = false
            agent.player_knowledge.own_hand[action.card_index].known_color = nothing
            agent.player_knowledge.own_hand[action.card_index].known_number = nothing
        else
            # No replacement - remove from hand beliefs
            deleteat!(agent.player_knowledge.own_hand, action.card_index)
        end
    end
    
    return action
end

"""
    update_beliefs_hint!(agent::RSAHanabiAgent, hint::CardHint, game::FullGameState)

Update beliefs after a hint is given.
- If agent received hint: use pragmatic listener (L₁) for RSA inference
- If other player received hint: update theory of mind with literal update
"""
function update_beliefs_hint!(agent::RSAHanabiAgent, hint::CardHint, game::FullGameState)
    if hint.reciever == agent.player_id
        # This agent received the hint - use pragmatic listener
        current_qud = select_qud(agent, game)
        
        # Create speaker function that can be called by pragmatic_listener
        speaker_fn = function(hand, beliefs, pub; α, qud, receiver_id, giver_id)
            return pragmatic_speaker(
                hand,
                beliefs,
                pub;
                α=α,
                qud=qud,
                receiver_id=receiver_id,
                giver_id=giver_id
            )
        end
        
        agent.player_knowledge.own_hand = pragmatic_listener(
            agent.player_knowledge.own_hand,
            hint,
            game,
            hint.giver;
            α=agent.α,
            qud=current_qud,
            speaker_fn=speaker_fn
        )
    else
        # Another player received hint - update theory of mind
        if haskey(agent.player_knowledge.theory_of_mind, hint.reciever)
            # Use literal update for theory of mind (simpler)
            label_hinted_cards!(
                agent.player_knowledge.theory_of_mind[hint.reciever],
                hint.indices,
                hint.attribute
            )
        end
    end
    
    return agent
end

"""
    update_beliefs_action!(agent::RSAHanabiAgent, action::Action, acting_player::Int, game::FullGameState)

Update beliefs after observing any action (play, discard).
Uses literal belief update (consistency filter) - removes revealed cards from possible set.
"""
function update_beliefs_action!(agent::RSAHanabiAgent, action::Action, acting_player::Int, game::FullGameState)
    # Update beliefs about own hand based on visible cards
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

# Helper functions (reuse from baseline agent)
function argmin(v::Vector{Float64})
    return findmin(v)[2]
end

function argmax(v::Vector{Float64})
    return findmax(v)[2]
end
