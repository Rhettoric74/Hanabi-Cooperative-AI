include("agent.jl")
using Statistics, DataFrames, StatsPlots, Plots, CSV


function compute_information_gain(before::PlayerKnowledge, after::PlayerKnowledge, player_id::Int)
    """Compute mutual information gain (reduction in entropy) from hint."""
    total_gain = 0.0
    
    before_hand = before.own_hand
    after_hand = after.own_hand
    
    for i in 1:length(before_hand)
        before_belief = before_hand[i]
        after_belief = after_hand[i]
        
        # Compute entropy before
        before_probs = before_belief.probs
        if isempty(before_probs)
            before_entropy = 0.0
        else
            before_entropy = -sum(p * log2(p) for p in values(before_probs) if p > 0)
        end
        
        # Compute entropy after
        after_probs = after_belief.probs
        if isempty(after_probs)
            after_entropy = 0.0
        else
            after_entropy = -sum(p * log2(p) for p in values(after_probs) if p > 0)
        end
        
        # Information gain = reduction in entropy
        gain = before_entropy - after_entropy
        if gain > 0
            total_gain += gain
        end
    end
    
    return total_gain
end

function record_knowledge_state(agent::AbstractHanabiAgent)
    """Record the current knowledge state for an agent"""
    return deepcopy(agent.player_knowledge)
end

function play_game_with_logging(agents::Vector{<:AbstractHanabiAgent}, game::FullGameState, seed=nothing; record_info_gain=true)
    !isnothing(seed) && Random.seed!(seed)
    
    info_gains = Float64[]
    hint_types = String[]
    hint_receivers = Int[]
    turn_numbers = Int[]
    
    for (i, agent) in enumerate(agents)
        agent.player_id = i
        agent.player_knowledge = init_player_knowledge(game, i)
    end
    
    turn = 1
    while true
        player = game.current_player
        action = choose_action(agents[player], game)
        
        if action isa GiveHint && record_info_gain
            receiver = action.receiver
            
            # Simple debug: print the receiver's hand knowledge before
            println("\n--- BEFORE HINT for player $receiver ---")
            for (i, belief) in enumerate(agents[receiver].player_knowledge.own_hand)
                println("Position $i: known_color=$(belief.known_color), known_number=$(belief.known_number)")
            end
            
            # Store the current state (deepcopy should work)
            knowledge_before = deepcopy(agents[receiver].player_knowledge)
        end
        
        game = execute_action!(game, action)
        
        if action isa GiveHint
            hint = last(game.public.hint_history)
            
            # Update beliefs
            for agent in agents
                update_beliefs_hint!(agent, hint, game)
            end
            
            if record_info_gain
                # Simple debug: print the receiver's hand knowledge after
                println("\n--- AFTER HINT for player $receiver ---")
                for (i, belief) in enumerate(agents[receiver].player_knowledge.own_hand)
                    println("Position $i: known_color=$(belief.known_color), known_number=$(belief.known_number)")
                end
                
                # Verify if anything actually changed
                knowledge_after = agents[receiver].player_knowledge
                
                # Quick check if distributions changed
                any_change = false
                for i in 1:length(knowledge_before.own_hand)
                    if knowledge_before.own_hand[i].probs != knowledge_after.own_hand[i].probs
                        any_change = true
                        println("Position $i: probabilities changed!")
                    end
                end
                
                if !any_change
                    println("WARNING: No probability changes detected!")
                    println("This suggests update_beliefs_hint! isn't modifying the receiver's knowledge")
                end
                
                info_gain = compute_information_gain(knowledge_before, knowledge_after, receiver)
                push!(info_gains, info_gain)
                println("Info gain: $info_gain bits\n")
            end
        end
        
        # Update beliefs for non-hint actions
        if !(action isa GiveHint)
            for (i, agent) in enumerate(agents)
                update_beliefs_action!(agent, action, player, game)
            end
        end
        
        over, reason = is_game_over(game.public)
        over && break
        turn += 1
    end
    
    return current_score(game.public), info_gains, hint_types, hint_receivers, turn_numbers
end


# Run simulations
num_simulations = 100
num_players = 2
num_cards = 5
threshold = 1.0

println("Running $num_simulations simulations...")
println("="^60)

scores, all_info_gains, all_hint_types, all_hint_receivers, all_turn_numbers, game_results = run_simulations_with_logging(
    num_simulations, num_players, num_cards, threshold
)

# Print statistics
println("\n" * "="^60)
println("RESULTS SUMMARY")
println("="^60)
println("Average score: $(mean(scores))")
println("Score std dev: $(std(scores))")
println("Total hints given across all games: $(length(all_info_gains))")
println("Average hints per game: $(length(all_info_gains)/num_simulations)")
if length(all_info_gains) > 0
    println("Average info gain per hint: $(mean(all_info_gains)) bits")
    println("Median info gain per hint: $(median(all_info_gains)) bits")
    println("Std dev info gain: $(std(all_info_gains)) bits")
else
    println("No hints were given across all games!")
end

# Statistics by hint type
if !isempty(all_hint_types)
    color_gains = all_info_gains[all_hint_types .== "Color"]
    number_gains = all_info_gains[all_hint_types .== "Number"]
    
    println("\nBy hint type:")
    if !isempty(color_gains)
        println("  Color hints: $(length(color_gains)) hints, Avg gain: $(mean(color_gains)) bits")
    else
        println("  Color hints: 0 hints")
    end
    if !isempty(number_gains)
        println("  Number hints: $(length(number_gains)) hints, Avg gain: $(mean(number_gains)) bits")
    else
        println("  Number hints: 0 hints")
    end
end
