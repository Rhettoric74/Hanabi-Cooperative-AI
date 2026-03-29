include("rsa_agent.jl")
using Statistics
using CSV, DataFrames, Dates

function play_game_silent(agents::Vector{<:AbstractHanabiAgent}, game::FullGameState, seed=nothing)
    """Silent version of play_game - no logging output"""
    !isnothing(seed) && Random.seed!(seed)

    num_hints_given = 0
    
    for (i, agent) in enumerate(agents)
        agent.player_id = i
        agent.player_knowledge = init_player_knowledge(game, i)
    end
    
    turn = 1
    while true
        player = game.current_player
        action = choose_action(agents[player], game)
        game = execute_action!(game, action)
        
        if action isa GiveHint
            hint = last(game.public.hint_history)
            for agent in agents
                update_beliefs_hint!(agent, hint, game)
            end
            num_hints_given += 1
        end
        
        for (i, agent) in enumerate(agents)
            update_beliefs_action!(agent, action, player, game)
        end
        
        over, reason = is_game_over(game.public)
        over && break
        turn += 1
    end
    
    score = current_score(game.public)
    return score, turn, num_hints_given
end

function run_experiment_iteration(num_players::Int, threshold::Float64, rationality::Float64, 
                                   hint_cost_scaling::Float64, num_simulations::Int)
    """Run one iteration (multiple simulations) of the experiment"""
    player_card_dict = Dict(2 => 5, 3 => 5, 4 => 4, 5 => 4)
    num_cards = player_card_dict[num_players]
    
    scores = Int[]
    hints_given = Int[]
    turns_taken = Int[]
    
    for i in 1:num_simulations
        cur_game = init_game(num_players, num_cards)
        cur_agents = [RSAHanabiAgent(i, init_player_knowledge(cur_game, i), threshold, 
                                     rationality, true, hint_cost_scaling) for i in 1:num_players]
        score, turns, hints = play_game_silent(cur_agents, cur_game)
        push!(scores, score)
        push!(hints_given, hints)
        push!(turns_taken, turns)
    end
    
    return scores, hints_given, turns_taken
end

function run_grid_search(;
    num_players_list::Vector{Int} = [2, 3, 4, 5],
    threshold_list::Vector{Float64} = [0.5, 0.6, 0.66, 0.7, 0.75, 1.0],
    rationality_list::Vector{Float64} = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    hint_cost_scaling_list::Vector{Float64} = [0.5, 1.0, 2.0],
    simulations_per_iteration::Int = 100,
    iterations_per_combo::Int = 3,
    csv_filename::String = "grid_search_results.csv"
)
    
    # Ensure output directory exists
    output_dir = dirname(csv_filename)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Create CSV header
    results_file = open(csv_filename, "w")
    header = "num_players,threshold,rationality,hint_cost_scaling,best_score," *
             "avg_score,std_score,avg_num_hints,std_num_hints,avg_num_turns,std_num_turns,timestamp\n"
    write(results_file, header)
    
    total_combos = length(num_players_list) * length(threshold_list) * 
                   length(rationality_list) * length(hint_cost_scaling_list)
    combo_count = 0
    
    for num_players in num_players_list
        for threshold in threshold_list
            for rationality in rationality_list
                for hint_cost_scaling in hint_cost_scaling_list
                    combo_count += 1
                    
                    println("\n[$combo_count/$total_combos] Testing: players=$num_players, " *
                           "threshold=$threshold, rationality=$rationality, " *
                           "hint_cost_scaling=$hint_cost_scaling")
                    
                    # Run multiple iterations for this combination
                    all_iteration_scores = Int[]
                    all_iteration_hints = Int[]
                    all_iteration_turns = Int[]
                    
                    for iteration in 1:iterations_per_combo
                        print("  Iteration $iteration/$iterations_per_combo... ")
                        scores, hints, turns = run_experiment_iteration(
                            num_players, threshold, rationality, hint_cost_scaling,
                            simulations_per_iteration
                        )
                        
                        # Store all scores/hints/turns from this iteration
                        append!(all_iteration_scores, scores)
                        append!(all_iteration_hints, hints)
                        append!(all_iteration_turns, turns)
                        
                        println("Score: $(maximum(scores))")
                    end
                    
                    # Calculate statistics across all iterations
                    best_score = maximum(all_iteration_scores)
                    avg_score = mean(all_iteration_scores)
                    std_score = std(all_iteration_scores)
                    avg_hints = mean(all_iteration_hints)
                    std_hints = std(all_iteration_hints)
                    avg_turns = mean(all_iteration_turns)
                    std_turns = std(all_iteration_turns)
                    
                    # Write result to CSV immediately
                    result_line = "$num_players,$threshold,$rationality,$hint_cost_scaling," *
                                 "$best_score,$avg_score,$std_score,$avg_hints,$std_hints," *
                                 "$avg_turns,$std_turns\n"
                    write(results_file, result_line)
                    flush(results_file)
                    
                    println("  Best Score: $best_score | Avg Score: $avg_score ± $std_score")
                    println("  Avg Hints: $avg_hints ± $std_hints | Avg Turns: $avg_turns ± $std_turns")
                end
            end
        end
    end
    
    close(results_file)
    println("\n✓ Grid search completed! Results saved to: $csv_filename")
    
    # Load and display results table
    results_df = CSV.read(csv_filename, DataFrame)
    println("\nResults Summary:")
    println(results_df)
    
    return results_df
end

# Run the grid search with default parameters
# Modify these lists to change which hyperparameters to search over
results = run_grid_search(
    num_players_list = [2, 3, 4, 5],
    threshold_list = [0.5, 0.6, 0.66, 0.7, 0.75, 1.0],
    rationality_list = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    hint_cost_scaling_list = [0, 0.5, 1.0, 2.0],
    simulations_per_iteration = 100,
    iterations_per_combo = 3,
    csv_filename = "grid_search_results.csv"
)
