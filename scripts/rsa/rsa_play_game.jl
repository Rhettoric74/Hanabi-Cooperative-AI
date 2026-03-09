# RSA Play Game Module
# Run simulations with RSA agents and compare with baseline

import Pkg; Pkg.add("Gen")

include("../hanabi_game_state.jl")
include("../agent.jl")
include("rsa_listener.jl")
include("rsa_speaker.jl")
include("rsa_agent.jl")

using Statistics
using Random
using Dates

"""
    play_rsa_game(agents::Vector{RSAHanabiAgent}, game::FullGameState; verbose::Bool=false, 
                  seed=nothing, log_file::Union{String,Nothing}=nothing) -> Int

Play a single game with RSA agents and return the final score.
Similar to baseline play_game_with_logging but adapted for RSA agents.
If log_file is provided, saves detailed game log to that file.
"""
function play_rsa_game(agents::Vector{RSAHanabiAgent}, game::FullGameState; 
                      verbose::Bool=false, seed=nothing, log_file::Union{String,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
    # Initialize log content if logging
    log_content = []
    if !isnothing(log_file)
        push!(log_content, "="^80)
        push!(log_content, "INDIVIDUAL GAME LOG")
        push!(log_content, "="^80)
        push!(log_content, "Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        push!(log_content, "Seed: $(isnothing(seed) ? "random" : seed)")
        push!(log_content, "")
        push!(log_content, "GAME CONFIGURATION")
        push!(log_content, "-"^80)
        push!(log_content, "Number of players: $(length(agents))")
        push!(log_content, "Cards per player: $(length(game.player_hands[1]))")
        push!(log_content, "Starting info tokens: $(game.public.info_tokens)")
        push!(log_content, "")
        push!(log_content, "AGENT PARAMETERS")
        push!(log_content, "-"^80)
        push!(log_content, "α (speaker rationality): $(agents[1].α)")
        push!(log_content, "θ_play (play threshold): $(agents[1].θ_play)")
        push!(log_content, "θ_discard (discard threshold): $(agents[1].θ_discard)")
        push!(log_content, "QUD mode: $(agents[1].qud_mode)")
        push!(log_content, "Clue threshold: $(agents[1].clue_threshold)")
        push!(log_content, "")
        push!(log_content, "GAME TURNS")
        push!(log_content, "="^80)
    end
    
    # Initialize agents with player IDs and knowledge
    for (i, agent) in enumerate(agents)
        agent.player_id = i
        agent.player_knowledge = init_player_knowledge(game, i)
    end
    
    turn = 1
    max_turns = 100  # Safety limit
    
    while turn <= max_turns
        player = game.current_player
        
        # Log turn start
        if !isnothing(log_file)
            push!(log_content, "")
            push!(log_content, "Turn $turn - Player $player")
            push!(log_content, "-"^80)
            push!(log_content, "Info tokens: $(game.public.info_tokens) | Explosion tokens: $(game.public.explosion_tokens)")
            push!(log_content, "Played stacks: $(game.public.played_stacks)")
            push!(log_content, "Current score: $(current_score(game.public))")
            push!(log_content, "Deck size: $(game.public.deck_size)")
        end
        
        if verbose
            println("\n" * "="^60)
            println("Turn $turn - Player $player")
            println("Info tokens: $(game.public.info_tokens) | Explosion tokens: $(game.public.explosion_tokens)")
            println("Played stacks: $(game.public.played_stacks)")
            println("Score: $(current_score(game.public))")
            println("Deck size: $(game.public.deck_size)")
        end
        
        # Agent chooses action
        action = choose_action(agents[player], game)
        
        # Log action
        if !isnothing(log_file)
            push!(log_content, "Action: $action")
        end
        
        if verbose
            println("Action: $action")
        end
        
        # Execute action
        game = execute_action!(game, action)
        
        # Log result
        if !isnothing(log_file)
            if action isa PlayCard
                # Check if play was successful
                last_history = game.history[end]
                if contains(last_history, "Successful")
                    push!(log_content, "Result: ✓ Play successful")
                else
                    push!(log_content, "Result: ✗ Play failed (explosion)")
                end
            elseif action isa DiscardCard
                push!(log_content, "Result: Card discarded, info token gained")
            elseif action isa GiveHint
                hint = last(game.public.hint_history)
                push!(log_content, "Result: Clue given (indices: $(hint.indices))")
            end
        end
        
        # Update all agents' beliefs
        if action isa GiveHint
            hint = last(game.public.hint_history)
            for agent in agents
                update_beliefs_hint!(agent, hint, game)
            end
        end
        
        # Update beliefs for all agents based on action
        for (i, agent) in enumerate(agents)
            update_beliefs_action!(agent, action, player, game)
        end
        
        # Check if game is over
        over, reason = is_game_over(game.public)
        if over
            if !isnothing(log_file)
                push!(log_content, "")
                push!(log_content, "="^80)
                push!(log_content, "GAME OVER")
                push!(log_content, "="^80)
                push!(log_content, "Reason: $reason")
                push!(log_content, "Final score: $(current_score(game.public))")
                push!(log_content, "Total turns: $turn")
                push!(log_content, "Final info tokens: $(game.public.info_tokens)")
                push!(log_content, "Final explosion tokens: $(game.public.explosion_tokens)")
                push!(log_content, "")
            end
            
            if verbose
                println("\n" * "="^60)
                println("Game over: $reason")
                println("Final score: $(current_score(game.public))")
            end
            break
        end
        
        turn += 1
    end
    
    score = current_score(game.public)
    
    # Save log file if specified
    if !isnothing(log_file)
        # Add final statistics
        push!(log_content, "FINAL STATISTICS")
        push!(log_content, "-"^80)
        push!(log_content, "Score: $score / 25")
        push!(log_content, "Info tokens remaining: $(game.public.info_tokens)")
        push!(log_content, "Explosion tokens used: $(game.public.explosion_tokens)")
        push!(log_content, "Cards in discard pile: $(length(game.public.discard_pile))")
        push!(log_content, "")
        push!(log_content, "PLAYED STACKS")
        push!(log_content, "-"^80)
        for (color, height) in sort(collect(game.public.played_stacks))
            push!(log_content, "  $color: $height")
        end
        push!(log_content, "")
        push!(log_content, "="^80)
        push!(log_content, "End of game log")
        push!(log_content, "="^80)
        
        # Write to file
        open(log_file, "w") do io
            for line in log_content
                println(io, line)
            end
        end
    end
    
    return score
end

"""
    run_rsa_simulations(num_simulations::Int, num_players::Int, cards_per_player::Int;
                        α::Float64=1.0, θ_play::Float64=0.85, θ_discard::Float64=0.70,
                        qud_mode::Symbol=:dynamic, clue_threshold::Float64=0.6,
                        verbose::Bool=false, log_file::Union{String,Nothing}=nothing,
                        save_individual_games::Bool=false, experiment_name::String="rsa_default") -> Dict

Run multiple simulations with RSA agents and return statistics.
If log_file is provided, saves detailed results to that file.
If save_individual_games is true, saves each game to a separate log file in a subfolder.
"""
function run_rsa_simulations(
    num_simulations::Int,
    num_players::Int,
    cards_per_player::Int;
    α::Float64 = 1.0,
    θ_play::Float64 = 0.85,
    θ_discard::Float64 = 0.70,
    qud_mode::Symbol = :dynamic,
    clue_threshold::Float64 = 0.6,
    verbose::Bool = false,
    log_file::Union{String,Nothing} = nothing,
    save_individual_games::Bool = false,
    experiment_name::String = "rsa_default"
)
    scores = Int[]
    explosion_losses = 0
    victories = 0
    deck_runouts = 0
    
    # Create experiment subfolder if saving individual games
    individual_games_dir = nothing
    if save_individual_games
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        individual_games_dir = joinpath(dirname(@__FILE__), "logs", "individual_games", 
                                       "$(experiment_name)_$(timestamp)")
        mkpath(individual_games_dir)
        println("  Saving individual games to: $individual_games_dir")
    end
    
    println("\n" * "="^70)
    println("Running $num_simulations simulations with RSA agents")
    println("Parameters: α=$α, θ_play=$θ_play, θ_discard=$θ_discard, qud=$qud_mode")
    if save_individual_games
        println("Individual game logs: ENABLED")
    end
    println("="^70)
    
    for i in 1:num_simulations
        if !verbose && i % 10 == 0
            print(".")
            if i % 50 == 0
                println(" $i/$num_simulations")
            end
        end
        
        # Initialize game
        cur_game = init_game(num_players, cards_per_player)
        
        # Create RSA agents
        cur_agents = [
            RSAHanabiAgent(
                j,
                init_player_knowledge(cur_game, j),
                α,
                θ_play,
                θ_discard,
                qud_mode,
                clue_threshold
            ) for j in 1:num_players
        ]
        
        # Determine individual game log file
        game_log_file = nothing
        if save_individual_games
            game_log_file = joinpath(individual_games_dir, "game_$(lpad(i, 4, '0')).txt")
        end
        
        # Play game
        score = play_rsa_game(cur_agents, cur_game; verbose=verbose, log_file=game_log_file)
        push!(scores, score)
        
        # Track end conditions
        _, reason = is_game_over(cur_game.public)
        if reason == :explosion_loss
            explosion_losses += 1
        elseif reason == :victory
            victories += 1
        elseif reason == :deck_runout_loss
            deck_runouts += 1
        end
    end
    
    if !verbose
        println()
    end
    
    # Compute statistics
    mean_score = mean(scores)
    std_score = std(scores)
    min_score = minimum(scores)
    max_score = maximum(scores)
    
    # Display results
    println("\n" * "="^70)
    println("RSA Agent Results ($num_simulations games)")
    println("="^70)
    println("Mean score: $(round(mean_score, digits=2)) ± $(round(std_score, digits=2))")
    println("Min score: $min_score")
    println("Max score: $max_score")
    println("Victories (25 points): $victories ($(round(100*victories/num_simulations, digits=1))%)")
    println("Explosion losses: $explosion_losses ($(round(100*explosion_losses/num_simulations, digits=1))%)")
    println("Deck runout losses: $deck_runouts ($(round(100*deck_runouts/num_simulations, digits=1))%)")
    println("="^70)
    
    results = Dict(
        "scores" => scores,
        "mean" => mean_score,
        "std" => std_score,
        "min" => min_score,
        "max" => max_score,
        "victories" => victories,
        "explosion_losses" => explosion_losses,
        "deck_runouts" => deck_runouts
    )
    
    # Save to log file if specified
    if !isnothing(log_file)
        save_simulation_log(log_file, "RSA Agent", results, num_simulations, num_players, 
                          cards_per_player; α=α, θ_play=θ_play, θ_discard=θ_discard, 
                          qud_mode=qud_mode, clue_threshold=clue_threshold)
    end
    
    return results
end

"""
    save_simulation_log(filename::String, agent_type::String, results::Dict, 
                       num_simulations::Int, num_players::Int, cards_per_player::Int;
                       kwargs...) -> Nothing

Save simulation results to a log file.
"""
function save_simulation_log(
    filename::String, 
    agent_type::String, 
    results::Dict,
    num_simulations::Int,
    num_players::Int,
    cards_per_player::Int;
    kwargs...
)
    # Create logs directory if it doesn't exist
    log_dir = joinpath(dirname(@__FILE__), "logs")
    if !isdir(log_dir)
        mkdir(log_dir)
    end
    
    # Full path to log file
    log_path = joinpath(log_dir, filename)
    
    open(log_path, "w") do io
        # Header
        println(io, "="^80)
        println(io, "HANABI SIMULATION LOG - $agent_type")
        println(io, "="^80)
        println(io, "Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "")
        
        # Configuration
        println(io, "CONFIGURATION")
        println(io, "-"^80)
        println(io, "Number of simulations: $num_simulations")
        println(io, "Number of players: $num_players")
        println(io, "Cards per player: $cards_per_player")
        
        # Agent-specific parameters
        if !isempty(kwargs)
            println(io, "")
            println(io, "Agent Parameters:")
            for (key, value) in kwargs
                println(io, "  $key: $value")
            end
        end
        println(io, "")
        
        # Results Summary
        println(io, "RESULTS SUMMARY")
        println(io, "-"^80)
        println(io, "Mean score: $(round(results["mean"], digits=2)) ± $(round(results["std"], digits=2))")
        println(io, "Min score: $(results["min"])")
        println(io, "Max score: $(results["max"])")
        println(io, "Median score: $(round(median(results["scores"]), digits=2))")
        println(io, "")
        
        # Game outcomes
        println(io, "GAME OUTCOMES")
        println(io, "-"^80)
        victory_pct = round(100 * results["victories"] / num_simulations, digits=2)
        explosion_pct = round(100 * results["explosion_losses"] / num_simulations, digits=2)
        deck_runout_pct = round(100 * results["deck_runouts"] / num_simulations, digits=2)
        
        println(io, "Victories (score 25): $(results["victories"]) ($victory_pct%)")
        println(io, "Explosion losses: $(results["explosion_losses"]) ($explosion_pct%)")
        println(io, "Deck runout losses: $(results["deck_runouts"]) ($deck_runout_pct%)")
        println(io, "")
        
        # Score distribution
        println(io, "SCORE DISTRIBUTION")
        println(io, "-"^80)
        score_counts = Dict{Int, Int}()
        for score in results["scores"]
            score_counts[score] = get(score_counts, score, 0) + 1
        end
        
        for score in sort(collect(keys(score_counts)))
            count = score_counts[score]
            pct = round(100 * count / num_simulations, digits=1)
            bar = repeat("█", max(1, div(count * 50, num_simulations)))
            println(io, "Score $score: $count games ($pct%) $bar")
        end
        println(io, "")
        
        # Individual game scores
        println(io, "INDIVIDUAL GAME SCORES")
        println(io, "-"^80)
        for (i, score) in enumerate(results["scores"])
            if i % 10 == 1
                print(io, "Games $(lpad(i, 3))-$(lpad(min(i+9, num_simulations), 3)): ")
            end
            print(io, "$(lpad(score, 2)) ")
            if i % 10 == 0 || i == num_simulations
                println(io)
            end
        end
        println(io, "")
        
        # Statistics
        println(io, "DETAILED STATISTICS")
        println(io, "-"^80)
        println(io, "Quartiles:")
        sorted_scores = sort(results["scores"])
        q1 = sorted_scores[div(num_simulations, 4)]
        q2 = sorted_scores[div(num_simulations, 2)]
        q3 = sorted_scores[3 * div(num_simulations, 4)]
        println(io, "  Q1 (25th percentile): $q1")
        println(io, "  Q2 (50th percentile/median): $q2")
        println(io, "  Q3 (75th percentile): $q3")
        println(io, "  Interquartile range (IQR): $(q3 - q1)")
        println(io, "")
        
        # Footer
        println(io, "="^80)
        println(io, "End of log")
        println(io, "="^80)
    end
    
    println("  ✓ Log saved to: $log_path")
end

"""
    compare_agents(num_simulations::Int, num_players::Int, cards_per_player::Int;
                  save_logs::Bool=true, save_individual_games::Bool=false) -> Nothing

Compare RSA agents with baseline greedy agents. If save_logs is true, saves results to files.
If save_individual_games is true, saves each game to a separate log file.
"""
function compare_agents(num_simulations::Int, num_players::Int, cards_per_player::Int;
                       save_logs::Bool = true, save_individual_games::Bool = false)
    println("\n" * "="^70)
    println("COMPARING RSA vs BASELINE AGENTS")
    println("="^70)
    
    # Run RSA simulations
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    rsa_log_file = save_logs ? "rsa_agent_$(timestamp).txt" : nothing
    
    rsa_results = run_rsa_simulations(
        num_simulations,
        num_players,
        cards_per_player;
        α=1.0,
        θ_play=0.85,
        θ_discard=0.70,
        qud_mode=:dynamic,
        verbose=false,
        log_file=rsa_log_file,
        save_individual_games=save_individual_games,
        experiment_name="comparison_rsa"
    )
    
    # Run baseline greedy agent simulations
    println("\n" * "="^70)
    println("Running $num_simulations simulations with Greedy baseline agents")
    println("="^70)
    
    baseline_scores = Int[]
    for i in 1:num_simulations
        if i % 10 == 0
            print(".")
            if i % 50 == 0
                println(" $i/$num_simulations")
            end
        end
        
        cur_game = init_game(num_players, cards_per_player)
        cur_agents = [
            GreedyHanabiAgent(j, init_player_knowledge(cur_game, j), 0.75)
            for j in 1:num_players
        ]
        
        # Simple game loop for baseline
        for (i, agent) in enumerate(cur_agents)
            agent.player_id = i
            agent.player_knowledge = init_player_knowledge(cur_game, i)
        end
        
        turn = 1
        while turn <= 100
            player = cur_game.current_player
            action = choose_action(cur_agents[player], cur_game)
            cur_game = execute_action!(cur_game, action)
            
            if action isa GiveHint
                hint = last(cur_game.public.hint_history)
                for agent in cur_agents
                    update_beliefs_hint!(agent, hint, cur_game)
                end
            end
            
            for (i, agent) in enumerate(cur_agents)
                update_beliefs_action!(agent, action, player, cur_game)
            end
            
            over, _ = is_game_over(cur_game.public)
            if over
                break
            end
            turn += 1
        end
        
        push!(baseline_scores, current_score(cur_game.public))
    end
    println()
    
    # Compute baseline statistics
    baseline_mean = mean(baseline_scores)
    baseline_std = std(baseline_scores)
    baseline_min = minimum(baseline_scores)
    baseline_max = maximum(baseline_scores)
    
    # Count baseline outcomes (approximate based on scores)
    baseline_victories = count(s -> s == 25, baseline_scores)
    baseline_explosion_losses = 0  # Not tracked in simple loop
    baseline_deck_runouts = num_simulations - baseline_victories
    
    baseline_results = Dict(
        "scores" => baseline_scores,
        "mean" => baseline_mean,
        "std" => baseline_std,
        "min" => baseline_min,
        "max" => baseline_max,
        "victories" => baseline_victories,
        "explosion_losses" => baseline_explosion_losses,
        "deck_runouts" => baseline_deck_runouts
    )
    
    println("\n" * "="^70)
    println("Baseline Greedy Agent Results ($num_simulations games)")
    println("="^70)
    println("Mean score: $(round(baseline_mean, digits=2)) ± $(round(baseline_std, digits=2))")
    println("Min score: $baseline_min")
    println("Max score: $baseline_max")
    println("="^70)
    
    # Save baseline log
    if save_logs
        baseline_log_file = "baseline_greedy_$(timestamp).txt"
        save_simulation_log(baseline_log_file, "Baseline Greedy Agent", baseline_results,
                          num_simulations, num_players, cards_per_player;
                          threshold=0.75)
    end
    
    # Comparison
    println("\n" * "="^70)
    println("COMPARISON")
    println("="^70)
    println("RSA mean:      $(round(rsa_results["mean"], digits=2)) ± $(round(rsa_results["std"], digits=2))")
    println("Baseline mean: $(round(baseline_mean, digits=2)) ± $(round(baseline_std, digits=2))")
    
    improvement = rsa_results["mean"] - baseline_mean
    percent_improvement = 100 * improvement / baseline_mean
    
    if improvement > 0
        println("\nRSA agents score $(round(improvement, digits=2)) points higher ($(round(percent_improvement, digits=1))% improvement)")
    elseif improvement < 0
        println("\nRSA agents score $(round(abs(improvement), digits=2)) points lower ($(round(abs(percent_improvement), digits=1))% worse)")
    else
        println("\nRSA and baseline agents perform equally")
    end
    println("="^70)
    
    # Save comparison log
    if save_logs
        comparison_log_file = "comparison_$(timestamp).txt"
        save_comparison_log(comparison_log_file, rsa_results, baseline_results, 
                          num_simulations, improvement, percent_improvement)
    end
end

"""
    save_comparison_log(filename::String, rsa_results::Dict, baseline_results::Dict,
                       num_simulations::Int, improvement::Float64, percent_improvement::Float64) -> Nothing

Save comparison results between RSA and baseline agents.
"""
function save_comparison_log(
    filename::String,
    rsa_results::Dict,
    baseline_results::Dict,
    num_simulations::Int,
    improvement::Float64,
    percent_improvement::Float64
)
    log_dir = joinpath(dirname(@__FILE__), "logs")
    if !isdir(log_dir)
        mkdir(log_dir)
    end
    
    log_path = joinpath(log_dir, filename)
    
    open(log_path, "w") do io
        println(io, "="^80)
        println(io, "AGENT COMPARISON: RSA vs BASELINE GREEDY")
        println(io, "="^80)
        println(io, "Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Number of simulations: $num_simulations")
        println(io, "")
        
        # Side-by-side comparison
        println(io, "PERFORMANCE COMPARISON")
        println(io, "-"^80)
        println(io, "                              RSA Agent    Baseline Greedy")
        println(io, "-"^80)
        println(io, "Mean score:              $(lpad(round(rsa_results["mean"], digits=2), 12))  $(lpad(round(baseline_results["mean"], digits=2), 15))")
        println(io, "Std deviation:           $(lpad(round(rsa_results["std"], digits=2), 12))  $(lpad(round(baseline_results["std"], digits=2), 15))")
        println(io, "Min score:               $(lpad(rsa_results["min"], 12))  $(lpad(baseline_results["min"], 15))")
        println(io, "Max score:               $(lpad(rsa_results["max"], 12))  $(lpad(baseline_results["max"], 15))")
        println(io, "Victories:               $(lpad(rsa_results["victories"], 12))  $(lpad(baseline_results["victories"], 15))")
        println(io, "")
        
        # Improvement analysis
        println(io, "IMPROVEMENT ANALYSIS")
        println(io, "-"^80)
        if improvement > 0
            println(io, "✓ RSA agents score $(round(improvement, digits=2)) points higher")
            println(io, "✓ Improvement: $(round(percent_improvement, digits=1))%")
        elseif improvement < 0
            println(io, "✗ RSA agents score $(round(abs(improvement), digits=2)) points lower")
            println(io, "✗ Performance: $(round(percent_improvement, digits=1))%")
        else
            println(io, "= Both agents perform equally")
        end
        println(io, "")
        
        # Score distribution comparison
        println(io, "SCORE DISTRIBUTION COMPARISON")
        println(io, "-"^80)
        
        all_scores = sort(unique(vcat(rsa_results["scores"], baseline_results["scores"])))
        
        println(io, "Score    RSA Count    Baseline Count")
        println(io, "-"^40)
        for score in all_scores
            rsa_count = count(s -> s == score, rsa_results["scores"])
            baseline_count = count(s -> s == score, baseline_results["scores"])
            println(io, "$(lpad(score, 3))      $(lpad(rsa_count, 9))    $(lpad(baseline_count, 14))")
        end
        println(io, "")
        
        println(io, "="^80)
        println(io, "End of comparison log")
        println(io, "="^80)
    end
    
    println("  ✓ Comparison log saved to: $log_path")
end

"""
    test_parameter_sensitivity(; save_logs::Bool=true, save_individual_games::Bool=false) -> Nothing

Test different parameter settings to understand their impact.
If save_logs is true, saves results to a file.
If save_individual_games is true, saves each game to separate log files organized by parameter.
"""
function test_parameter_sensitivity(; save_logs::Bool = true, save_individual_games::Bool = false)
    num_simulations = 50  # Smaller for parameter sweep
    num_players = 5
    cards_per_player = 5
    
    println("\n" * "="^70)
    println("PARAMETER SENSITIVITY ANALYSIS")
    println("="^70)
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Collect all results for logging
    all_param_results = []
    
    # Test different α values
    println("\nTesting speaker rationality (α):")
    for α in [0.5, 1.0, 1.5, 2.0]
        log_file = save_logs ? "param_alpha_$(α)_$(timestamp).txt" : nothing
        exp_name = "param_alpha_$(α)"
        results = run_rsa_simulations(
            num_simulations,
            num_players,
            cards_per_player;
            α=α,
            θ_play=0.85,
            θ_discard=0.70,
            qud_mode=:dynamic,
            verbose=false,
            log_file=log_file,
            save_individual_games=save_individual_games,
            experiment_name=exp_name
        )
        println("  α=$α: mean=$(round(results["mean"], digits=2))")
        push!(all_param_results, ("α", α, results))
    end
    
    # Test different play thresholds
    println("\nTesting play threshold (θ_play):")
    for θ_play in [0.70, 0.80, 0.85, 0.90]
        log_file = save_logs ? "param_theta_play_$(θ_play)_$(timestamp).txt" : nothing
        exp_name = "param_theta_play_$(θ_play)"
        results = run_rsa_simulations(
            num_simulations,
            num_players,
            cards_per_player;
            α=1.0,
            θ_play=θ_play,
            θ_discard=0.70,
            qud_mode=:dynamic,
            verbose=false,
            log_file=log_file,
            save_individual_games=save_individual_games,
            experiment_name=exp_name
        )
        println("  θ_play=$θ_play: mean=$(round(results["mean"], digits=2))")
        push!(all_param_results, ("θ_play", θ_play, results))
    end
    
    # Test different QUD modes
    println("\nTesting QUD modes:")
    for qud in [:play, :discard, :dynamic]
        log_file = save_logs ? "param_qud_$(qud)_$(timestamp).txt" : nothing
        exp_name = "param_qud_$(qud)"
        results = run_rsa_simulations(
            num_simulations,
            num_players,
            cards_per_player;
            α=1.0,
            θ_play=0.85,
            θ_discard=0.70,
            qud_mode=qud,
            verbose=false,
            log_file=log_file,
            save_individual_games=save_individual_games,
            experiment_name=exp_name
        )
        println("  qud=$qud: mean=$(round(results["mean"], digits=2))")
        push!(all_param_results, ("qud", qud, results))
    end
    
    # Save summary log
    if save_logs
        save_parameter_sensitivity_summary("param_sensitivity_summary_$(timestamp).txt",
                                          all_param_results, num_simulations)
    end
end

"""
    save_parameter_sensitivity_summary(filename::String, results::Vector, num_simulations::Int) -> Nothing

Save parameter sensitivity analysis summary to a log file.
"""
function save_parameter_sensitivity_summary(
    filename::String,
    results::Vector,
    num_simulations::Int
)
    log_dir = joinpath(dirname(@__FILE__), "logs")
    if !isdir(log_dir)
        mkdir(log_dir)
    end
    
    log_path = joinpath(log_dir, filename)
    
    open(log_path, "w") do io
        println(io, "="^80)
        println(io, "PARAMETER SENSITIVITY ANALYSIS SUMMARY")
        println(io, "="^80)
        println(io, "Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Number of simulations per parameter: $num_simulations")
        println(io, "")
        
        # Group by parameter type
        param_groups = Dict{String, Vector}()
        for (param_name, param_value, result) in results
            if !haskey(param_groups, param_name)
                param_groups[param_name] = []
            end
            push!(param_groups[param_name], (param_value, result))
        end
        
        # Print each parameter group
        for param_name in sort(collect(keys(param_groups)))
            println(io, uppercase(param_name) * " PARAMETER ANALYSIS")
            println(io, "-"^80)
            println(io, "Value      Mean Score    Std Dev    Min    Max    Victories")
            println(io, "-"^80)
            
            for (value, result) in sort(param_groups[param_name], by=x->x[1])
                mean_str = lpad(round(result["mean"], digits=2), 10)
                std_str = lpad(round(result["std"], digits=2), 9)
                min_str = lpad(result["min"], 6)
                max_str = lpad(result["max"], 6)
                vic_str = lpad(result["victories"], 12)
                println(io, "$(rpad(string(value), 9)) $mean_str  $std_str  $min_str  $max_str  $vic_str")
            end
            
            # Find best parameter value
            best_idx = argmax([r[2]["mean"] for r in param_groups[param_name]])
            best_value, best_result = param_groups[param_name][best_idx]
            println(io, "")
            println(io, "Best $param_name value: $best_value (mean score: $(round(best_result["mean"], digits=2)))")
            println(io, "")
        end
        
        println(io, "="^80)
        println(io, "End of parameter sensitivity summary")
        println(io, "="^80)
    end
    
    println("  ✓ Parameter sensitivity summary saved to: $log_path")
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("RSA Hanabi Agent - Simulation Script")
    println("=====================================\n")
    
    # Configuration
    num_simulations = 100
    num_players = 5
    cards_per_player = 5
    
    # Run main comparison
    compare_agents(num_simulations, num_players, cards_per_player,save_individual_games=true)
    
    # Optional: Run parameter sensitivity analysis
    # Uncomment to test different parameters
    # test_parameter_sensitivity()
    
    println("\nSimulations complete!")
end
