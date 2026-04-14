include("rsa_agent.jl")
using Statistics
function play_game_with_logging(agents::Vector{<:AbstractHanabiAgent}, game::FullGameState, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)

    num_hints_given = 0
    
    for (i, agent) in enumerate(agents)
        agent.player_id = i
        agent.player_knowledge = init_player_knowledge(game, i)
    end
    
    turn = 1
    while true
        player = game.current_player
        println("Turn $turn - Player $player")
        println("  Info: $(game.public.info_tokens) | Fuse: $(game.public.explosion_tokens)")
        println("   $(game.public.played_stacks)")
        println("  Score: $(current_score(game.public))")
        
        action = choose_action(agents[player], game)
        println("  Action: $action")
        
        game = execute_action!(game, action)
        
        if action isa GiveHint
            hint = last(game.public.hint_history)
            for agent in agents
                update_beliefs_hint!(agent, hint, game)
            end
            num_hints_given += 1
        end
        # Update beliefs...
        for (i, agent) in enumerate(agents)
            update_beliefs_action!(agent, action, player, game)
        end
        
        over, reason = is_game_over(game.public)
        over && (println("Game over: $reason"); break)
        turn += 1
    end
    
    score = current_score(game.public)
    println("Final score: $score")
    println("Total turns: $turn")
    println("Total hints given: $num_hints_given")
    return score, turn, num_hints_given
end

num_simulations = 100
scores = Int[]
hints_given = Int[]
turns_taken = Int[]
num_players = 5
num_cards = 4
threshold = 0.9
rationality = 5.0  # RSA rationality parameter (alpha)
use_softmax = true
hint_cost_scaling = 1.0  # Cost sensitivity parameter for hint cost (beta)

for i in 1:num_simulations
    cur_game = init_game(num_players, num_cards)
    # Using RSAHanabiAgent with pragmatic speaker reasoning
    cur_agents = [RSAHanabiAgent(i, init_player_knowledge(cur_game, i), threshold, rationality, use_softmax, hint_cost_scaling) for i in 1:num_players]
    score, turns, hints = play_game_with_logging(cur_agents, cur_game)
    push!(scores, score)
    push!(hints_given, hints)
    push!(turns_taken, turns)
end
println("==========================")
println("Average score over 100 simulations: $(mean(scores))")
println("Maximum score over 100 simulations: $(maximum(scores))")
println("Minimum score over 100 simulations: $(minimum(scores))")
println("Standard deviation of scores over 100 simulations: $(std(scores))")
println("==========================")
println("Average hints given over 100 simulations: $(mean(hints_given))")
println("Maximum hints given over 100 simulations: $(maximum(hints_given))")
println("Minimum hints given over 100 simulations: $(minimum(hints_given))")
println("Standard deviation of hints given over 100 simulations: $(std(hints_given))")
println("==========================")
println("Average turns taken over 100 simulations: $(mean(turns_taken))")
println("Maximum turns taken over 100 simulations: $(maximum(turns_taken))")
println("Minimum turns taken over 100 simulations: $(minimum(turns_taken))")
println("Standard deviation of turns taken over 100 simulations: $(std(turns_taken))")

println("==========================")
println("$(mean(scores)) | $(round(std(scores), digits=2))")
println("$(mean(hints_given)) | $(round(std(hints_given), digits=2))")
println("$(mean(turns_taken)) | $(round(std(turns_taken), digits=2))")

println("Configuration: RSAHanabiAgent with rationality = $rationality, threshold = $threshold, players = $num_players, cards = $num_cards, hint cost scaling = $hint_cost_scaling, use_softmax = $use_softmax")