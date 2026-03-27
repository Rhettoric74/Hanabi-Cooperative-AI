include("rsa_v2_agent.jl")
using Statistics
function play_game_with_logging(agents::Vector{<:AbstractHanabiAgent}, game::FullGameState, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
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
    return score
end

num_simulations = 100
scores = Int[]
num_players = 4
num_cards = 4
threshold = 0.75
rationality = 3.0  # RSA rationality parameter for Phase 2 testing

for i in 1:num_simulations
    cur_game = init_game(num_players, num_cards)
    # Phase 2: Using RSAHanabiAgentV2 with pragmatic speaker reasoning
    cur_agents = [RSAHanabiAgentV2(i, init_player_knowledge(cur_game, i), threshold, rationality) for i in 1:num_players]
    play_game_with_logging(cur_agents, cur_game)
    push!(scores, current_score(cur_game.public))
end
println("Average score over 100 simulations: $(mean(scores))")
println("Standard deviation of scores over 100 simulations: $(std(scores))")