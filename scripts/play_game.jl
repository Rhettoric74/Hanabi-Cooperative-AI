include("agent.jl")
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

game = init_game(4, 4)
agents = [GreedyHanabiAgent(i, init_player_knowledge(game, i), 0.99) for i in 1:4]
play_game_with_logging(agents, game)