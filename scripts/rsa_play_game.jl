include("agent.jl")
include("rsa_agent.jl")
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
num_players = 2
num_cards = 5

# Configuration: Choose agent type
# Options: "greedy", "rsa", "mixed"
agent_type = "rsa"
threshold = 0.75
rsa_alpha = 3.0

println("Running simulations with agent_type = $agent_type")
println("=" ^ 50)

for i in 1:num_simulations
    cur_game = init_game(num_players, num_cards)
    
    if agent_type == "greedy"
        cur_agents = [GreedyHanabiAgent(j, init_player_knowledge(cur_game, j), threshold) 
                      for j in 1:num_players]
    elseif agent_type == "rsa"
        cur_agents = [RSAHanabiAgent(j, init_player_knowledge(cur_game, j), threshold, rsa_alpha) 
                      for j in 1:num_players]
    elseif agent_type == "mixed"
        # Mix of greedy and RSA agents
        cur_agents = AbstractHanabiAgent[]
        for j in 1:num_players
            if j == 1
                push!(cur_agents, RSAHanabiAgent(j, init_player_knowledge(cur_game, j), threshold, rsa_alpha))
            else
                push!(cur_agents, GreedyHanabiAgent(j, init_player_knowledge(cur_game, j), threshold))
            end
        end
    else
        error("Unknown agent_type: $agent_type")
    end
    
    play_game_with_logging(cur_agents, cur_game)
    push!(scores, current_score(cur_game.public))
end

println("\n" * "=" ^ 50)
println("Results for $agent_type agents:")
println("Average score over $num_simulations simulations: $(mean(scores))")
println("Standard deviation: $(std(scores))")
println("Min score: $(minimum(scores))")
println("Max score: $(maximum(scores))")