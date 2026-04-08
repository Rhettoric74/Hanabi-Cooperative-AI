using Plots, StatsBase, Random, Statistics
include("hanabi_game_state.jl")
include("agent.jl")
include("rsa/rsa_agent.jl")

# Helper function to create a single slot plot (the only plotting function needed)
function create_slot_plot(belief::CardBelief, slot_num::Int, phase::String, color::Symbol, show_xlabel::Bool=false, show_ylabel::Bool=false)
    # Get top 5 cards by probability (always take top 5)
    sorted_pairs = sort([(c, p) for (c, p) in belief.probs], by=x->x[2], rev=true)
    
    # Always take exactly 5 items (pad with empty strings and zero probabilities if needed)
    n_to_show = 5
    top5_cards = Card[]
    top5_probs = Float64[]
    
    for idx in 1:n_to_show
        if idx <= length(sorted_pairs)
            push!(top5_cards, sorted_pairs[idx][1])
            push!(top5_probs, sorted_pairs[idx][2])
        else
            # Pad with dummy card (won't be displayed as bar will have zero height)
            push!(top5_cards, Card(:none, 0))
            push!(top5_probs, 0.0)
        end
    end
    
    # Create abbreviated card labels (empty string for dummy cards)
    cards = String[]
    for card in top5_cards
        if card.color == :none && card.number == 0
            push!(cards, "")
        else
            push!(cards, string(first(string(card.color)), card.number))
        end
    end
    probs = top5_probs
    
    # Create title
    title_text = phase == "" ? "Slot $slot_num" : "Slot $slot_num ($phase)"
    
    # Create bar plot
    p = bar(1:n_to_show, probs,  # Use numeric positions instead of string labels for better control
        title=title_text,
        xlabel=show_xlabel ? "Card" : "",
        ylabel=show_ylabel ? "Probability" : "",
        legend=false,
        ylims=(0, 1),
        color=color,
        alpha=0.7,
        framestyle=:box,
        titlefontsize=10,
        tickfontsize=8,
        bar_width=0.7,
        xticks=(1:n_to_show, cards),  # Set custom x-tick labels
        xtickfontrotation=0,
        xlims=(0.5, n_to_show + 0.5)
    )
    
    # Add rounded probability labels centered on bars
    for i in 1:length(probs)
        if probs[i] > 0.01
            # Use the bar's center position (i) and place text centered
            annotate!(p, i, probs[i] + 0.03, 
                     Plots.text(round(probs[i], digits=2), 9, :black, :center, :center))
        end
    end
    """
    # Mark known information
    if belief.known isa Card && belief.known.color != :none
        known_str = string(first(string(belief.known.color)), belief.known.number)
        known_idx = findfirst(x -> x == known_str, cards)
        if known_idx !== nothing && known_idx <= length(probs) && probs[known_idx] > 0
            scatter!(p, [known_idx], [probs[known_idx] + 0.02], 
                markershape=:star, markersize=10, color=:red, label="", markeralpha=1.0)
        end
    elseif belief.known_color !== nothing && belief.known_color != :none
        # Highlight cards of known color
        color_abbrev = string(first(string(belief.known_color)))
        color_idxs = findall(x -> startswith(x, color_abbrev), cards)
        for idx in color_idxs
            if idx <= length(probs) && probs[idx] > 0
                scatter!(p, [idx], [probs[idx] + 0.02], 
                    markershape=:circle, markersize=4, color=:orange, alpha=0.8, label="")
            end
        end
    elseif belief.known_number !== nothing && belief.known_number > 0
        # Highlight cards of known number
        number_idxs = findall(x -> endswith(x, string(belief.known_number)), cards)
        for idx in number_idxs
            if idx <= length(probs) && probs[idx] > 0
                scatter!(p, [idx], [probs[idx] + 0.02], 
                    markershape=:square, markersize=4, color=:purple, alpha=0.8, label="")
            end
        end
    end
    """
    
    return p
end

# Plot entire hand
function plot_hand_beliefs(hand::Vector{CardBelief}, player_id::Int=1, title_prefix::String="")
    n_cards = length(hand)
    plots = []
    
    for i in 1:n_cards
        show_ylabel = (i == 1)
        if !isempty(title_prefix)
            p = create_slot_plot(hand[i], i, title_prefix, :lightgreen, false, show_ylabel)
        else
            p = create_slot_plot(hand[i], i, "", :lightgreen, false, show_ylabel)
        end
        push!(plots, p)
    end
    
    # Arrange plots
    if length(plots) > 5
        n_cols = ceil(Int, n_cards / 2)
        final_plot = plot(plots..., layout=(2, n_cols), size=(200 * n_cols, 500))
    else
        final_plot = plot(plots..., layout=(1, n_cards), size=(220 * n_cards, 450))
    end
    
    return final_plot
end

# Compare before and after beliefs
function compare_beliefs(before_hand::Vector{CardBelief}, after_hand::Vector{CardBelief}, 
                         player_id::Int=1, title::String="Belief Comparison")
    n_cards = length(before_hand)
    
    before_plots = []
    after_plots = []
    
    for i in 1:n_cards
        # Before plot (top row)
        show_ylabel_before = (i == 1)
        p_before = create_slot_plot(before_hand[i], i, "Before", :steelblue, false, show_ylabel_before)
        push!(before_plots, p_before)
        
        # After plot (bottom row)
        show_ylabel_after = (i == 1)
        show_xlabel = true  # Show xlabel only for bottom row
        p_after = create_slot_plot(after_hand[i], i, "After", :lightgreen, show_xlabel, show_ylabel_after)
        push!(after_plots, p_after)
    end
    
    # Combine: top row = before, bottom row = after
    all_plots = vcat(before_plots, after_plots)
    final_plot = plot(all_plots..., layout=(2, n_cards), size=(220 * n_cards, 700),
                     title=title, titlefontsize=12)
    
    return final_plot
end

# ============ MAIN SCRIPT ============
function run_hanabi_visualization()
    
    # Set fixed seed
    Random.seed!(31)
    threshold = 0.5
    rationality = 1.0  # RSA rationality parameter (alpha)
    use_softmax = true
    hint_cost_scaling = 1.0  # Cost sensitivity parameter for hint cost (beta)
    println("="^60)
    println("HANABI BELIEF VISUALIZATION")
    println("="^60)
    
    # Initialize game
    num_players = 4
    num_cards = 4
    initial_game = init_game(num_players, num_cards)
    
    # Initialize agents
    
    agents = [
        GreedyHanabiAgent(1, init_player_knowledge(initial_game, 1), threshold),
        GreedyHanabiAgent(2, init_player_knowledge(initial_game, 2), threshold),
        GreedyHanabiAgent(3, init_player_knowledge(initial_game, 3), threshold),
        GreedyHanabiAgent(4, init_player_knowledge(initial_game, 4), threshold)
    ]
    
    agents = [RSAHanabiAgent(i, init_player_knowledge(initial_game, i), threshold, rationality, use_softmax, hint_cost_scaling) for i in 1:num_players]
    
    for (i, agent) in enumerate(agents)
        agent.player_id = i
        agent.player_knowledge = init_player_knowledge(initial_game, i)
    end
    
    # Store beliefs before and after
    println("\n📊 Initial beliefs for Player 1:")
    before_beliefs = deepcopy(agents[1].player_knowledge.own_hand)
    display(plot_hand_beliefs(before_beliefs, 1, "INITIAL"))
    
    # Play a few turns to see belief updates
    println("\n🎮 Playing 3 turns...\n")
    # Use a mutable copy for the game state that we'll update
    current_game = initial_game

    # Display initial game state
    println("\n" * "="^60)
    println("INITIAL GAME STATE")
    println("="^60)
    println("\nOther players' hands (visible to you):")
    for p in 2:num_players
        println("  Player $p's hand:")
        for (i, card) in enumerate(current_game.player_hands[p])
            println("    Slot $i: $(card.color) $(card.number)")
        end
    end
    println("\nPlayed stacks: $(current_game.public.played_stacks)")
    println("Info tokens: $(current_game.public.info_tokens)")
    println("="^60)

    # ===== HARD-CODE YOUR ACTIONS HERE =====
    # Uncomment and modify these lines to create your scenario
    #
    action_sequence = [
          (1, GiveHint(1, 4, :rainbow))      # Player 1 hints red to Player 2
    #     (2, PlayAction(1)),                   # Player 2 plays slot 1
    #     (3, DiscardAction(2)),                # Player 3 discards slot 2
    #     (1, GiveHint(3, :number, 5)),         # Player 1 hints 5's to Player 3
    ]

    
    # Store beliefs before
    println("\n📊 Initial beliefs for Player 4:")
    before_beliefs = deepcopy(agents[4].player_knowledge.own_hand)
    display(plot_hand_beliefs(before_beliefs, 4, "INITIAL"))

    # Execute your action sequence
    println("\n🎮 Playing custom actions...\n")

    for (turn, (player_id, action)) in enumerate(action_sequence)
        println("Turn $turn - Player $player_id")
        println("  Action: $action")
        
        current_game = execute_action!(current_game, action)
        
        # Update beliefs
        for (i, agent) in enumerate(agents)
            update_beliefs_action!(agent, action, player_id, current_game)
        end
        
        if action isa GiveHint
            hint = last(current_game.public.hint_history)
            println("  Hint: $hint")
            for agent in agents
                update_beliefs_hint!(agent, hint, current_game)
            end
        end
        
        println("  Score: $(current_score(current_game.public))\n")
        
        over, reason = is_game_over(current_game.public)
        over && (println("Game over: $reason"); break)
    end

    # Show updated beliefs
    println("\n📊 Updated beliefs for Player 4:")
    after_beliefs = deepcopy(agents[4].player_knowledge.own_hand)
    display(compare_beliefs(before_beliefs, after_beliefs, 4, "P1 Beliefs"))

    println("\n🃏 Actual cards in Player 4's hand:")
    for (i, card) in enumerate(current_game.player_hands[4])
        println("  Slot $i: $(card.color) $(card.number)")
    end
    
    # Show updated beliefs
    println("\n📊 Updated beliefs for Player 4 after 1 turns:")
    after_beliefs = deepcopy(agents[4].player_knowledge.own_hand)
    
    # Compare before and after using the comparison function
    println("\n📈 Comparing beliefs (Top row = Before, Bottom row = After):")
    comparison_plot = compare_beliefs(before_beliefs, after_beliefs, 4, "P4 Beliefs")
    display(comparison_plot)
    
    # Also show actual cards for reference
    println("\n🃏 Actual cards in Player 4's hand:")
    for (i, card) in enumerate(current_game.player_hands[4])
        println("  Slot $i: $(card.color) $(card.number)")
    end
end

# Run the main function
run_hanabi_visualization()
