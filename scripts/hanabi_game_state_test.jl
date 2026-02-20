function verify_game_logic()
    println("="^70)
    println("VERIFYING HANABI GAME LOGIC")
    println("="^70)
    
    # Test 1: Deck composition
    println("\n1. Testing deck composition...")
    full_deck = create_full_deck()
    println("   Full deck size: $(length(full_deck)) (should be 60)")
    
    # Count cards in full deck
    counts = Dict{Card, Int}()
    for card in full_deck
        counts[card] = get(counts, card, 0) + 1
    end
    
    all_correct = true
    for (card, expected) in DECK_COMPOSITION
        if counts[card] != expected
            println("   ❌ $card: expected $expected, got $(counts[card])")
            all_correct = false
        end
    end
    
    if all_correct
        println("   ✅ Deck composition correct")
    end
    
    # Test 2: Initialize a game
    println("\n2. Initializing a 3-player game...")
    game = init_game(4, 4)
    println("   Players: $(length(game.player_hands))")
    println("   Cards per player: $(length(game.player_hands[1]))")
    println("   Deck size: $(length(game.deck))")
    println("   Total cards: $(sum(length, game.player_hands) + length(game.deck)) (should be 60)")
    
    # Test 3: Player knowledge for player 1
    println("\n3. Initializing player 1's knowledge...")
    p1_knowledge = init_player_knowledge(game, 1)
    
    println("   Player 1's actual hand: $(game.player_hands[1])")
    println("   Player 1 sees player 2's hand: $(p1_knowledge.other_hands[2])")
    println("   Player 1 sees player 3's hand: $(p1_knowledge.other_hands[3])")
    println("   Discard pile: $(p1_knowledge.discard_pile)")
    
    # Show belief probabilities for some interesting cards
    println("\n   BELIEF PROBABILITIES FOR PLAYER 1'S HAND:")
    
    # Pick a few cards to examine
    test_cards = [
        Card(:red, 1),    # Common card (3 copies)
        Card(:red, 5),    # Rare card (1 copy)
        Card(:rainbow, 3) # Rainbow card
    ]
    
    for (slot, belief) in enumerate(p1_knowledge.own_hand)
        println("\n   Slot $slot beliefs:")
        
        # Get probabilities for test cards
        for test_card in test_cards
            prob = get(belief.probs, test_card, 0.0)
            visible_count = count(==(test_card), vcat(values(p1_knowledge.other_hands)...))
            visible_count += count(==(test_card), p1_knowledge.discard_pile)
            played_count = 0
            for (color, level) in p1_knowledge.played_stacks
                for num in 1:level
                    if Card(color, num) == test_card
                        played_count += 1
                    end
                end
            end
            
            total_visible = visible_count + played_count
            max_copies = DECK_COMPOSITION[test_card]
            remaining = max_copies - total_visible
            
            println("      $test_card: $(round(prob*100, digits=1))% (visible: $total_visible/$max_copies, remaining in deck: $remaining)")
        end
        
        # Show top 3 most probable cards
        println("      Top 3 most likely:")
        sorted = sort(collect(belief.probs), by=x->x[2], rev=true)
        for (card, prob) in sorted[1:min(3, length(sorted))]
            println("        $card: $(round(prob*100, digits=1))%")
        end
    end
    
    # Test 4: Play a card
    println("\n4. Testing card play...")
    println("   Initial played stacks: $(game.public.played_stacks)")
    
    # Find a playable card in player 1's hand
    playable_idx = nothing
    for (i, card) in enumerate(game.player_hands[1])
        if can_play_card(game.public, card)
            playable_idx = i
            break
        end
    end
    
    if playable_idx !== nothing
        card = game.player_hands[1][playable_idx]
        println("   Playing $card from player 1's hand")
        game = execute_action!(game, PlayCard(1, playable_idx))
        println("   New played stacks: $(game.public.played_stacks)")
        println("   Current player: $(game.current_player)")
        println("   Last action: $(game.history[end])")
    else
        println("   No playable card found (this can happen randomly)")
    end
    
    # Test 5: Discard a card
    println("\n5. Testing discard...")
    if game.public.info_tokens < 8 && !isempty(game.player_hands[game.current_player])
        card = game.player_hands[game.current_player][1]
        println("   Discarding $card from player $(game.current_player)'s hand")
        game = execute_action!(game, DiscardCard(game.current_player, 1))
        println("   Info tokens: $(game.public.info_tokens)")
        println("   Discard pile size: $(length(game.public.discard_pile))")
    end
    
    # Test 6: Check game over conditions
    println("\n6. Game status:")
    over, reason = is_game_over(game.public)
    println("   Game over: $over")
    if !over
        println("   Reason: $reason")
    end
    println("   Current score: $(current_score(game.public))/$MAX_SCORE")
    println("\n7. Test Give Hint")
    println("$(game.current_player)")
    execute_action!(game, GiveHint(game.current_player, mod1(game.current_player + 1, length(game.player_hands)), game.player_hands[1][1].color))
    
    return game
end
verify_game_logic()