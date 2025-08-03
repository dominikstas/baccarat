#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

typedef enum { BANKER, PLAYER, TIE } Winner;
typedef enum { HEARTS, DIAMONDS, CLUBS, SPADES } Suit;

typedef struct {
    int rank;  // 1-13 (A, 2-10, J, Q, K)
    Suit suit;
} Card;

typedef struct {
    Card cards[52];
    int top;
} Deck;

typedef struct {
    Winner winner;
    int player_total;
    int banker_total;
} GameResult;

// Struktura do przechowywania danych symulacji
typedef struct {
    Card player_cards[3];
    int player_card_count;
    Card banker_cards[3];
    int banker_card_count;
    int player_final_total;
    int banker_final_total;
    Winner winner;
    int player_natural;  // 1 jeśli gracz miał natural (8 lub 9), 0 w przeciwnym razie
    int banker_natural;  // 1 jeśli bank miał natural (8 lub 9), 0 w przeciwnym razie
} SimulationData;

// Global variables
int coins = 1000;
Deck deck;
GameResult last_results[10];  // Historia ostatnich 10 wyników
int results_count = 0;        // Liczba zapisanych wyników

// Function prototypes
void start();
void game_loop();
void simulation_mode();
void run_simulation(int num_games, const char* filename);
void save_simulation_to_csv(SimulationData* data, int count, const char* filename);
SimulationData simulate_single_game();
void play_round(int bet_type, int bet_amount);
Winner evaluate_winner(int p_total, int b_total);
void init_deck();
void shuffle_deck();
Card draw_card();
int card_value(Card card);
const char* winner_str(Winner w);
const char* card_name(Card card);
const char* suit_name(Suit suit);
int get_bet_amount();
int get_bet_type();
void display_cards(Card* cards, int count, const char* player_name);
int calculate_total(Card* cards, int count);
int should_player_draw_third(int player_total);
int should_banker_draw_third(int banker_total, int player_third_card_value);
void display_stats();
int validate_input(int min, int max);
void add_result(Winner winner, int player_total, int banker_total);
void display_last_results();
const char* winner_symbol(Winner w);
void card_to_string(Card card, char* buffer);

// Main function
int main() {
    srand(time(NULL));
    
    printf("╔══════════════════════════════════════╗\n");
    printf("║               BACCARAT               ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    printf("Choose mode:\n");
    printf("1 - 🎮 Play Game\n");
    printf("2 - 🤖 AI Simulation Mode\n");
    printf("Your choice (1-2): ");
    
    int mode = validate_input(1, 2);
    
    if (mode == 1) {
        start();
    } else {
        simulation_mode();
    }
    
    return 0;
}

// Simulation mode menu
void simulation_mode() {
    printf("\nSIMULATION MODE\n");
    printf("═══════════════════════\n");
    printf("Choose simulation size:\n");
    printf("1 - 1,000 games\n");
    printf("2 - 10,000 games\n");
    printf("3 - 100,000 games\n");
    printf("4 - Custom amount\n");
    printf("Your choice (1-4): ");
    
    int choice = validate_input(1, 4);
    int num_games;
    
    switch (choice) {
        case 1: num_games = 1000; break;
        case 2: num_games = 10000; break;
        case 3: num_games = 100000; break;
        case 4:
            printf("Enter number of games (1-1000000): ");
            num_games = validate_input(1, 1000000);
            break;
        default: num_games = 100000;
    }
    
    char filename[100];
    printf("Enter filename for CSV output (without .csv extension): ");
    scanf("%99s", filename);
    strcat(filename, ".csv");
    
    printf("\n🚀 Starting simulation of %d games...\n", num_games);
    printf("📁 Results will be saved to: %s\n\n", filename);
    
    run_simulation(num_games, filename);
}

// Run the simulation
void run_simulation(int num_games, const char* filename) {
    SimulationData* simulation_data = malloc(num_games * sizeof(SimulationData));
    if (!simulation_data) {
        printf("❌ Memory allocation failed!\n");
        return;
    }
    
    init_deck();
    shuffle_deck();
    
    clock_t start_time = clock();
    
    for (int i = 0; i < num_games; i++) {
        // Reshuffle if deck is low
        if (deck.top > 42) {  // Keep more cards for safety
            init_deck();
            shuffle_deck();
        }
        
        simulation_data[i] = simulate_single_game();
        
        // Progress indicator
        if ((i + 1) % 10000 == 0 || i == num_games - 1) {
            printf("Progress: %d/%d games (%.1f%%)\n", 
                   i + 1, num_games, ((float)(i + 1) / num_games) * 100.0);
        }
    }
    
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n✅ Simulation completed!\n");
    printf("⏱️  Time taken: %.2f seconds\n", time_taken);
    printf("🎯 Games per second: %.0f\n", num_games / time_taken);
    
    // Save results to CSV
    save_simulation_to_csv(simulation_data, num_games, filename);
    
    // Display summary statistics
    int banker_wins = 0, player_wins = 0, ties = 0;
    int naturals = 0;
    
    for (int i = 0; i < num_games; i++) {
        switch (simulation_data[i].winner) {
            case BANKER: banker_wins++; break;
            case PLAYER: player_wins++; break;
            case TIE: ties++; break;
        }
        if (simulation_data[i].player_natural || simulation_data[i].banker_natural) {
            naturals++;
        }
    }
    
    printf("\nSIMULATION STATISTICS:\n");
    printf("════════════════════════\n");
    printf("Banker wins: %d (%.2f%%)\n", banker_wins, (float)banker_wins / num_games * 100);
    printf("Player wins: %d (%.2f%%)\n", player_wins, (float)player_wins / num_games * 100);
    printf("Ties: %d (%.2f%%)\n", ties, (float)ties / num_games * 100);
    printf("Games with naturals: %d (%.2f%%)\n", naturals, (float)naturals / num_games * 100);
    printf("Data saved to: %s\n", filename);
    
    free(simulation_data);
    printf("\n Simulation complete\n");
}

// Simulate a single game and return data
SimulationData simulate_single_game() {
    SimulationData data = {0};
    
    // Deal initial cards
    data.player_cards[0] = draw_card();
    data.banker_cards[0] = draw_card();
    data.player_cards[1] = draw_card();
    data.banker_cards[1] = draw_card();
    data.player_card_count = 2;
    data.banker_card_count = 2;
    
    // Calculate initial totals
    int player_total = calculate_total(data.player_cards, 2);
    int banker_total = calculate_total(data.banker_cards, 2);
    
    // Check for naturals
    data.player_natural = (player_total >= 8) ? 1 : 0;
    data.banker_natural = (banker_total >= 8) ? 1 : 0;
    
    // Apply third card rules if no natural
    if (player_total < 8 && banker_total < 8) {
        int player_third_value = -1;
        
        // Player's third card
        if (should_player_draw_third(player_total)) {
            data.player_cards[2] = draw_card();
            data.player_card_count = 3;
            player_third_value = card_value(data.player_cards[2]);
            player_total = calculate_total(data.player_cards, 3);
        }
        
        // Banker's third card
        if (should_banker_draw_third(banker_total, player_third_value)) {
            data.banker_cards[2] = draw_card();
            data.banker_card_count = 3;
            banker_total = calculate_total(data.banker_cards, 3);
        }
    }
    
    data.player_final_total = player_total;
    data.banker_final_total = banker_total;
    data.winner = evaluate_winner(player_total, banker_total);
    
    return data;
}

// Save simulation data to CSV file
void save_simulation_to_csv(SimulationData* data, int count, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("❌ Error: Could not create file %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "game_id,");
    fprintf(file, "player_card1_rank,player_card1_suit,");
    fprintf(file, "player_card2_rank,player_card2_suit,");
    fprintf(file, "player_card3_rank,player_card3_suit,");
    fprintf(file, "player_card_count,");
    fprintf(file, "banker_card1_rank,banker_card1_suit,");
    fprintf(file, "banker_card2_rank,banker_card2_suit,");
    fprintf(file, "banker_card3_rank,banker_card3_suit,");
    fprintf(file, "banker_card_count,");
    fprintf(file, "player_final_total,banker_final_total,");
    fprintf(file, "player_natural,banker_natural,");
    fprintf(file, "winner\n");
    
    // Write data
    for (int i = 0; i < count; i++) {
        SimulationData* game = &data[i];
        
        fprintf(file, "%d,", i + 1);
        
        // Player cards
        fprintf(file, "%d,%d,", game->player_cards[0].rank, game->player_cards[0].suit);
        fprintf(file, "%d,%d,", game->player_cards[1].rank, game->player_cards[1].suit);
        if (game->player_card_count == 3) {
            fprintf(file, "%d,%d,", game->player_cards[2].rank, game->player_cards[2].suit);
        } else {
            fprintf(file, "0,0,");  // No third card
        }
        fprintf(file, "%d,", game->player_card_count);
        
        // Banker cards
        fprintf(file, "%d,%d,", game->banker_cards[0].rank, game->banker_cards[0].suit);
        fprintf(file, "%d,%d,", game->banker_cards[1].rank, game->banker_cards[1].suit);
        if (game->banker_card_count == 3) {
            fprintf(file, "%d,%d,", game->banker_cards[2].rank, game->banker_cards[2].suit);
        } else {
            fprintf(file, "0,0,");  // No third card
        }
        fprintf(file, "%d,", game->banker_card_count);
        
        // Totals and results
        fprintf(file, "%d,%d,", game->player_final_total, game->banker_final_total);
        fprintf(file, "%d,%d,", game->player_natural, game->banker_natural);
        fprintf(file, "%d\n", game->winner);  // 0=BANKER, 1=PLAYER, 2=TIE
    }
    
    fclose(file);
    printf("💾 Successfully saved %d games to %s\n", count, filename);
}

// Initialize the game
void start() {
    printf("Welcome! You start with %d coins.\n", coins);
    init_deck();
    shuffle_deck();
    game_loop();
}

// Main game loop
void game_loop() {
    int rounds_played = 0;
    
    while (coins > 0) {
        // Reshuffle deck if running low on cards
        if (deck.top < 10) {
            printf("\n🔄 Reshuffling deck...\n");
            init_deck();
            shuffle_deck();
        }
        
        rounds_played++;
        printf("\n" "═══════════════════════════════════════\n");
        printf("                ROUND %d\n", rounds_played);
        printf("═══════════════════════════════════════\n");
        printf("💰 Current coins: %d\n\n", coins);
        
        // Pokaż historię wyników jeśli są jakieś
        if (results_count > 0) {
            display_last_results();
            printf("\n");
        }
        
        // Get bet type and amount
        int bet_type = get_bet_type();
        int bet_amount = get_bet_amount();
        
        if (bet_amount > coins) {
            printf("❌ Insufficient coins! You only have %d coins.\n", coins);
            continue;
        }
        
        printf("\n🎲 Dealing cards...\n");
        play_round(bet_type, bet_amount);
        
        printf("\n📊 ");
        display_stats();
        
        if (coins > 0) {
            printf("Continue playing? (1=Yes, 0=No): ");
            int continue_game = validate_input(0, 1);
            if (!continue_game) break;
        }
    }
    
    printf("\n🎮 Game Over!\n");
    printf("📈 Rounds played: %d\n", rounds_played);
    printf("💰 Final coins: %d\n", coins);
    
    if (coins == 0) {
        printf("💸 Better luck next time!\n");
    } else {
        printf("🎉 Thanks for playing!\n");
    }
}

// Play a single round
void play_round(int bet_type, int bet_amount) {
    Card player_cards[3] = {0};
    Card banker_cards[3] = {0};
    int player_card_count = 2;
    int banker_card_count = 2;
    
    // Deal initial two cards to each
    player_cards[0] = draw_card();
    banker_cards[0] = draw_card();
    player_cards[1] = draw_card();
    banker_cards[1] = draw_card();
    
    // Calculate initial totals
    int player_total = calculate_total(player_cards, 2);
    int banker_total = calculate_total(banker_cards, 2);
    
    // Display initial cards
    printf("\n🎴 Initial Cards:\n");
    display_cards(player_cards, 2, "Player");
    printf("   Player total: %d\n\n", player_total);
    
    display_cards(banker_cards, 2, "Banker");
    printf("   Banker total: %d\n\n", banker_total);
    
    // Check for natural win (8 or 9)
    if (player_total >= 8 || banker_total >= 8) {
        printf("🌟 Natural! No more cards drawn.\n");
    } else {
        // Player's third card logic
        int player_third_value = -1;
        if (should_player_draw_third(player_total)) {
            player_cards[2] = draw_card();
            player_card_count = 3;
            player_third_value = card_value(player_cards[2]);
            player_total = calculate_total(player_cards, 3);
            
            printf("🃏 Player draws third card:\n");
            printf("   %s\n", card_name(player_cards[2]));
            printf("   Player new total: %d\n\n", player_total);
        }
        
        // Banker's third card logic
        if (should_banker_draw_third(banker_total, player_third_value)) {
            banker_cards[2] = draw_card();
            banker_card_count = 3;
            banker_total = calculate_total(banker_cards, 3);
            
            printf("🃏 Banker draws third card:\n");
            printf("   %s\n", card_name(banker_cards[2]));
            printf("   Banker new total: %d\n\n", banker_total);
        }
    }
    
    
    // Determine winner and calculate payout
    Winner result = evaluate_winner(player_total, banker_total);
    printf("🏆 Result: %s wins!\n", winner_str(result));
    
    // Dodaj wynik do historii
    add_result(result, player_total, banker_total);
    
    // Calculate winnings
    int winnings = 0;
    if (result == bet_type) {
        if (result == TIE) {
            winnings = bet_amount * 8; // 8:1 payout for tie
        } else if (result == PLAYER) {
            winnings = bet_amount * 2; // 1:1 payout (double your bet)
        } else { // BANKER
            winnings = (bet_amount * 2); // 1:1 
        }
        
        coins = coins - bet_amount + winnings;
        printf("🎉 You win! Winnings: %d coins\n", winnings - bet_amount);
    } else {
        coins -= bet_amount;
        printf("💔 You lose %d coins.\n", bet_amount);
    }
    
    printf("💰 Current coins: %d\n", coins);
}

// Initialize deck with all 52 cards
void init_deck() {
    int index = 0;
    for (int suit = 0; suit < 4; suit++) {
        for (int rank = 1; rank <= 13; rank++) {
            deck.cards[index].rank = rank;
            deck.cards[index].suit = suit;
            index++;
        }
    }
    deck.top = 0;
}

// Shuffle the deck using Fisher-Yates algorithm
void shuffle_deck() {
    for (int i = 51; i > 0; i--) {
        int j = rand() % (i + 1);
        Card temp = deck.cards[i];
        deck.cards[i] = deck.cards[j];
        deck.cards[j] = temp;
    }
    deck.top = 0;
}

// Draw a card from the deck
Card draw_card() {
    if (deck.top >= 52) {
        printf("⚠️  Deck empty! Reshuffling...\n");
        init_deck();
        shuffle_deck();
    }
    return deck.cards[deck.top++];
}

// Get the Baccarat value of a card
int card_value(Card card) {
    if (card.rank >= 10) return 0; // 10, J, Q, K are worth 0
    return card.rank; // A=1, 2-9 face value
}

// Calculate total value for a hand
int calculate_total(Card* cards, int count) {
    int total = 0;
    for (int i = 0; i < count; i++) {
        total += card_value(cards[i]);
    }
    return total % 10;
}

// Display cards in a formatted way
void display_cards(Card* cards, int count, const char* player_name) {
    printf("   %s: ", player_name);
    for (int i = 0; i < count; i++) {
        printf("%s", card_name(cards[i]));
        if (i < count - 1) printf(", ");
    }
    printf("\n");
}

// Player third card drawing rules
int should_player_draw_third(int player_total) {
    return player_total <= 5;
}

// Banker third card drawing rules (official Baccarat rules)
int should_banker_draw_third(int banker_total, int player_third_card_value) {
    if (player_third_card_value == -1) { // Player didn't draw
        return banker_total <= 5;
    }
    
    switch (banker_total) {
        case 0:
        case 1:
        case 2:
            return 1; // Always draw
        case 3:
            return player_third_card_value != 8;
        case 4:
            return player_third_card_value >= 2 && player_third_card_value <= 7;
        case 5:
            return player_third_card_value >= 4 && player_third_card_value <= 7;
        case 6:
            return player_third_card_value == 6 || player_third_card_value == 7;
        default:
            return 0; // Never draw on 7, 8, 9
    }
}

// Determine the winner
Winner evaluate_winner(int p_total, int b_total) {
    if (p_total > b_total) return PLAYER;
    if (b_total > p_total) return BANKER;
    return TIE;
}

// Get bet type from user
int get_bet_type() {
    printf("Choose your bet:\n");
    printf("0 - 🏦 Banker (1:1)\n");
    printf("1 - 👤 Player (1:1)\n");
    printf("2 - 🤝 Tie (8:1)\n");
    printf("Your choice (0-2): ");
    
    return validate_input(0, 2);
}

// Get bet amount from user
int get_bet_amount() {
    printf("Enter bet amount (1-%d): ", coins);
    return validate_input(1, coins);
}

// Validate user input within range
int validate_input(int min, int max) {
    int input;
    while (1) {
        if (scanf("%d", &input) == 1 && input >= min && input <= max) {
            return input;
        } else {
            printf("❌ Invalid input. Please enter a number between %d and %d: ", min, max);
            while (getchar() != '\n'); // Clear input buffer
        }
    }
}

// Dodaj wynik do historii
void add_result(Winner winner, int player_total, int banker_total) {
    // Przesuń wszystkie wyniki o jedną pozycję w lewo jeśli tablica jest pełna
    if (results_count >= 10) {
        for (int i = 0; i < 9; i++) {
            last_results[i] = last_results[i + 1];
        }
        results_count = 9;
    }
    
    // Dodaj nowy wynik na końcu
    last_results[results_count].winner = winner;
    last_results[results_count].player_total = player_total;
    last_results[results_count].banker_total = banker_total;
    results_count++;
}

// Wyświetl ostatnie wyniki
void display_last_results() {
    printf("📊 Last %d results: ", results_count);
    for (int i = 0; i < results_count; i++) {
        printf("%s", winner_symbol(last_results[i].winner));
        if (i < results_count - 1) printf(" ");
    }
    printf("\n");
    
    // Szczegółowa historia (opcjonalnie)
    printf("   Detailed: ");
    for (int i = 0; i < results_count; i++) {
        printf("%s(%d-%d)", 
               winner_symbol(last_results[i].winner),
               last_results[i].player_total,
               last_results[i].banker_total);
        if (i < results_count - 1) printf(" ");
    }
}

// Symbol zwycięzcy dla kompaktowego wyświetlania
const char* winner_symbol(Winner w) {
    switch (w) {
        case BANKER: return "🏦";
        case PLAYER: return "👤";
        case TIE: return "🤝";
        default: return "❓";
    }
}

// Helper functions for display
const char* winner_str(Winner w) {
    switch (w) {
        case BANKER: return "Banker";
        case PLAYER: return "Player";
        case TIE: return "Tie";
        default: return "Unknown";
    }
}

const char* card_name(Card card) {
    static char card_str[20];
    const char* rank_names[] = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};
    const char* suit_symbols[] = {"♥", "♦", "♣", "♠"};
    
    snprintf(card_str, sizeof(card_str), "%s%s", rank_names[card.rank], suit_symbols[card.suit]);
    return card_str;
}

const char* suit_name(Suit suit) {
    switch (suit) {
        case HEARTS: return "Hearts";
        case DIAMONDS: return "Diamonds";
        case CLUBS: return "Clubs";
        case SPADES: return "Spades";
        default: return "Unknown";
    }
}

// Convert card to string for CSV output
void card_to_string(Card card, char* buffer) {
    sprintf(buffer, "%d-%d", card.rank, card.suit);
}

// Display current game statistics
void display_stats() {
    int remaining_cards = 52 - deck.top;
    printf("Cards remaining in deck: %d\n", remaining_cards);
}