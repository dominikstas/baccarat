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

// Global variables
int coins = 1000;
Deck deck;

// Function prototypes
void start();
void game_loop();
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
void display_rules();
void display_stats();
int validate_input(int min, int max);

// Main function
int main() {
    srand(time(NULL));
    start();
    return 0;
}

// Initialize the game
void start() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║        BACCARAT CARD GAME            ║\n");
    printf("║     Complete Rules Implementation    ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    printf("Welcome! You start with %d coins.\n", coins);
    printf("Would you like to see the rules? (1=Yes, 0=No): ");
    
    int show_rules = validate_input(0, 1);
    if (show_rules) {
        display_rules();
    }
    
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
    
    // Final card display
    printf("🎯 Final Results:\n");
    display_cards(player_cards, player_card_count, "Player");
    printf("   Player final total: %d\n\n", player_total);
    
    display_cards(banker_cards, banker_card_count, "Banker");
    printf("   Banker final total: %d\n\n", banker_total);
    
    // Determine winner and calculate payout
    Winner result = evaluate_winner(player_total, banker_total);
    printf("🏆 Result: %s wins!\n", winner_str(result));
    
    // Calculate winnings
    int winnings = 0;
    if (result == bet_type) {
        if (result == TIE) {
            winnings = bet_amount * 8; // 8:1 payout for tie
        } else if (result == PLAYER) {
            winnings = bet_amount * 2; // 1:1 payout (double your bet)
        } else { // BANKER
            winnings = (bet_amount * 2) - (bet_amount / 20); // 1:1 minus 5% commission
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
    printf("0 - 🏦 Banker (1:1, 5%% commission)\n");
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

// Display game rules
void display_rules() {
    printf("\n╔══════════════════════════════════════╗\n");
    printf("║             BACCARAT RULES           ║\n");
    printf("╚══════════════════════════════════════╝\n");
    printf("🎯 OBJECTIVE: Bet on which hand will be closer to 9\n\n");
    printf("🃏 CARD VALUES:\n");
    printf("   • Ace = 1 point\n");
    printf("   • 2-9 = Face value\n");
    printf("   • 10, J, Q, K = 0 points\n\n");
    printf("📊 SCORING:\n");
    printf("   • Only rightmost digit counts (e.g., 15 = 5)\n");
    printf("   • Best possible hand = 9\n\n");
    printf("🎲 BETTING OPTIONS:\n");
    printf("   • Player: 1:1 payout\n");
    printf("   • Banker: 1:1 payout (5%% commission)\n");
    printf("   • Tie: 8:1 payout\n\n");
    printf("🎴 DRAWING RULES:\n");
    printf("   • Player draws 3rd card if total ≤ 5\n");
    printf("   • Banker follows complex rules based on player's 3rd card\n");
    printf("   • Natural 8 or 9 stops all drawing\n\n");
    printf("Press Enter to continue...");
    getchar();
    getchar();
}

// Display current game statistics
void display_stats() {
    int remaining_cards = 52 - deck.top;
    printf("Cards remaining in deck: %d\n", remaining_cards);
}