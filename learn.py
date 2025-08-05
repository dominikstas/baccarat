import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class BaccaratPredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Wczytaj dane z pliku CSV"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} games")
        return self.df
    
    def analyze_data(self):
        """Podstawowa analiza danych"""
        print("\n=== DATA ANALYSIS ===")
        print(f"Total games: {len(self.df)}")
        
        # Rozkład wyników
        winner_counts = self.df['winner'].value_counts()
        print(f"\nWinner distribution:")
        print(f"Banker (0): {winner_counts.get(0, 0)} ({winner_counts.get(0, 0)/len(self.df)*100:.2f}%)")
        print(f"Player (1): {winner_counts.get(1, 0)} ({winner_counts.get(1, 0)/len(self.df)*100:.2f}%)")
        print(f"Tie (2): {winner_counts.get(2, 0)} ({winner_counts.get(2, 0)/len(self.df)*100:.2f}%)")
        
        # Analiza naturalnych
        naturals = (self.df['player_natural'] == 1) | (self.df['banker_natural'] == 1)
        print(f"\nGames with naturals: {naturals.sum()} ({naturals.sum()/len(self.df)*100:.2f}%)")
        
        # Średnie sumy punktów
        print(f"\nAverage totals:")
        print(f"Player: {self.df['player_final_total'].mean():.2f}")
        print(f"Banker: {self.df['banker_final_total'].mean():.2f}")
    
    def create_card_counting_features(self, data, start_idx, end_idx):
        """Tworzenie cech związanych z liczeniem kart"""
        game_slice = data.iloc[start_idx:end_idx]
        
        # Liczenie kart według rang (1-13)
        card_counts = np.zeros(14)  # index 0 nie używany, 1-13 dla rang
        
        # Policz karty gracza i bankiera
        for _, game in game_slice.iterrows():
            cards = [
                game['player_card1_rank'], game['player_card2_rank'], game['player_card3_rank'],
                game['banker_card1_rank'], game['banker_card2_rank'], game['banker_card3_rank']
            ]
            for card in cards:
                if card > 0:
                    card_counts[card] += 1
        
        # Przygotuj cechy liczenia kart
        high_cards = card_counts[10:14].sum()  # 10, J, Q, K (wartość 0 w baccarat)
        low_cards = card_counts[1:5].sum()     # A, 2, 3, 4
        mid_cards = card_counts[5:10].sum()    # 5, 6, 7, 8, 9
        
        return {
            'high_cards_count': high_cards,
            'low_cards_count': low_cards,
            'mid_cards_count': mid_cards,
            'cards_used': card_counts[1:].sum(),
            'high_low_ratio': high_cards / (low_cards + 1),  # +1 żeby uniknąć dzielenia przez 0
        }
    
    def create_advanced_pattern_features(self, winners):
        """Tworzenie zaawansowanych cech wzorców"""
        features = {}
        
        # 1. STREAK_LENGTH - obecna seria zwycięstw
        current_winner = winners[-1]
        streak_length = 1
        for i in range(len(winners) - 2, -1, -1):
            if winners[i] == current_winner:
                streak_length += 1
            else:
                break
        features['streak_length'] = streak_length
        
        # 2. LAST_SWITCH - ile gier temu zmienił się zwycięzca
        last_switch = 0
        for i in range(len(winners) - 2, -1, -1):
            if winners[i] != current_winner:
                break
            last_switch += 1
        features['last_switch'] = last_switch
        
        # 3. PLAYER_RUN & BANKER_RUN - serie dla każdej strony
        player_run = 0
        banker_run = 0
        for i in range(len(winners) - 1, -1, -1):
            if winners[i] == 1:  # Player
                player_run += 1
            elif winners[i] == 0:  # Banker
                banker_run += 1
            else:
                break
        features['player_run'] = player_run if current_winner == 1 else 0
        features['banker_run'] = banker_run if current_winner == 0 else 0
        
        # 4. MOMENTUM - różnica między ostatnimi 3 wynikami
        if len(winners) >= 3:
            last_3 = winners[-3:]
            banker_momentum = (last_3 == 0).sum()
            player_momentum = (last_3 == 1).sum()
            features['momentum'] = banker_momentum - player_momentum
        else:
            features['momentum'] = 0
            
        # 5. OSCILLATION - zmienność (czy wynik się zmienia często?)
        changes = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i-1]:
                changes += 1
        features['oscillation'] = changes / (len(winners) - 1) if len(winners) > 1 else 0
        
        return features
    
    def create_pattern_features(self, winners):
        """Tworzenie podstawowych cech wzorców"""
        features = {}
        
        # Podstawowe sekwencje
        features['banker_wins'] = (winners == 0).sum()
        features['player_wins'] = (winners == 1).sum()
        features['ties'] = (winners == 2).sum()
        
        # Streaks (serie)
        current_streak = 1
        max_streak = 1
        last_winner = winners[0] if len(winners) > 0 else 0
        
        for winner in winners[1:]:
            if winner == last_winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
            last_winner = winner
        
        features['current_streak'] = current_streak
        features['max_streak'] = max_streak
        features['last_winner'] = winners[-1] if len(winners) > 0 else 0
        
        # Wzorce ostatnich gier
        if len(winners) >= 3:
            features['last_3_banker'] = (winners[-3:] == 0).sum()
            features['last_3_player'] = (winners[-3:] == 1).sum()
            features['last_3_ties'] = (winners[-3:] == 2).sum()
        else:
            features['last_3_banker'] = 0
            features['last_3_player'] = 0
            features['last_3_ties'] = 0
        
        # Alternowanie wzorców
        alternations = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i-1]:
                alternations += 1
        features['alternation_rate'] = alternations / (len(winners) - 1) if len(winners) > 1 else 0
        
        return features
    
    def create_game_features(self, game_slice):
        """Tworzenie cech z przebiegu gier"""
        features = {}
        
        # Średnie sumy punktów
        features['avg_player_total'] = game_slice['player_final_total'].mean()
        features['avg_banker_total'] = game_slice['banker_final_total'].mean()
        features['avg_total_diff'] = (game_slice['banker_final_total'] - game_slice['player_final_total']).mean()
        
        # Częstość naturalnych
        features['natural_rate'] = ((game_slice['player_natural'] == 1) | (game_slice['banker_natural'] == 1)).mean()
        features['player_natural_rate'] = (game_slice['player_natural'] == 1).mean()
        features['banker_natural_rate'] = (game_slice['banker_natural'] == 1).mean()
        
        # Średnia liczba kart
        features['avg_player_cards'] = game_slice['player_card_count'].mean()
        features['avg_banker_cards'] = game_slice['banker_card_count'].mean()
        
        # Częstość trzecich kart
        features['player_third_card_rate'] = (game_slice['player_card_count'] == 3).mean()
        features['banker_third_card_rate'] = (game_slice['banker_card_count'] == 3).mean()
        
        return features
    
    def create_features(self):
        """Tworzenie cech dla modelu z optymalizacją"""
        print("Creating features with sliding window...")
        
        # Pre-allocate arrays for better performance
        n_samples = len(self.df) - self.window_size
        features_list = []
        targets = np.zeros(n_samples, dtype=int)
        
        # Progress tracking
        progress_step = max(1, n_samples // 20)  # 20 updates
        
        # Tworzenie sliding window
        for i in range(self.window_size, len(self.df)):
            idx = i - self.window_size
            
            # Progress indicator
            if idx % progress_step == 0:
                progress = (idx / n_samples) * 100
                print(f"Progress: {progress:.1f}% ({idx}/{n_samples})")
            
            # Pobierz okno danych
            window_data = self.df.iloc[i-self.window_size:i]
            winners = window_data['winner'].values
            
            # Utwórz cechy
            feature_dict = {}
            
            # 1. Podstawowe cechy wzorców
            pattern_features = self.create_pattern_features(winners)
            feature_dict.update(pattern_features)
            
            # 2. Zaawansowane cechy wzorców (NOWE!)
            advanced_features = self.create_advanced_pattern_features(winners)
            feature_dict.update(advanced_features)
            
            # 3. Cechy liczenia kart
            card_features = self.create_card_counting_features(self.df, i-self.window_size, i)
            feature_dict.update(card_features)
            
            # 4. Cechy przebiegu gier
            game_features = self.create_game_features(window_data)
            feature_dict.update(game_features)
            
            # 5. Dodatkowe cechy sekwencyjne
            feature_dict.update({
                'winner_std': np.std(winners),
                'winner_trend': np.corrcoef(range(len(winners)), winners)[0,1] if len(set(winners)) > 1 else 0,
                'recent_volatility': np.std(winners[-5:]) if len(winners) >= 5 else 0,
            })
            
            features_list.append(feature_dict)
            targets[idx] = self.df.iloc[i]['winner']
        
        # Konwersja do DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.targets = targets
        self.feature_names = list(self.features_df.columns)
        
        print(f"Created {len(self.features_df)} feature vectors with {len(self.feature_names)} features each")
        return self.features_df, self.targets
    
    def train_model(self, test_size=0.2, random_state=42):
        """Trenowanie modelu XGBoost"""
        print("\n=== TRAINING XGBOOST MODEL ===")
        
        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_df, self.targets, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.targets
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Skalowanie cech
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Trenowanie XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        
        print("Training XGBoost...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predykcje
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Prawdopodobieństwa dla strategii zakładów
        y_test_proba = self.model.predict_proba(X_test_scaled)
        
        # Zapisz wyniki
        self.train_accuracy = accuracy_score(y_train, y_train_pred)
        self.test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training accuracy: {self.train_accuracy:.4f}")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test, y_test_pred, y_test_proba
    
    def evaluate_betting_strategy(self, y_test, y_pred, y_proba, confidence_threshold=0.6):
        """Ewaluacja strategii zakładów z progiem pewności"""
        print(f"\n=== BETTING STRATEGY EVALUATION (Confidence > {confidence_threshold:.0%}) ===")
        
        # Payouts w Baccarat
        banker_payout = 1.95  # 1:1 minus 5% prowizji
        player_payout = 2.0   # 1:1
        tie_payout = 9.0      # 8:1
        
        payouts = [banker_payout, player_payout, tie_payout]
        
        # Strategia 1: Zawsze stawiaj (baseline)
        total_profit_always = 0
        bets_always = 0
        
        for true_winner, predicted in zip(y_test, y_pred):
            bet_amount = 100  # Stała stawka
            if true_winner == predicted:
                total_profit_always += bet_amount * (payouts[predicted] - 1)
            else:
                total_profit_always -= bet_amount
            bets_always += 1
        
        # Strategia 2: Stawiaj tylko przy wysokiej pewności
        total_profit_confident = 0
        bets_confident = 0
        wins_confident = 0
        
        for i, (true_winner, predicted) in enumerate(zip(y_test, y_pred)):
            max_prob = np.max(y_proba[i])
            
            if max_prob >= confidence_threshold:
                bet_amount = 100
                bets_confident += 1
                
                if true_winner == predicted:
                    total_profit_confident += bet_amount * (payouts[predicted] - 1)
                    wins_confident += 1
                else:
                    total_profit_confident -= bet_amount
        
        # Strategia 3: Progressive betting (wyższa stawka przy większej pewności)
        total_profit_progressive = 0
        bets_progressive = 0
        
        for i, (true_winner, predicted) in enumerate(zip(y_test, y_pred)):
            max_prob = np.max(y_proba[i])
            
            if max_prob >= 0.4:  # Niższy próg
                # Stawka proporcjonalna do pewności
                bet_amount = int(50 + (max_prob - 0.4) * 250)  # 50-200 w zależności od pewności
                bets_progressive += 1
                
                if true_winner == predicted:
                    total_profit_progressive += bet_amount * (payouts[predicted] - 1)
                else:
                    total_profit_progressive -= bet_amount
        
        # Wyniki
        print("\nSTRATEGY COMPARISON:")
        print("═" * 50)
        
        print(f"1. ALWAYS BET:")
        print(f"   Total bets: {bets_always}")
        print(f"   Total profit: {total_profit_always:+.0f} coins")
        print(f"   Profit per bet: {total_profit_always/bets_always:+.2f} coins")
        print(f"   Win rate: {accuracy_score(y_test, y_pred):.1%}")
        
        if bets_confident > 0:
            print(f"\n2. HIGH CONFIDENCE ({confidence_threshold:.0%}+):")
            print(f"   Total bets: {bets_confident} ({bets_confident/len(y_test):.1%} of games)")
            print(f"   Total profit: {total_profit_confident:+.0f} coins")
            print(f"   Profit per bet: {total_profit_confident/bets_confident:+.2f} coins")
            print(f"   Win rate: {wins_confident/bets_confident:.1%}")
        else:
            print(f"\n2. HIGH CONFIDENCE ({confidence_threshold:.0%}+):")
            print("   No bets met confidence threshold!")
        
        if bets_progressive > 0:
            print(f"\n3. PROGRESSIVE BETTING (40%+):")
            print(f"   Total bets: {bets_progressive} ({bets_progressive/len(y_test):.1%} of games)")
            print(f"   Total profit: {total_profit_progressive:+.0f} coins")
            print(f"   Profit per bet: {total_profit_progressive/bets_progressive:+.2f} coins")
        
        # Analiza rozkładu pewności
        print(f"\nCONFIDENCE DISTRIBUTION:")
        confidence_levels = np.max(y_proba, axis=1)
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = (confidence_levels >= threshold).sum()
            if count > 0:
                subset_accuracy = accuracy_score(
                    y_test[confidence_levels >= threshold], 
                    y_pred[confidence_levels >= threshold]
                )
                print(f"   {threshold:.0%}+ confidence: {count} games ({count/len(y_test):.1%}) - Accuracy: {subset_accuracy:.1%}")
        
        return {
            'always_bet_profit': total_profit_always,
            'confident_bet_profit': total_profit_confident,
            'progressive_bet_profit': total_profit_progressive,
            'confident_bets_made': bets_confident,
            'confident_win_rate': wins_confident/bets_confident if bets_confident > 0 else 0
        }
    
    def evaluate_model(self, X_test, y_test, y_pred, y_proba):
        """Ewaluacja modelu"""
        print("\n=== MODEL EVALUATION ===")
        
        # Classification report
        target_names = ['Banker', 'Player', 'Tie']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("          Predicted")
        print("         Ban  Pla  Tie")
        for i, actual in enumerate(['Banker', 'Player', 'Tie  ']):
            print(f"Actual {actual}: {cm[i]}")
        
        # Baseline accuracy (przewidywanie najczęstszej klasy)
        baseline_accuracy = max(pd.Series(y_test).value_counts()) / len(y_test)
        print(f"\nBaseline accuracy (most frequent class): {baseline_accuracy:.4f}")
        print(f"Model improvement: {self.test_accuracy - baseline_accuracy:.4f}")
        
        # Analiza betting strategy
        betting_results = self.evaluate_betting_strategy(y_test, y_pred, y_proba)
        
        return betting_results
    
    def show_feature_importance(self, top_n=20):
        """Pokazuje najważniejsze cechy"""
        print(f"\n=== TOP {top_n} FEATURE IMPORTANCE ===")
        for i, (_, row) in enumerate(self.feature_importance.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    def predict_next_game(self, recent_games):
        """Przewidywanie następnej gry na podstawie ostatnich N gier"""
        if len(recent_games) != self.window_size:
            raise ValueError(f"Need exactly {self.window_size} recent games")
        
        # Utwórz cechy dla ostatnich gier
        winners = np.array([game['winner'] for game in recent_games])
        
        feature_dict = {}
        
        # Podstawowe wzorce
        pattern_features = self.create_pattern_features(winners)
        feature_dict.update(pattern_features)
        
        # Zaawansowane wzorce
        advanced_features = self.create_advanced_pattern_features(winners)
        feature_dict.update(advanced_features)
        
        # Dodaj pozostałe cechy (uproszczone dla predykcji)
        feature_dict.update({
            'high_cards_count': 0,
            'low_cards_count': 0,
            'mid_cards_count': 0,
            'cards_used': 0,
            'high_low_ratio': 1,
            'avg_player_total': np.mean([g.get('player_total', 4.5) for g in recent_games]),
            'avg_banker_total': np.mean([g.get('banker_total', 4.5) for g in recent_games]),
            'avg_total_diff': 0,
            'natural_rate': 0.2,
            'player_natural_rate': 0.1,
            'banker_natural_rate': 0.1,
            'avg_player_cards': 2.5,
            'avg_banker_cards': 2.5,
            'player_third_card_rate': 0.5,
            'banker_third_card_rate': 0.5,
            'winner_std': np.std(winners),
            'winner_trend': 0,
            'recent_volatility': np.std(winners[-5:]) if len(winners) >= 5 else 0,
        })
        
        # Utwórz DataFrame z cechami
        feature_vector = pd.DataFrame([feature_dict])
        feature_vector = feature_vector.reindex(columns=self.feature_names, fill_value=0)
        
        # Skalowanie i predykcja
        feature_vector_scaled = self.scaler.transform(feature_vector)
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        return {
            'prediction': prediction,
            'confidence': np.max(probabilities),
            'probabilities': {
                'banker': probabilities[0],
                'player': probabilities[1], 
                'tie': probabilities[2]
            }
        }

def main():
    # Inicjalizacja modelu
    predictor = BaccaratPredictor(window_size=10)
    
    # Wczytanie i analiza danych
    df = predictor.load_data('piecdziesiona.csv')
    predictor.analyze_data()
    
    # Tworzenie cech
    X, y = predictor.create_features()
    
    # Trenowanie modelu
    X_test, y_test, y_pred, y_proba = predictor.train_model()
    
    # Ewaluacja
    betting_results = predictor.evaluate_model(X_test, y_test, y_pred, y_proba)
    
    # Analiza ważności cech
    predictor.show_feature_importance()
    
    print("\n=== CONCLUSIONS FOR BETTING STRATEGY ===")
    print("1. XGBoost model with advanced pattern features")
    print("2. New features: streak_length, momentum, oscillation show pattern importance")
    print("3. High-confidence betting (60%+) reduces risk but limits opportunities")
    print("4. Progressive betting balances risk and reward")
    print("5. Focus on top feature importance for manual analysis")
    
    # Przykład predykcji
    if len(df) >= 10:
        print(f"\n=== EXAMPLE PREDICTION ===")
        recent_games = []
        for i in range(-10, 0):
            recent_games.append({
                'winner': df.iloc[i]['winner'],
                'player_total': df.iloc[i]['player_final_total'],
                'banker_total': df.iloc[i]['banker_final_total']
            })
        
        result = predictor.predict_next_game(recent_games)
        print(f"Prediction: {['Banker', 'Player', 'Tie'][result['prediction']]}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Probabilities: Banker {result['probabilities']['banker']:.1%}, "
              f"Player {result['probabilities']['player']:.1%}, "
              f"Tie {result['probabilities']['tie']:.1%}")
        
        if result['confidence'] >= 0.6:
            print("🎯 HIGH CONFIDENCE - Recommended bet!")
        else:
            print("⚠️  Low confidence - Consider skipping this bet.")

if __name__ == "__main__":
    main()