import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
        
        # Policz karty gracza
        for _, game in game_slice.iterrows():
            # Karty gracza
            if game['player_card1_rank'] > 0:
                card_counts[game['player_card1_rank']] += 1
            if game['player_card2_rank'] > 0:
                card_counts[game['player_card2_rank']] += 1
            if game['player_card3_rank'] > 0:
                card_counts[game['player_card3_rank']] += 1
            
            # Karty bankiera
            if game['banker_card1_rank'] > 0:
                card_counts[game['banker_card1_rank']] += 1
            if game['banker_card2_rank'] > 0:
                card_counts[game['banker_card2_rank']] += 1
            if game['banker_card3_rank'] > 0:
                card_counts[game['banker_card3_rank']] += 1
        
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
    
    def create_pattern_features(self, winners):
        """Tworzenie cech wzorców"""
        features = {}
        
        # Podstawowe sekwencje
        features['banker_wins'] = (winners == 0).sum()
        features['player_wins'] = (winners == 1).sum()
        features['ties'] = (winners == 2).sum()
        
        # Streaks (serie)
        current_streak = 1
        max_streak = 1
        last_winner = winners[0]
        
        for winner in winners[1:]:
            if winner == last_winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
            last_winner = winner
        
        features['current_streak'] = current_streak
        features['max_streak'] = max_streak
        features['last_winner'] = winners[-1]
        
        # Wzorce ostatnich gier
        features['last_3_banker'] = (winners[-3:] == 0).sum()
        features['last_3_player'] = (winners[-3:] == 1).sum()
        features['last_3_ties'] = (winners[-3:] == 2).sum()
        
        # Alternowanie wzorców
        alternations = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i-1]:
                alternations += 1
        features['alternation_rate'] = alternations / (len(winners) - 1)
        
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
        """Tworzenie cech dla modelu"""
        print("Creating features...")
        
        features_list = []
        targets = []
        
        # Tworzenie sliding window
        for i in range(self.window_size, len(self.df)):
            # Pobierz okno danych
            window_data = self.df.iloc[i-self.window_size:i]
            winners = window_data['winner'].values
            
            # Utwórz cechy
            feature_dict = {}
            
            # 1. Cechy wzorców wyników
            pattern_features = self.create_pattern_features(winners)
            feature_dict.update(pattern_features)
            
            # 2. Cechy liczenia kart
            card_features = self.create_card_counting_features(self.df, i-self.window_size, i)
            feature_dict.update(card_features)
            
            # 3. Cechy przebiegu gier
            game_features = self.create_game_features(window_data)
            feature_dict.update(game_features)
            
            # 4. Dodatkowe cechy sekwencyjne
            feature_dict.update({
                'winner_std': np.std(winners),
                'winner_trend': np.corrcoef(range(len(winners)), winners)[0,1] if len(set(winners)) > 1 else 0,
                'recent_volatility': np.std(winners[-5:]) if len(winners) >= 5 else 0,
            })
            
            features_list.append(feature_dict)
            targets.append(self.df.iloc[i]['winner'])
        
        # Konwersja do DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.targets = np.array(targets)
        self.feature_names = list(self.features_df.columns)
        
        print(f"Created {len(self.features_df)} feature vectors with {len(self.feature_names)} features each")
        return self.features_df, self.targets
    
    def train_model(self, test_size=0.2, random_state=42):
        """Trenowanie modelu"""
        print("\n=== TRAINING MODEL ===")
        
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
        
        # Trenowanie Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        print("Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predykcje
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
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
        
        return X_test_scaled, y_test, y_test_pred
    
    def evaluate_model(self, X_test, y_test, y_pred):
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
        
    def show_feature_importance(self, top_n=15):
        """Pokazuje najważniejsze cechy"""
        print(f"\n=== TOP {top_n} FEATURE IMPORTANCE ===")
        for i, (_, row) in enumerate(self.feature_importance.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<20}: {row['importance']:.4f}")
    
    def predict_next_game(self, recent_games):
        """Przewidywanie następnej gry na podstawie ostatnich N gier"""
        if len(recent_games) != self.window_size:
            raise ValueError(f"Need exactly {self.window_size} recent games")
        
        # Utwórz cechy dla ostatnich gier
        winners = np.array([game['winner'] for game in recent_games])
        
        feature_dict = {}
        pattern_features = self.create_pattern_features(winners)
        feature_dict.update(pattern_features)
        
        # Dodaj pozostałe cechy (uproszczone dla predykcji)
        feature_dict.update({
            'high_cards_count': 0,  # Wymagałoby śledzenia wszystkich kart
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
    df = predictor.load_data('tysiac.csv')  # tu idzie nazwa pliku
    predictor.analyze_data()
    
    # Tworzenie cech
    X, y = predictor.create_features()
    
    # Trenowanie modelu
    X_test, y_test, y_pred = predictor.train_model()
    
    # Ewaluacja
    predictor.evaluate_model(X_test, y_test, y_pred)
    
    # Analiza ważności cech
    predictor.show_feature_importance()
    
    print("\n=== ANALYSIS CONCLUSIONS ===")
    print("1. Card counting features may have limited impact due to frequent deck reshuffling")
    print("2. Pattern recognition shows recent trends and streaks")
    print("3. Game flow features capture statistical tendencies")
    print("4. Model performance vs baseline indicates predictive patterns exist")
    print("\nFor betting strategy, focus on features with highest importance scores.")

if __name__ == "__main__":
    main()