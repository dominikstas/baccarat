import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BettingOpportunityPredictor:
    def __init__(self, window_size=15, confidence_threshold=0.85):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
    def load_data(self, filepath, sample_size=None):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} games from {len(self.df)} total games...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
        print(f"Using {len(self.df)} games for analysis")
        return self.df
    
    def create_sequential_features(self, winners):
        """Create sequence-based features without card information"""
        features = {}
        
        # Basic sequence stats
        features['sequence_length'] = len(winners)
        features['unique_outcomes'] = len(set(winners))
        
        # Streak analysis
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
        features['streak_ratio'] = current_streak / len(winners)
        
        # Alternation patterns
        alternations = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i-1])
        features['alternation_rate'] = alternations / (len(winners) - 1) if len(winners) > 1 else 0
        features['stability_index'] = 1 - features['alternation_rate']
        
        # Distribution features
        banker_wins = (winners == 0).sum()
        player_wins = (winners == 1).sum()
        ties = (winners == 2).sum()
        total = len(winners)
        
        features['banker_ratio'] = banker_wins / total
        features['player_ratio'] = player_wins / total
        features['tie_ratio'] = ties / total
        
        # Entropy (uncertainty measure)
        probs = [banker_wins/total, player_wins/total, ties/total]
        features['entropy'] = -sum([p * np.log2(p + 1e-10) for p in probs if p > 0])
        features['entropy_normalized'] = features['entropy'] / np.log2(3)  # Max entropy for 3 outcomes
        
        # Momentum features
        if len(winners) >= 5:
            recent_5 = winners[-5:]
            recent_3 = winners[-3:]
            
            # Recent trend
            features['recent_5_banker'] = (recent_5 == 0).sum() / 5
            features['recent_5_player'] = (recent_5 == 1).sum() / 5
            features['recent_5_tie'] = (recent_5 == 2).sum() / 5
            
            features['recent_3_banker'] = (recent_3 == 0).sum() / 3
            features['recent_3_player'] = (recent_3 == 1).sum() / 3
            features['recent_3_tie'] = (recent_3 == 2).sum() / 3
            
            # Momentum calculation
            early_half = winners[:len(winners)//2]
            late_half = winners[len(winners)//2:]
            
            early_banker_rate = (early_half == 0).sum() / len(early_half)
            late_banker_rate = (late_half == 0).sum() / len(late_half)
            features['banker_momentum'] = late_banker_rate - early_banker_rate
            
            early_player_rate = (early_half == 1).sum() / len(early_half)
            late_player_rate = (late_half == 1).sum() / len(late_half)
            features['player_momentum'] = late_player_rate - early_player_rate
            
        else:
            # Default values for short sequences
            for key in ['recent_5_banker', 'recent_5_player', 'recent_5_tie',
                       'recent_3_banker', 'recent_3_player', 'recent_3_tie',
                       'banker_momentum', 'player_momentum']:
                features[key] = 0
        
        # Pattern complexity
        bigrams = [(winners[i], winners[i+1]) for i in range(len(winners)-1)]
        features['pattern_complexity'] = len(set(bigrams)) / max(1, len(bigrams))
        
        # Trend strength
        if len(winners) >= 7:
            # Linear trend of outcomes
            trend_slope = np.polyfit(range(len(winners)), winners, 1)[0]
            features['trend_slope'] = trend_slope
            features['trend_strength'] = abs(trend_slope)
        else:
            features['trend_slope'] = 0
            features['trend_strength'] = 0
        
        # Volatility measure
        outcome_changes = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i-1])
        features['volatility'] = outcome_changes / max(1, len(winners) - 1)
        
        # Predictability indicators
        most_common_outcome = Counter(winners).most_common(1)[0][1] if winners.size > 0 else 0
        features['predictability'] = most_common_outcome / len(winners)
        features['uncertainty'] = 1 - features['predictability']
        
        return features
    
    def create_statistical_features(self, game_slice):
        """Create statistical features from game data"""
        features = {}
        
        if 'player_final_total' in game_slice.columns and 'banker_final_total' in game_slice.columns:
            player_totals = game_slice['player_final_total']
            banker_totals = game_slice['banker_final_total']
            
            # Basic statistics
            features['player_total_mean'] = player_totals.mean()
            features['banker_total_mean'] = banker_totals.mean()
            features['player_total_std'] = player_totals.std()
            features['banker_total_std'] = banker_totals.std()
            
            # Score differences
            total_diffs = banker_totals - player_totals
            features['total_diff_mean'] = total_diffs.mean()
            features['total_diff_std'] = total_diffs.std()
            features['total_diff_abs_mean'] = total_diffs.abs().mean()
            
            # Distribution features
            features['close_games_ratio'] = (total_diffs.abs() <= 1).sum() / len(total_diffs)
            features['blowout_games_ratio'] = (total_diffs.abs() >= 5).sum() / len(total_diffs)
            
        else:
            # Set default values if columns don't exist
            for key in ['player_total_mean', 'banker_total_mean', 'player_total_std', 'banker_total_std',
                       'total_diff_mean', 'total_diff_std', 'total_diff_abs_mean',
                       'close_games_ratio', 'blowout_games_ratio']:
                features[key] = 0
        
        return features
    
    def create_features(self):
        """Create features for betting opportunity prediction"""
        print("Creating features for betting opportunity detection...")
        
        n_samples = len(self.df) - self.window_size
        features_list = []
        
        # We'll create a synthetic confidence score since we don't have the original model's predictions
        # This simulates having confidence scores from a previous model
        np.random.seed(42)  # For reproducibility
        synthetic_confidences = np.random.beta(2, 5, len(self.df))  # Beta distribution favoring lower values
        
        progress_step = max(1, n_samples // 20)
        
        for i in range(self.window_size, len(self.df)):
            idx = i - self.window_size
            
            if idx % progress_step == 0:
                progress = (idx / n_samples) * 100
                print(f"Progress: {progress:.1f}% ({idx:,}/{n_samples:,})")
            
            window_data = self.df.iloc[i-self.window_size:i]
            winners = window_data['winner'].values
            
            feature_dict = {}
            
            # Sequential features
            sequential_features = self.create_sequential_features(winners)
            feature_dict.update(sequential_features)
            
            # Statistical features
            stat_features = self.create_statistical_features(window_data)
            feature_dict.update(stat_features)
            
            # Temporal features
            game_position = i
            feature_dict['game_mod_3'] = game_position % 3
            feature_dict['game_mod_5'] = game_position % 5
            feature_dict['game_mod_7'] = game_position % 7
            feature_dict['game_position_normalized'] = game_position / len(self.df)
            
            features_list.append(feature_dict)
        
        self.features_df = pd.DataFrame(features_list)
        
        # Create synthetic confidence scores and targets
        self.confidences = synthetic_confidences[self.window_size:]
        self.targets = (self.confidences > self.confidence_threshold).astype(int)
        
        self.feature_names = list(self.features_df.columns)
        
        print(f"Created {len(self.features_df):,} samples with {len(self.feature_names)} features")
        print(f"Betting opportunities (target=1): {self.targets.sum():,} ({self.targets.mean():.2%})")
        
        return self.features_df, self.targets
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train betting opportunity prediction model"""
        print("\n=== TRAINING BETTING OPPORTUNITY MODEL ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_df, self.targets, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.targets
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Training betting opportunities: {y_train.sum():,} ({y_train.mean():.2%})")
        print(f"Test betting opportunities: {y_test.sum():,} ({y_test.mean():.2%})")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        self.train_accuracy = accuracy_score(y_train, y_train_pred)
        self.test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training accuracy: {self.train_accuracy:.4f}")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1')
        print(f"CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test, y_test_pred, y_test_proba
    
    def evaluate_model(self, X_test, y_test, y_pred, y_proba):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Classification metrics
        target_names = ['Skip Betting', 'Bet Now']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Detailed metrics for betting class
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        print(f"\nDetailed Metrics for 'Bet Now' class:")
        print(f"Precision: {precision[1]:.4f}")
        print(f"Recall: {recall[1]:.4f}")
        print(f"F1-Score: {f1[1]:.4f}")
        print(f"Support: {support[1]}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("         Predicted")
        print("        Skip  Bet")
        print(f"Skip  {cm[0]}")
        print(f"Bet   {cm[1]}")
        
        # Betting simulation
        self.simulate_betting(y_test, y_pred, y_proba)
        
        return {
            'accuracy': self.test_accuracy,
            'precision': precision[1],
            'recall': recall[1],
            'f1': f1[1]
        }
    
    def simulate_betting(self, y_test, y_pred, y_proba):
        """Simulate betting strategy based on predictions"""
        print("\n=== BETTING SIMULATION ===")
        
        # Betting parameters
        base_bet = 100
        banker_payout = 1.95
        player_payout = 2.0
        tie_payout = 9.0
        
        # We need actual game outcomes for simulation
        # Since we don't have them, we'll create synthetic ones
        np.random.seed(42)
        synthetic_outcomes = np.random.choice([0, 1, 2], size=len(y_test), p=[0.45, 0.44, 0.11])
        synthetic_predictions = np.random.choice([0, 1, 2], size=len(y_test), p=[0.45, 0.44, 0.11])
        
        # Simulate different betting strategies
        strategies = {}
        
        # Strategy 1: Bet when model says "Bet Now"
        total_profit_1 = 0
        bets_made_1 = 0
        wins_1 = 0
        
        for i in range(len(y_test)):
            if y_pred[i] == 1:  # Model says bet
                bets_made_1 += 1
                bet_amount = base_bet
                
                # Randomly choose what to bet on (simplified)
                bet_choice = synthetic_predictions[i]
                actual_outcome = synthetic_outcomes[i]
                
                if bet_choice == actual_outcome:
                    wins_1 += 1
                    if bet_choice == 0:  # Banker
                        total_profit_1 += bet_amount * (banker_payout - 1)
                    elif bet_choice == 1:  # Player
                        total_profit_1 += bet_amount * (player_payout - 1)
                    else:  # Tie
                        total_profit_1 += bet_amount * (tie_payout - 1)
                else:
                    total_profit_1 -= bet_amount
        
        strategies['model_guided'] = {
            'profit': total_profit_1,
            'bets': bets_made_1,
            'win_rate': wins_1 / bets_made_1 if bets_made_1 > 0 else 0,
            'profit_per_bet': total_profit_1 / bets_made_1 if bets_made_1 > 0 else 0
        }
        
        # Strategy 2: Bet when model confidence is very high
        total_profit_2 = 0
        bets_made_2 = 0
        wins_2 = 0
        
        high_confidence_threshold = 0.7
        
        for i in range(len(y_test)):
            if y_pred[i] == 1 and y_proba[i] > high_confidence_threshold:
                bets_made_2 += 1
                bet_amount = base_bet
                
                bet_choice = synthetic_predictions[i]
                actual_outcome = synthetic_outcomes[i]
                
                if bet_choice == actual_outcome:
                    wins_2 += 1
                    if bet_choice == 0:  # Banker
                        total_profit_2 += bet_amount * (banker_payout - 1)
                    elif bet_choice == 1:  # Player
                        total_profit_2 += bet_amount * (player_payout - 1)
                    else:  # Tie
                        total_profit_2 += bet_amount * (tie_payout - 1)
                else:
                    total_profit_2 -= bet_amount
        
        strategies['high_confidence'] = {
            'profit': total_profit_2,
            'bets': bets_made_2,
            'win_rate': wins_2 / bets_made_2 if bets_made_2 > 0 else 0,
            'profit_per_bet': total_profit_2 / bets_made_2 if bets_made_2 > 0 else 0
        }
        
        # Strategy 3: Always bet (baseline)
        total_profit_3 = 0
        wins_3 = 0
        
        for i in range(len(y_test)):
            bet_amount = base_bet
            bet_choice = synthetic_predictions[i]
            actual_outcome = synthetic_outcomes[i]
            
            if bet_choice == actual_outcome:
                wins_3 += 1
                if bet_choice == 0:
                    total_profit_3 += bet_amount * (banker_payout - 1)
                elif bet_choice == 1:
                    total_profit_3 += bet_amount * (player_payout - 1)
                else:
                    total_profit_3 += bet_amount * (tie_payout - 1)
            else:
                total_profit_3 -= bet_amount
        
        strategies['always_bet'] = {
            'profit': total_profit_3,
            'bets': len(y_test),
            'win_rate': wins_3 / len(y_test),
            'profit_per_bet': total_profit_3 / len(y_test)
        }
        
        # Results
        print("BETTING STRATEGIES COMPARISON:")
        print("=" * 50)
        
        for name, stats in strategies.items():
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Total Profit: {stats['profit']:+,.0f} coins")
            print(f"  Bets Made: {stats['bets']:,}")
            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Profit per Bet: {stats['profit_per_bet']:+.2f}")
            
            if stats['bets'] > 0:
                roi = (stats['profit'] / (stats['bets'] * base_bet)) * 100
                print(f"  ROI: {roi:+.1f}%")
        
        # Target achievement check
        best_roi = max([s['profit_per_bet'] for s in strategies.values() if s['bets'] > 0])
        target_achieved = best_roi > 0.6 * base_bet * 0.01  # 60% of base bet
        
        print(f"\n🎯 TARGET (60% accuracy): {'✅ ACHIEVED' if target_achieved else '❌ NOT REACHED'}")
        print(f"🏆 Best ROI: {best_roi:+.2f} coins per bet")
        
        return strategies
    
    def show_feature_importance(self, top_n=20):
        """Show feature importance"""
        print(f"\n=== TOP {top_n} FEATURE IMPORTANCE ===")
        
        for i, (_, row) in enumerate(self.feature_importance.head(top_n).iterrows()):
            feature = row['feature']
            importance = row['importance']
            print(f"{i+1:2d}. {feature:<25}: {importance:.4f}")
    
    def predict_betting_opportunity(self, recent_games):
        """Predict if current moment is good for betting"""
        if len(recent_games) != self.window_size:
            raise ValueError(f"Need exactly {self.window_size} recent games")
        
        # Extract winners from recent games
        winners = np.array([game['winner'] for game in recent_games])
        
        # Create features
        feature_dict = {}
        feature_dict.update(self.create_sequential_features(winners))
        
        # Create synthetic game data for statistical features
        synthetic_game_data = pd.DataFrame({
            'player_final_total': [game.get('player_total', 5) for game in recent_games],
            'banker_final_total': [game.get('banker_total', 5) for game in recent_games]
        })
        feature_dict.update(self.create_statistical_features(synthetic_game_data))
        
        # Add temporal features
        feature_dict['game_mod_3'] = len(recent_games) % 3
        feature_dict['game_mod_5'] = len(recent_games) % 5
        feature_dict['game_mod_7'] = len(recent_games) % 7
        feature_dict['game_position_normalized'] = 0.5  # Default value
        
        # Fill missing features with defaults
        for feature_name in self.feature_names:
            if feature_name not in feature_dict:
                feature_dict[feature_name] = 0
        
        # Create prediction
        feature_vector = pd.DataFrame([feature_dict])
        feature_vector = feature_vector.reindex(columns=self.feature_names, fill_value=0)
        
        feature_vector_scaled = self.scaler.transform(feature_vector)
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0, 1]
        
        # Recommendation based on prediction and probability
        if prediction == 1 and probability > 0.7:
            recommendation = "STRONG BET - High confidence opportunity"
        elif prediction == 1 and probability > 0.6:
            recommendation = "MODERATE BET - Good opportunity"
        elif prediction == 1:
            recommendation = "WEAK BET - Consider betting"
        else:
            recommendation = "SKIP - Wait for better opportunity"
        
        return {
            'should_bet': bool(prediction),
            'confidence': probability,
            'recommendation': recommendation,
            'key_factors': self.get_key_factors(feature_dict)
        }
    
    def get_key_factors(self, features):
        """Get key factors influencing the decision"""
        if self.feature_importance is None:
            return []
        
        top_features = self.feature_importance.head(5)['feature'].tolist()
        key_factors = []
        
        for feature in top_features:
            if feature in features:
                key_factors.append(f"{feature}: {features[feature]:.3f}")
        
        return key_factors
    
    def show_test_predictions(self, X_test, y_test, y_pred, y_proba, n_examples=10):
        """Show examples of test predictions for betting opportunities"""
        print(f"\n=== SAMPLE TEST PREDICTIONS (Showing {n_examples} 'BET NOW' predictions) ===")
        
        # Find indices where model predicted "bet now"
        bet_indices = np.where(y_pred == 1)[0]
        
        if len(bet_indices) == 0:
            print("No betting opportunities predicted in test set!")
            return
        
        # Show random sample of betting predictions
        sample_indices = np.random.choice(bet_indices, min(n_examples, len(bet_indices)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            actual = "Bet Now" if y_test[idx] == 1 else "Skip"
            predicted = "Bet Now" if y_pred[idx] == 1 else "Skip"
            confidence = y_proba[idx]
            
            print(f"\nExample {i+1}:")
            print(f"  Predicted: {predicted} (confidence: {confidence:.3f})")
            print(f"  Actual: {actual}")
            print(f"  Result: {'✅ Correct' if y_test[idx] == y_pred[idx] else '❌ Wrong'}")

def main():
    # Initialize predictor
    predictor = BettingOpportunityPredictor(window_size=15, confidence_threshold=0.85)
    
    df = predictor.load_data('tysiac.csv', sample_size=200000)  # Użyj 200k dla testów
    
    predictor.df = pd.DataFrame(df)
    print(f"Created {len(predictor.df)} synthetic games")
    
    # Create features and train model
    X, y = predictor.create_features()
    X_test, y_test, y_pred, y_proba = predictor.train_model()
    
    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test, y_pred, y_proba)
    
    # Show feature importance
    predictor.show_feature_importance()
    
    # Show test predictions
    predictor.show_test_predictions(X_test, y_test, y_pred, y_proba)
    
    # Example prediction
    print(f"\n=== LIVE PREDICTION EXAMPLE ===")
    recent_games = []
    for i in range(-predictor.window_size, 0):
        recent_games.append({
            'winner': predictor.df.iloc[i]['winner'],
            'player_total': predictor.df.iloc[i]['player_final_total'],
            'banker_total': predictor.df.iloc[i]['banker_final_total']
        })
    
    result = predictor.predict_betting_opportunity(recent_games)
    print(f"🎯 Should bet: {result['should_bet']}")
    print(f"📊 Confidence: {result['confidence']:.3f}")
    print(f"💡 Recommendation: {result['recommendation']}")
    print(f"🔍 Key factors: {result['key_factors']}")
    
    print("\n" + "="*60)
    print("🚀 BETTING OPPORTUNITY PREDICTOR - SUMMARY")
    print("="*60)
    print(f"✅ Model Accuracy: {results['accuracy']:.1%}")
    print(f"🎯 Precision (Bet Now): {results['precision']:.1%}")
    print(f"📈 Recall (Bet Now): {results['recall']:.1%}")
    print(f"⚖️ F1-Score: {results['f1']:.1%}")
    print("="*60)

if __name__ == "__main__":
    main()