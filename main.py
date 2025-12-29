import sys
import os
import traceback

# Add src to path
sys.path.append('src')

try:
    from data_preprocessing import DataPreprocessor
    from model_architecture import DigitRecognizer
    from train_model import ModelTrainer
    from evaluate_model import ModelEvaluator
    from predict import DigitPredictor
    from utils import *
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📁 Checking directory structure...")
    
    # Create src directory if it doesn't exist
    if not os.path.exists('src'):
        print("Creating src directory...")
        os.makedirs('src')
    
    # Check if files exist
    required_files = [
        'data_preprocessing.py',
        'model_architecture.py', 
        'train_model.py',
        'evaluate_model.py',
        'predict.py',
        'utils.py'
    ]
    
    for file in required_files:
        if not os.path.exists(f'src/{file}'):
            print(f"⚠️  Missing: src/{file}")
    
    print("\n📋 Please make sure all source files are in the src/ directory.")
    sys.exit(1)

def main():
    # Configuration
    TRAIN_PATH = 'data/train.csv'
    TEST_PATH = 'data/test_fixed.csv'  # Use the fixed file
    MODEL_PATH = 'models/digit_recognizer.h5'
    
    print("=" * 50)
    print("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("=" * 50)
    
    try:
        # Step 1: Load and preprocess data
        print("\n1. 📊 Loading and preprocessing data...")
        
        # Check if data files exist
        if not os.path.exists(TRAIN_PATH):
            print(f"❌ Training data not found at: {TRAIN_PATH}")
            print("📥 Please download MNIST data from Kaggle:")
            print("   https://www.kaggle.com/competitions/digit-recognizer/data")
            print("   Place train.csv and test.csv in the data/ folder")
            return
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Initialize preprocessor
        # Try both with and without labels in test data
        try:
            preprocessor = DataPreprocessor(TRAIN_PATH, TEST_PATH, has_labels_in_test=False)
            X_train, X_val, y_train, y_val, X_test = preprocessor.get_preprocessed_data()
        except ValueError as e:
            print(f"⚠️  First attempt failed: {e}")
            print("🔄 Trying with has_labels_in_test=True...")
            preprocessor = DataPreprocessor(TRAIN_PATH, TEST_PATH, has_labels_in_test=True)
            X_train, X_val, y_train, y_val, X_test = preprocessor.get_preprocessed_data()
        
        # Visualize some samples
        try:
            visualize_dataset(X_train[:10], y_train[:10])
        except:
            print("⚠️  Could not visualize dataset, continuing...")
        
        # Check data distribution
        try:
            check_data_distribution(np.argmax(y_train, axis=1))
        except:
            print("⚠️  Could not check data distribution, continuing...")
        
        # Step 2: Build model
        print("\n2. 🏗️  Building model...")
        recognizer = DigitRecognizer()
        
        # Choose model type
        print("Select model type:")
        print("1. Simple model (faster training)")
        print("2. Improved model (better accuracy)")
        
        try:
            choice = input("Enter choice (1 or 2, default 1): ").strip()
            if choice == '2':
                model = recognizer.build_improved_model()
                print("✓ Building improved model...")
            else:
                model = recognizer.build_simple_model()
                print("✓ Building simple model...")
        except:
            model = recognizer.build_simple_model()
            print("✓ Building simple model (default)...")
        
        model = recognizer.compile_model(model)
        recognizer.summary()
        
        # Step 3: Train model
        print("\n3. 🏋️  Training model...")
        trainer = ModelTrainer(model, MODEL_PATH)
        
        # Get training parameters
        try:
            epochs_input = input("Enter number of epochs (default 10): ").strip()
            epochs = int(epochs_input) if epochs_input else 10
        except:
            epochs = 10
        
        try:
            batch_size_input = input("Enter batch size (default 32): ").strip()
            batch_size = int(batch_size_input) if batch_size_input else 32
        except:
            batch_size = 32
        
        print(f"📈 Training for {epochs} epochs with batch size {batch_size}...")
        
        history = trainer.train(
            X_train, y_train, X_val, y_val, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Plot training history
        trainer.plot_training_history('results/training_history.png')
        
        # Save the model
        trainer.save_model()
        
        # Step 4: Evaluate model
        print("\n4. 📊 Evaluating model...")
        evaluator = ModelEvaluator(model)
        
        # Convert y_val from one-hot to labels
        y_val_labels = np.argmax(y_val, axis=1)
        
        # Make predictions
        _, y_pred = evaluator.predict(X_val)
        
        # Evaluate
        loss, accuracy = evaluator.evaluate(X_val, y_val)
        
        # Confusion matrix
        try:
            cm = evaluator.plot_confusion_matrix(y_val_labels, y_pred)
        except:
            print("⚠️  Could not plot confusion matrix")
        
        # Classification report
        try:
            report = evaluator.get_classification_report(y_val_labels, y_pred)
        except:
            print("⚠️  Could not generate classification report")
        
        # Analyze misclassifications
        try:
            misclassified_idx = evaluator.analyze_misclassifications(X_val, y_val_labels, y_pred)
        except:
            print("⚠️  Could not analyze misclassifications")
        
        # Step 5: Make predictions on test set
        print("\n5. 🔮 Making predictions on test set...")
        predictor = DigitPredictor(MODEL_PATH)
        
        # Predict on test data
        test_predictions, test_pred_labels = predictor.predict_batch(X_test)
        
        # Visualize predictions
        try:
            predictor.visualize_predictions(X_test, test_predictions)
        except:
            print("⚠️  Could not visualize predictions")
        
        # Create submission file
        create_submission(test_pred_labels, 'submission.csv')
        
        # Save results
        save_results(history, accuracy)
        
        print("\n" + "=" * 50)
        print("🎉 PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"\n📁 Model saved to: {MODEL_PATH}")
        print(f"📈 Validation accuracy: {accuracy*100:.2f}%")
        print(f"📄 Submission file created: submission.csv")
        print(f"📊 Results saved to: results/")
        print(f"\n🌐 To run the web app: python app/simple_main.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that train.csv and test.csv are in the data/ folder")
        print("2. Make sure test.csv doesn't have label column (should be 784 columns)")
        print("3. If test.csv has 785 columns, it might include labels")
        print("4. Try removing the label column from test.csv")
        print("\n📋 Stack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main()