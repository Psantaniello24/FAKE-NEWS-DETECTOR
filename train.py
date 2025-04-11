import argparse
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data
from model import FakeNewsDetector
import os
from tqdm import tqdm
import time
import torch
import gc
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train a fake news detector model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV file containing the dataset')
    parser.add_argument('--model_name', type=str, default='distilroberta-base',
                      help='Name of the pre-trained model to use')
    parser.add_argument('--output_dir', type=str, default='saved_model',
                      help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                      help='Proportion of training data to use for validation')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--use_amp', action='store_true',
                      help='Use Automatic Mixed Precision for faster training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of updates steps to accumulate before performing a backward/update pass')
    
    args = parser.parse_args()
    
    # Set CUDA memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set up CUDA for optimal performance
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Display CUDA info
        print("\n🔥 CUDA GPU Acceleration Enabled")
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"📊 CUDA Version: {torch.version.cuda}")
    else:
        print("\n⚠️ CUDA not available. Training will run on CPU (slow).")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    texts, labels = load_and_preprocess_data(args.data_path)
    print(f"✓ Loaded {len(texts)} articles")
    
    # Split data into train, validation, and test sets
    print("\n🔄 Splitting data into train, validation, and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=args.val_size, random_state=42, stratify=train_labels
    )
    
    print(f"✓ Training set: {len(train_texts)} articles")
    print(f"✓ Validation set: {len(val_texts)} articles")
    print(f"✓ Test set: {len(test_texts)} articles")
    
    # Initialize and train the model
    print("\n🤖 Initializing model...")
    detector = FakeNewsDetector(model_name=args.model_name)
    print(f"✓ Using model: {args.model_name}")
    print(f"✓ Device: {detector.device}")
    
    # Optimize memory usage
    print("\n🧹 Optimizing memory for 4GB GPU...")
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Try to reserve GPU memory
        try:
            torch.cuda.empty_cache()
            # Use less memory for a 4GB GPU
            torch.cuda.set_per_process_memory_fraction(0.6)  # Reserve only 60% of GPU memory
            print(f"✓ Memory reserved: 60% of available GPU memory")
        except:
            print("✓ Using default memory management")
    
    print("\n🚀 Starting training with low memory settings...")
    print(f"✓ Epochs: {args.epochs}")
    print(f"✓ Batch size: {args.batch_size} (reduced for 4GB GPU)")
    print(f"✓ Learning rate: {args.learning_rate}")
    print(f"✓ Max sequence length: {args.max_length} (reduced for 4GB GPU)")
    print(f"✓ Mixed precision: {'Enabled' if args.use_amp else 'Disabled'}")
    print(f"✓ Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Create smaller validation batch size
    val_batch_size = max(1, args.batch_size // 2)
    print(f"✓ Validation batch size: {val_batch_size} (half of training for memory)")
    
    start_time = time.time()
    history = detector.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        val_batch_size=val_batch_size
    )
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    
    # Optimize memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Evaluate the model
    print("\n📈 Evaluating model on test set...")
    metrics = detector.evaluate(test_texts, test_labels, batch_size=2, max_length=args.max_length)
    
    print("\n📊 Test Set Metrics:")
    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    print("\n📊 Plotting confusion matrix...")
    detector.plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Save the model
    print(f"\n💾 Saving model to {args.output_dir}...")
    detector.save_model(args.output_dir)
    print("✓ Training completed successfully!")
    
    # Print final summary
    print("\n🎉 Final Summary:")
    print(f"✓ Total training time: {training_time:.2f} seconds")
    print(f"✓ Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"✓ Final test accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Final test F1 score: {metrics['f1_score']:.4f}")
    
    # Print GPU memory stats
    if torch.cuda.is_available():
        print(f"✓ Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main() 