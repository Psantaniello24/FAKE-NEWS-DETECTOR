from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import math
import os

class FakeNewsDetector:
    def __init__(self, model_name: str = "roberta-base"):
        """
        Initialize the fake news detector with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use (default: "roberta-base")
            
        Note:
            This model uses the following label mapping:
            - 0: Real news
            - 1: Fake news
        """
        # Check CUDA availability and print GPU info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # Enable TF32 if available (for NVIDIA Ampere GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set memory allocation policy
            if hasattr(torch.cuda, 'memory_stats'):
                print(f"🧠 Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("⚠️ CUDA not available. Using CPU for training.")
        
        # Initialize tokenizer and model
        print(f"🔄 Loading model: {model_name}")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Free up cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load model with optimizations
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification: fake (0) or real (1)
        )
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
        # Enable mixed precision training if CUDA is available
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            print(f"🧠 GPU memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    def tokenize_texts(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts using the model's tokenizer.
        """
        try:
            print(f"Tokenizing {len(texts)} texts with max_length={max_length}")
            if len(texts) > 0:
                print(f"Sample text for tokenization: '{texts[0][:100]}...'")
            
            # Check if input is valid
            if not all(isinstance(text, str) for text in texts):
                print("Warning: Not all items in texts are strings. Converting to strings.")
                texts = [str(text) for text in texts]
            
            # Tokenize the texts
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            print(f"Moving tokenized inputs to {self.device}")
            encodings = encodings.to(self.device)
            
            # Print tokenization stats
            print(f"Tokenization complete. Input shape: {encodings['input_ids'].shape}")
            return encodings
            
        except Exception as e:
            import traceback
            print(f"Error in tokenize_texts: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def create_data_loader(self, texts: List[str], labels: List[int], batch_size: int, max_length: int) -> List[Tuple]:
        """
        Create efficient data batches for training.
        """
        # Process data in chunks to avoid OOM
        batches = []
        
        # For low memory GPUs (4GB), process in smaller chunks
        chunk_size = 100  # Process 100 samples at a time to avoid OOM
        
        for start_idx in range(0, len(texts), chunk_size):
            end_idx = min(start_idx + chunk_size, len(texts))
            chunk_texts = texts[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_labels = chunk_labels[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenize_texts(batch_texts, max_length)
                
                # Convert labels to tensor
                label_tensor = torch.tensor(batch_labels).to(self.device)
                
                # Add to batches
                batches.append((encodings, label_tensor))
                
                # Clear CUDA cache to prevent OOM
                if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        return batches
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              epochs: int = 3, batch_size: int = 8,
              learning_rate: float = 2e-5, max_length: int = 256,
              use_amp: bool = False, gradient_accumulation_steps: int = 1,
              val_batch_size: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model on the provided data with CUDA acceleration.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts
            val_labels: List of validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            use_amp: Whether to use automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            val_batch_size: Validation batch size (defaults to batch_size if None)
        
        Returns:
            Dictionary containing training history
        """
        # Configure for low memory setting (4GB GPU)
        use_amp = True  # Always use mixed precision for 4GB GPUs
        
        # Use separate batch size for validation if specified
        if val_batch_size is None:
            val_batch_size = batch_size
        
        # Set up training parameters
        num_training_steps = math.ceil(len(train_texts) / (batch_size * gradient_accumulation_steps)) * epochs
        
        # Prepare optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Create data loaders for better memory efficiency
        print(f"🔄 Creating efficient data batches for 4GB GPU...")
        train_batches = self.create_data_loader(train_texts, train_labels, batch_size, max_length)
        print(f"✓ Created {len(train_batches)} training batches")
        
        # Use history to track metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Clear memory before each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Training
            self.model.train()
            total_train_loss = 0
            total_train_steps = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Progress bar
            pbar = tqdm(train_batches, desc="Training")
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Track batch loss for accumulation
            accumulated_loss = 0
            
            for step, (batch_encoding, batch_labels) in enumerate(pbar):
                # Forward pass with mixed precision if enabled
                if use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(
                            input_ids=batch_encoding['input_ids'],
                            attention_mask=batch_encoding['attention_mask'],
                            labels=batch_labels
                        )
                        loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
                    
                    # Accumulate gradients
                    self.scaler.scale(loss).backward()
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    
                    # Update weights if needed
                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_batches):
                        # Clip gradients to prevent exploding gradients
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Update weights
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Add to total loss
                        total_train_loss += accumulated_loss
                        total_train_steps += 1
                        accumulated_loss = 0
                        
                        # Clear cache periodically 
                        if step % 8 == 0:
                            torch.cuda.empty_cache()
                else:
                    outputs = self.model(
                        input_ids=batch_encoding['input_ids'],
                        attention_mask=batch_encoding['attention_mask'],
                        labels=batch_labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
                    
                    # Accumulate gradients
                    loss.backward()
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    
                    # Update weights if needed
                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_batches):
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Update weights
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Add to total loss
                        total_train_loss += accumulated_loss
                        total_train_steps += 1
                        accumulated_loss = 0
                        
                        # Clear cache periodically
                        if step % 8 == 0:
                            torch.cuda.empty_cache()
                
                # Calculate accuracy for this batch
                with torch.no_grad():
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                    total_predictions += len(batch_labels)
                
                # Update progress bar with GPU memory info if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'acc': f'{correct_predictions/total_predictions:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'GPU': f'{gpu_memory:.1f}GB'
                    })
                else:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'acc': f'{correct_predictions/total_predictions:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Delete unnecessary tensors to free up memory
                del outputs
            
            # Calculate average loss and accuracy for the epoch
            avg_train_loss = total_train_loss / total_train_steps if total_train_steps > 0 else 0
            train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Validation
            print("\n🔍 Running validation...")
            self.model.eval()
            
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Process validation data in batches to avoid OOM
            val_batches = self.create_data_loader(val_texts, val_labels, val_batch_size, max_length)
            print(f"✓ Created {len(val_batches)} validation batches (batch size: {val_batch_size})")
            
            total_val_loss = 0
            correct_val_predictions = 0
            total_val_predictions = 0
            
            # Run validation in batches
            with torch.no_grad():
                for val_batch_encoding, val_batch_labels in tqdm(val_batches, desc="Validating"):
                    if use_amp and self.scaler is not None:
                        with torch.amp.autocast(device_type='cuda'):
                            val_outputs = self.model(
                                input_ids=val_batch_encoding['input_ids'],
                                attention_mask=val_batch_encoding['attention_mask'],
                                labels=val_batch_labels
                            )
                            batch_val_loss = val_outputs.loss.item()
                    else:
                        val_outputs = self.model(
                            input_ids=val_batch_encoding['input_ids'],
                            attention_mask=val_batch_encoding['attention_mask'],
                            labels=val_batch_labels
                        )
                        batch_val_loss = val_outputs.loss.item()
                    
                    # Accumulate loss
                    total_val_loss += batch_val_loss
                    
                    # Calculate batch accuracy
                    val_batch_predictions = torch.argmax(val_outputs.logits, dim=1)
                    correct_val_predictions += (val_batch_predictions == val_batch_labels).sum().item()
                    total_val_predictions += len(val_batch_labels)
                    
                    # Clear memory after each batch
                    del val_outputs, val_batch_predictions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Calculate average validation loss and accuracy
            val_loss = total_val_loss / len(val_batches) if len(val_batches) > 0 else 0
            val_acc = correct_val_predictions / total_val_predictions if total_val_predictions > 0 else 0
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"✓ Training Loss: {avg_train_loss:.4f}")
            print(f"✓ Training Accuracy: {train_acc:.4f}")
            print(f"✓ Validation Loss: {val_loss:.4f}")
            print(f"✓ Validation Accuracy: {val_acc:.4f}")
            print(f"✓ Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"✨ New best validation accuracy: {best_val_acc:.4f}")
                
                # Save checkpoint of best model
                torch.save(self.model.state_dict(), "best_model_checkpoint.pt")
                print(f"✓ Saved checkpoint of best model")
            
            # Free up memory at the end of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Load the best model
        print(f"🔄 Loading best model (validation accuracy: {best_val_acc:.4f})...")
        self.model.load_state_dict(torch.load("best_model_checkpoint.pt"))
        
        return history
    
    def predict(self, texts: List[str], batch_size: int = 16, max_length: int = 256) -> List[int]:
        """
        Make predictions on a list of texts.
        
        Returns a list of predictions:
          - 0: Real news
          - 1: Fake news
        """
        try:
            print(f"Starting prediction with batch_size={batch_size}, max_length={max_length}")
            
            # For 4GB GPUs, use smaller batch size
            batch_size = min(batch_size, 4)
            print(f"Using batch size: {batch_size}")
            
            # Set model to evaluation mode
            self.model.eval()
            predictions = []
            
            print(f"Processing {len(texts)} text samples")
            
            # Process data in batches to avoid OOM
            for i in range(0, len(texts), batch_size):
                try:
                    batch_texts = texts[i:i + batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{math.ceil(len(texts)/batch_size)}, sample text: {batch_texts[0][:50]}...")
                    
                    print("Tokenizing batch...")
                    encodings = self.tokenize_texts(batch_texts, max_length)
                    
                    print("Running prediction...")
                    with torch.no_grad():
                        outputs = self.model(**encodings)
                        batch_predictions = torch.argmax(outputs.logits, dim=1)
                        batch_result = batch_predictions.cpu().tolist()
                        predictions.extend(batch_result)
                    
                    print(f"Batch predictions: {batch_result}")
                    
                    # Clear cache to prevent OOM
                    if torch.cuda.is_available() and i % (batch_size * 8) == 0:
                        print("Clearing CUDA cache")
                        torch.cuda.empty_cache()
                        
                except Exception as batch_error:
                    import traceback
                    print(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
                    print(traceback.format_exc())
                    # In case of batch error, add default predictions (0 for fake news)
                    predictions.extend([0] * len(batch_texts))
            
            print(f"Prediction complete. Results: {predictions}")
            return predictions
            
        except Exception as e:
            import traceback
            print(f"Error in predict method: {str(e)}")
            print(traceback.format_exc())
            # Return default prediction in case of error
            return [0] * len(texts)
    
    def evaluate(self, test_texts: List[str], test_labels: List[int], batch_size: int = 16, max_length: int = 256) -> dict:
        """
        Evaluate the model on test data and return metrics.
        """
        # For 4GB GPUs, use smaller batch size
        batch_size = min(batch_size, 4)
        
        predictions = self.predict(test_texts, batch_size, max_length)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, predictions),
            'f1_score': f1_score(test_labels, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(test_labels, predictions)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """
        Plot the confusion matrix.
        """
        # Clear memory before plotting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create figure with smaller size
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Fake', 'Real'])
        plt.yticks([0.5, 1.5], ['Fake', 'Real'])
        plt.tight_layout()
        
        # Save to file with lower dpi to save memory
        plt.savefig('confusion_matrix.png', dpi=200)
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            print("✓ Confusion matrix has been saved to confusion_matrix.png")
        
        # Clear plot from memory
        plt.close()
        
        # Report classification results
        print(f"✓ Classification Report:")
        print(f"  - True Negatives (Correctly identified as fake): {cm[0][0]}")
        print(f"  - False Positives (Fake news classified as real): {cm[0][1]}")
        print(f"  - False Negatives (Real news classified as fake): {cm[1][0]}")
        print(f"  - True Positives (Correctly identified as real): {cm[1][1]}")
        
        # Calculate metrics
        total = np.sum(cm)
        accuracy = (cm[0][0] + cm[1][1]) / total
        precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
        recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
    
    def save_model(self, path: str):
        """
        Save the model and tokenizer to disk.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Model and tokenizer saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'FakeNewsDetector':
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model directory or checkpoint file
        
        Returns:
            A FakeNewsDetector instance with the loaded model
        """
        try:
            # Print clear messages about what's happening
            print(f"🔄 Attempting to load model from: {path}")
            
            # Check if path exists
            if not os.path.exists(path):
                print(f"❌ Error: Model path '{path}' does not exist!")
                raise FileNotFoundError(f"Model path '{path}' does not exist")
            
            # Create a detector instance
            instance = cls("distilroberta-base")  # Initialize with same base model
            
            # Free memory before loading
            if torch.cuda.is_available():
                print("Clearing CUDA cache before loading model...")
                torch.cuda.empty_cache()
                gc.collect()
            
            # Use try/except for each step to identify where errors occur
            try:
                # Check if this is a directory (for transformers model) or a .pt file (for torch.save)
                if os.path.isdir(path):
                    print(f"Loading model from directory: {path}")
                    # Try to load model from directory
                    instance.model = AutoModelForSequenceClassification.from_pretrained(path)
                    instance.tokenizer = AutoTokenizer.from_pretrained(path)
                elif path.endswith('.pt') or path.endswith('.bin'):
                    print(f"Loading model from checkpoint file: {path}")
                    # Load checkpoint
                    checkpoint = torch.load(path, map_location=instance.device)
                    
                    # Check if checkpoint is a state_dict or a complete model
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # Handle checkpoints that contain more than just the state dict
                        print("Loading state_dict from checkpoint...")
                        instance.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Directly load the state dict
                        print("Loading state_dict directly...")
                        instance.model.load_state_dict(checkpoint)
                else:
                    print(f"⚠️ Warning: Unclear path format. Attempting to load as Hugging Face model.")
                    # Try both approaches
                    try:
                        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
                        instance.tokenizer = AutoTokenizer.from_pretrained(path)
                    except Exception as hf_error:
                        print(f"Failed to load as HuggingFace model: {str(hf_error)}")
                        # Try as a checkpoint file
                        if os.path.isfile(path):
                            checkpoint = torch.load(path, map_location=instance.device)
                            instance.model.load_state_dict(checkpoint)
                
                # Move model to device
                print(f"Moving model to {instance.device}...")
                instance.model = instance.model.to(instance.device)
                
                # Print success message
                print(f"✓ Model successfully loaded from {path}")
                return instance
                
            except Exception as model_error:
                print(f"❌ Error loading model components: {str(model_error)}")
                import traceback
                print(traceback.format_exc())
                raise
                
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise 