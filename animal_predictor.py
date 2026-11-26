"""
ANIMAL CLASSIFICATION - PREDICTION INTERFACE
=============================================
Use this script to classify new animal images after training the model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def print_section(text):
    """Print section header"""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)

def load_model_and_classes(model_path='animal_classifier_model.keras', 
                          classes_path='class_names.json'):
    """Load the trained model and class names"""
    print_section("Loading Model")
    
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file '{model_path}' not found!")
        print("Please train the model first using animal_classifier_complete.py")
        return None, None
    
    if not os.path.exists(classes_path):
        print(f"✗ Error: Classes file '{classes_path}' not found!")
        return None, None
    
    try:
        model = keras.models.load_model(model_path)
        
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Number of classes: {len(classes)}")
        print(f"✓ Classes: {', '.join(classes)}")
        
        return model, classes
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def predict_single_image(model, classes, image_path, img_size=(224, 224), show_plot=True):
    """
    Predict animal in a single image with visualization
    
    Args:
        model: Trained Keras model
        classes: List of class names
        image_path: Path to image file
        img_size: Target image size
        show_plot: Whether to display matplotlib plot
    
    Returns:
        prediction: Predicted class name
        confidence: Confidence score
        top3: Top 3 predictions list
    """
    if not os.path.exists(image_path):
        print(f"✗ File not found: {image_path}")
        return None, None, None
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(img_size)
        
        # Preprocess for model
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top3 = [(classes[i], float(predictions[i])) for i in top_3_indices]
        
        prediction = classes[predicted_idx]
        
        # Display results
        print_header("PREDICTION RESULTS")
        print(f"\n🎯 Predicted Animal: {prediction}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"\nTop 3 Predictions:")
        for i, (animal, conf) in enumerate(top3, 1):
            bar = "█" * int(conf * 20)
            print(f"  {i}. {animal:12s}: {conf:6.2%} {bar}")
        
        if show_plot:
            create_prediction_plot(img, prediction, confidence, top3, image_path)
        
        return prediction, confidence, top3
        
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return None, None, None

def create_prediction_plot(img, prediction, confidence, top3, image_path):
    """Create visualization of prediction results"""
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Image display
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis('off')
    
    # Color code based on confidence
    if confidence > 0.9:
        color = '#27ae60'  # Green
        emoji = '✓'
    elif confidence > 0.7:
        color = '#f39c12'  # Orange
        emoji = '⚠'
    else:
        color = '#e74c3c'  # Red
        emoji = '✗'
    
    ax1.set_title(
        f'{emoji} Predicted: {prediction}\nConfidence: {confidence:.2%}', 
        fontsize=16, 
        fontweight='bold',
        color=color,
        pad=20
    )
    
    # Confidence chart
    ax2 = plt.subplot(1, 2, 2)
    
    animals = [t[0] for t in top3]
    scores = [t[1] for t in top3]
    colors = ['#27ae60', '#3498db', '#95a5a6']
    
    # Create horizontal bar chart
    bars = ax2.barh(range(len(animals)), scores, color=colors, edgecolor='white', linewidth=2)
    
    # Customize chart
    ax2.set_yticks(range(len(animals)))
    ax2.set_yticklabels(animals, fontsize=12)
    ax2.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax2.set_title('Top 3 Predictions with Confidence Scores', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax2.text(
            width + 0.02, 
            bar.get_y() + bar.get_height()/2, 
            f'{score:.1%}', 
            ha='left', 
            va='center',
            fontsize=11,
            fontweight='bold',
            color='#2c3e50'
        )
    
    ax2.invert_yaxis()
    
    # Add file info at bottom
    plt.figtext(
        0.5, 0.02, 
        f'File: {os.path.basename(image_path)}', 
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('last_prediction.png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.show()
    
    print(f"\n✓ Visualization saved as 'last_prediction.png'")

def batch_predict(model, classes, folder_path, img_size=(224, 224)):
    """
    Predict all images in a folder
    
    Args:
        model: Trained Keras model
        classes: List of class names
        folder_path: Path to folder containing images
        img_size: Target image size
    
    Returns:
        results: List of prediction dictionaries
    """
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return []
    
    # Get all image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"✗ No images found in {folder_path}")
        return []
    
    print_header(f"BATCH PREDICTION: {len(image_files)} IMAGES")
    
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # Load and predict
            img = Image.open(img_path).convert('RGB').resize(img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            
            result = {
                'file': img_file,
                'prediction': classes[predicted_idx],
                'confidence': float(confidence),
                'path': img_path
            }
            results.append(result)
            
            # Progress indicator
            status = "✓" if confidence > 0.8 else "⚠" if confidence > 0.6 else "?"
            print(f"  [{i:3d}/{len(image_files)}] {status} {img_file:30s} → {classes[predicted_idx]:12s} ({confidence:.2%})")
            
        except Exception as e:
            print(f"  [{i:3d}/{len(image_files)}] ✗ {img_file:30s} → ERROR: {e}")
    
    # Summary statistics
    print_section("Batch Prediction Summary")
    
    if results:
        avg_confidence = np.mean([r['confidence'] for r in results])
        high_conf = sum(1 for r in results if r['confidence'] > 0.9)
        med_conf = sum(1 for r in results if 0.7 < r['confidence'] <= 0.9)
        low_conf = sum(1 for r in results if r['confidence'] <= 0.7)
        
        print(f"  Total processed: {len(results)}")
        print(f"  Average confidence: {avg_confidence:.2%}")
        print(f"  High confidence (>90%): {high_conf}")
        print(f"  Medium confidence (70-90%): {med_conf}")
        print(f"  Low confidence (<70%): {low_conf}")
        
        # Most common predictions
        from collections import Counter
        pred_counts = Counter([r['prediction'] for r in results])
        print(f"\n  Predicted class distribution:")
        for animal, count in pred_counts.most_common():
            print(f"    • {animal}: {count} images")
    
    return results

def interactive_mode(model, classes):
    """Run interactive prediction mode"""
    print_header("INTERACTIVE PREDICTION MODE")
    
    while True:
        print("\nOptions:")
        print("  1. 🖼️  Predict single image")
        print("  2. 📁 Predict batch (entire folder)")
        print("  3. ℹ️  Show model info")
        print("  4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip()
            if os.path.exists(image_path):
                predict_single_image(model, classes, image_path)
            else:
                print(f"✗ File not found: {image_path}")
                
        elif choice == '2':
            folder_path = input("\nEnter folder path: ").strip()
            batch_predict(model, classes, folder_path)
                
        elif choice == '3':
            print_section("Model Information")
            print(f"  Total classes: {len(classes)}")
            print(f"  Classes: {', '.join(classes)}")
            print(f"  Input size: 224x224x3")
            print(f"  Preprocessing: Normalize to [0,1]")
            
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("✗ Invalid choice. Please enter 1, 2, 3, or 4.")

def quick_predict_demo(model, classes):
    """Demonstrate prediction with a sample visualization"""
    print_section("Creating Sample Prediction Visualization")
    
    # Create a sample figure showing the prediction workflow
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    # Workflow diagram
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    steps = [
        ("1. Upload\nImage", "#3498db", 2),
        ("2. Preprocess\n224×224", "#e67e22", 5),
        ("3. CNN Feature\nExtraction", "#9b59b6", 8),
        ("4. Classification\nSoftmax", "#e74c3c", 11),
        ("5. Results\nTop-3 + Conf", "#27ae60", 14)
    ]
    
    for text, color, x in steps:
        # Box
        box = FancyBboxPatch(
            (x-1.2, 3), 2.4, 2,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='white',
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Text
        ax.text(x, 4, text, ha='center', va='center', 
               fontsize=10, color='white', fontweight='bold')
        
        # Arrow
        if x < 14:
            ax.annotate('', xy=(x+1.4, 4), xytext=(x+1.2, 4),
                       arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    
    # Title
    ax.text(8, 7, "Animal Classification Workflow", 
           ha='center', fontsize=18, fontweight='bold')
    
    # Example output
    ax.text(8, 1.5, 
           "Example Output: 'Lion' with 94.2% confidence (Top-3: Lion 94.2%, Tiger 3.8%, Cat 1.5%)",
           ha='center', fontsize=12, style='italic',
           bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor='#f39c12', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('prediction_workflow.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Workflow diagram saved as 'prediction_workflow.png'")

def main():
    """Main entry point"""
    print_header("ANIMAL IMAGE CLASSIFICATION - PREDICTION INTERFACE")
    
    # Load model
    model, classes = load_model_and_classes()
    
    if model is None or classes is None:
        print("\n✗ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Show workflow demo on first run
    if not os.path.exists('prediction_workflow.png'):
        quick_predict_demo(model, classes)
    
    # Run interactive mode
    interactive_mode(model, classes)

if __name__ == "__main__":
    main()