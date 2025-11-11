"""
System Verification Script
Tests all components before submission
"""

import os
import sys
import json


def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n" + "="*70)
    print("TESTING DEPENDENCIES")
    print("="*70)
    
    required_modules = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'requests': 'Requests',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def test_data_files():
    """Test if data files exist and are valid JSON."""
    print("\n" + "="*70)
    print("TESTING DATA FILES")
    print("="*70)
    
    files = {
        'data/output/alternative_abstracts.json': 'Relevant documents',
        'data/output/non_relevant_abstracts.json': 'Non-relevant documents'
    }
    
    all_valid = True
    
    for filename, description in files.items():
        if not os.path.exists(filename):
            print(f"✗ {description}: File not found ({filename})")
            all_valid = False
            continue
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ {description}: Found {len(data)} documents")
            
            # Check structure
            if len(data) == 0:
                print(f"  ⚠ Warning: File is empty")
                all_valid = False
            else:
                # Check first document
                first_doc = next(iter(data.values()))
                if 'abstract' not in first_doc:
                    print(f"  ⚠ Warning: Documents missing 'abstract' field")
                    all_valid = False
                elif not first_doc['abstract']:
                    print(f"  ⚠ Warning: Empty abstracts found")
                    all_valid = False
                else:
                    print(f"  ✓ Data structure looks valid")
        
        except json.JSONDecodeError:
            print(f"✗ {description}: Invalid JSON format")
            all_valid = False
        except Exception as e:
            print(f"✗ {description}: Error reading file - {e}")
            all_valid = False
    
    return all_valid


def test_code_files():
    """Test if all code files exist."""
    print("\n" + "="*70)
    print("TESTING CODE FILES")
    print("="*70)
    
    required_files = {
        'model/transformer_encoder.py': 'Main training script',
        'model/inference.py': 'Inference script',
        'model/data_inspection.py': 'Data inspection utility',
        'model/visualizations.py': 'Visualization script',
        'requirements.txt': 'Dependencies file',
        'README.md': 'Documentation'
    }
    
    all_exist = True
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✓ {description}: {filename}")
        else:
            print(f"✗ {description}: {filename} NOT FOUND")
            all_exist = False
    
    return all_exist


def test_training_script():
    """Test if the training script can be imported."""
    print("\n" + "="*70)
    print("TESTING TRAINING SCRIPT")
    print("="*70)
    
    try:
        from transformer_encoder import TransformerIRSystem
        print("✓ Training script can be imported")
        
        # Try to initialize system
        try:
            ir_system = TransformerIRSystem()
            print("✓ TransformerIRSystem can be initialized")
            return True
        except Exception as e:
            print(f"✗ Error initializing system: {e}")
            return False
    
    except ImportError as e:
        print(f"✗ Cannot import training script: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing training script: {e}")
        return False


def test_data_loading():
    """Test if data can be loaded."""
    print("\n" + "="*70)
    print("TESTING DATA LOADING")
    print("="*70)
    
    try:
        from transformer_encoder import TransformerIRSystem
        
        ir_system = TransformerIRSystem()
        relevant_texts, non_relevant_texts = ir_system.load_data(
            'data/output/alternative_abstracts.json',
            'data/output/non_relevant_abstracts.json'
        )
        
        print(f"✓ Relevant documents loaded: {len(relevant_texts)}")
        print(f"✓ Non-relevant documents loaded: {len(non_relevant_texts)}")
        
        # Check balance
        ratio = len(relevant_texts) / len(non_relevant_texts) if len(non_relevant_texts) > 0 else 0
        print(f"✓ Dataset balance ratio: {ratio:.2f}")
        
        if abs(ratio - 1.0) < 0.1:
            print("✓ Datasets are well balanced")
        else:
            print("⚠ Warning: Datasets are imbalanced")
        
        # Check sample text
        if relevant_texts and len(relevant_texts[0][0]) > 0:
            print(f"✓ Sample text length: {len(relevant_texts[0][0])} characters")
            return True
        else:
            print("✗ Sample text is empty")
            return False
    
    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_exists():
    """Test if trained model exists."""
    print("\n" + "="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    
    if os.path.exists('transformer_ir_model.pth'):
        file_size = os.path.getsize('transformer_ir_model.pth') / (1024 * 1024)  # MB
        print(f"✓ Trained model found: transformer_ir_model.pth ({file_size:.2f} MB)")
        
        # Try to load it
        try:
            from transformer_encoder import TransformerIRSystem
            import torch
            
            checkpoint = torch.load('transformer_ir_model.pth', map_location='cpu')
            
            if 'model_state' in checkpoint and 'vocab' in checkpoint:
                print("✓ Model file structure is valid")
                print(f"  Vocabulary size: {len(checkpoint['vocab'])}")
                return True
            else:
                print("✗ Model file structure is invalid")
                return False
        
        except Exception as e:
            print(f"⚠ Warning: Cannot load model file - {e}")
            return False
    else:
        print("⚠ No trained model found (transformer_ir_model.pth)")
        print("  You need to train the model first: python transformer_ir_system.py")
        return False


def test_inference():
    """Test if inference works."""
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70)
    
    if not os.path.exists('transformer_ir_model.pth'):
        print("⚠ Skipping inference test (no trained model)")
        return True
    
    try:
        from transformer_encoder import TransformerIRSystem
        
        ir_system = TransformerIRSystem()
        ir_system.load_model('transformer_ir_model.pth')
        
        # Test prediction
        test_text = "Polyphenol composition and antioxidant activity in strawberries"
        pred, conf = ir_system.predict(test_text)
        
        print(f"✓ Prediction successful")
        print(f"  Test text: {test_text}")
        print(f"  Prediction: {'Relevant' if pred == 1 else 'Non-relevant'}")
        print(f"  Confidence: {conf:.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print test summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Train the model: python transformer_ir_system.py")
        print("2. Generate visualizations: python visualizations.py")
        print("3. Test inference: python inference.py --mode interactive")
        return True
    else:
        print("\n⚠ SOME TESTS FAILED. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Ensure data files are in the current directory")
        print("- Check data file format (JSON)")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BIOMEDICAL IR SYSTEM - VERIFICATION TESTS")
    print("="*70)
    print("This script will verify that everything is set up correctly.")
    
    results = {}
    
    # Run tests
    results['Dependencies'] = test_dependencies()
    results['Data Files'] = test_data_files()
    results['Code Files'] = test_code_files()
    results['Training Script'] = test_training_script()
    results['Data Loading'] = test_data_loading()
    results['Trained Model'] = test_model_exists()
    results['Inference'] = test_inference()
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()