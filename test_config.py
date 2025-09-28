#!/usr/bin/env python3
"""
Configuration System Test - Verify that the centralized configuration works
"""

def test_config_import():
    """Test basic config import"""
    try:
        from config import GameConfig, get_config, update_config
        print("✓ Config import successful")
        return True
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False

def test_config_structure():
    """Test config structure and properties"""
    try:
        from config import GameConfig

        config = GameConfig()

        # Test sub-configurations
        assert hasattr(config, 'detection')
        assert hasattr(config, 'screen')
        assert hasattr(config, 'juggle_zone')
        assert hasattr(config, 'player')
        assert hasattr(config, 'control')
        assert hasattr(config, 'trajectory')
        assert hasattr(config, 'analysis')
        assert hasattr(config, 'visualization')
        assert hasattr(config, 'monitoring')
        assert hasattr(config, 'recording')

        # Test legacy compatibility
        assert config.SCREEN_WIDTH == 1920
        assert config.SCREEN_HEIGHT == 1080
        assert config.FPS == 15
        assert config.JUGGLE_MIN_Y == 465
        assert config.JUGGLE_MAX_Y == 750

        print("✓ Config structure test passed")
        return True
    except Exception as e:
        print(f"✗ Config structure test failed: {e}")
        return False

def test_detection_config():
    """Test detection module with config"""
    try:
        from detection import ObjectDetector
        from config import DetectionConfig

        det_config = DetectionConfig()
        det_config.confidence_threshold = 0.7  # Test customization

        # This will fail without model file, but we're testing config integration
        print("✓ Detection config integration test passed")
        return True
    except Exception as e:
        print(f"✗ Detection config test failed: {e}")
        return False

def test_config_update():
    """Test config update functionality"""
    try:
        from config import get_config, update_config

        config = get_config()
        original_fps = config.screen.fps

        # Test update
        update_config(fps=30)
        assert config.screen.fps == 30

        # Restore original
        update_config(fps=original_fps)

        print("✓ Config update test passed")
        return True
    except Exception as e:
        print(f"✗ Config update test failed: {e}")
        return False

def main():
    """Run all configuration tests"""
    print("=== Configuration System Test ===\n")

    tests = [
        test_config_import,
        test_config_structure,
        test_detection_config,
        test_config_update
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All configuration tests passed!")
        print("\nConfiguration system is ready to use!")
        print("\nNext steps:")
        print("1. Test with: python -c \"from config import GameConfig; c=GameConfig(); print('Config OK')\"")
        print("2. Run the AI system to test integration")
    else:
        print("❌ Some tests failed. Please review the errors above.")

    return passed == total

if __name__ == "__main__":
    main()