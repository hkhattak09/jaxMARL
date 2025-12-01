"""Tests for llm_cfg.py - LLM configuration.

Run with: python tests/test_llm_cfg.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creating default LLM configuration."""
        print("Testing default LLM config creation...")
        
        from llm_cfg import LLMConfig
        
        config = LLMConfig()
        
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.task_name == "assembly"
        assert config.api_key is None  # Default is None (uses env var)
        
        print("  ✓ Default LLM config creation passed")
    
    def test_config_with_api_key(self):
        """Test config with explicit API key."""
        print("Testing LLM config with API key...")
        
        from llm_cfg import LLMConfig
        
        config = LLMConfig(api_key="sk-test-key")
        
        assert config.api_key == "sk-test-key"
        
        print("  ✓ LLM config with API key passed")
    
    def test_config_with_custom_model(self):
        """Test config with different model."""
        print("Testing LLM config with custom model...")
        
        from llm_cfg import LLMConfig
        
        config = LLMConfig(model="gpt-4-turbo")
        
        assert config.model == "gpt-4-turbo"
        
        print("  ✓ LLM config with custom model passed")


class TestCreateLLMConfig:
    """Tests for create_llm_config function."""
    
    def test_create_with_api_key_arg(self):
        """Test creating config with API key argument."""
        print("Testing create_llm_config with API key arg...")
        
        from llm_cfg import create_llm_config
        
        config = create_llm_config(api_key="sk-test-key")
        
        assert config.api_key == "sk-test-key"
        assert config.model == "gpt-4o"  # Default
        
        print("  ✓ create_llm_config with API key arg passed")
    
    def test_create_with_env_var(self):
        """Test creating config with environment variable."""
        print("Testing create_llm_config with env var...")
        
        from llm_cfg import create_llm_config
        
        # Set temporary env var
        original = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-env-test-key"
        
        try:
            config = create_llm_config()
            assert config.api_key == "sk-env-test-key"
            print("  ✓ create_llm_config with env var passed")
        finally:
            # Restore original
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
            else:
                del os.environ["OPENAI_API_KEY"]
    
    def test_create_without_api_key_raises(self):
        """Test that create_llm_config raises without API key."""
        print("Testing create_llm_config raises without key...")
        
        from llm_cfg import create_llm_config
        
        # Temporarily remove env var if set
        original = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            with pytest.raises(ValueError, match="API key required"):
                create_llm_config()
            print("  ✓ create_llm_config raises without key passed")
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
    
    def test_create_with_overrides(self):
        """Test creating config with overrides."""
        print("Testing create_llm_config with overrides...")
        
        from llm_cfg import create_llm_config
        
        config = create_llm_config(
            api_key="sk-test",
            model="gpt-3.5-turbo",
            temperature=0.5,
        )
        
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        
        print("  ✓ create_llm_config with overrides passed")


class TestGetAPIKey:
    """Tests for get_api_key function."""
    
    def test_get_from_config(self):
        """Test getting API key from config."""
        print("Testing get_api_key from config...")
        
        from llm_cfg import LLMConfig, get_api_key
        
        config = LLMConfig(api_key="sk-config-key")
        key = get_api_key(config)
        
        assert key == "sk-config-key"
        
        print("  ✓ get_api_key from config passed")
    
    def test_get_from_env(self):
        """Test getting API key from environment."""
        print("Testing get_api_key from environment...")
        
        from llm_cfg import LLMConfig, get_api_key
        
        config = LLMConfig()  # No api_key
        
        original = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-env-key"
        
        try:
            key = get_api_key(config)
            assert key == "sk-env-key"
            print("  ✓ get_api_key from environment passed")
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
            else:
                del os.environ["OPENAI_API_KEY"]
    
    def test_get_raises_when_missing(self):
        """Test get_api_key raises when no key available."""
        print("Testing get_api_key raises when missing...")
        
        from llm_cfg import LLMConfig, get_api_key
        
        config = LLMConfig()  # No api_key
        
        original = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            with pytest.raises(ValueError):
                get_api_key(config)
            print("  ✓ get_api_key raises when missing passed")
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original


class TestTaskDescriptions:
    """Tests for task description functions."""
    
    def test_get_task_description(self):
        """Test getting task description."""
        print("Testing get_task_description...")
        
        from llm_cfg import get_task_description
        
        desc = get_task_description("assembly")
        
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "robot" in desc.lower() or "swarm" in desc.lower()
        
        print("  ✓ get_task_description passed")
    
    def test_get_environment_description(self):
        """Test getting environment description."""
        print("Testing get_environment_description...")
        
        from llm_cfg import get_environment_description
        
        desc = get_environment_description("assembly")
        
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "robot" in desc.lower() or "2D" in desc or "target" in desc.lower()
        
        print("  ✓ get_environment_description passed")
    
    def test_get_robot_api_description(self):
        """Test getting robot API description."""
        print("Testing get_robot_api_description...")
        
        from llm_cfg import get_robot_api_description
        
        desc = get_robot_api_description("assembly")
        
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "def " in desc  # Should contain function definitions
        
        print("  ✓ get_robot_api_description passed")
    
    def test_unknown_task_falls_back(self):
        """Test that unknown task falls back to assembly."""
        print("Testing unknown task fallback...")
        
        from llm_cfg import get_task_description
        
        desc = get_task_description("unknown_task")
        assembly_desc = get_task_description("assembly")
        
        # Unknown should fallback to assembly
        assert desc == assembly_desc
        
        print("  ✓ Unknown task fallback passed")


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  LLM CONFIG TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    test_classes = [
        TestLLMConfig,
        TestCreateLLMConfig,
        TestGetAPIKey,
        TestTaskDescriptions,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except pytest.skip.Exception:
                    skipped += 1
                except Exception as e:
                    print(f"  ✗ {method_name} FAILED: {e}")
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
