#!/usr/bin/env python3
"""
Simplified test script to verify modality dropout code correctness.

This script performs static code analysis without running the full model:
1. Checks that configuration parameters are properly defined
2. Verifies the logic flow in the forward method
3. Validates the code structure

Usage:
    python test_modality_dropout_simple.py
"""

import ast
import sys


def test_configuration_parameters():
    """Test 1: Verify configuration parameters are correctly defined."""
    print("=" * 70)
    print("Test 1: Configuration parameters verification")
    print("=" * 70)

    config_file = "src/lerobot/policies/act/configuration_act.py"

    try:
        with open(config_file, 'r') as f:
            content = f.read()

        # Check for use_modality_dropout
        if "use_modality_dropout: bool = False" in content:
            print("  ✅ use_modality_dropout parameter found (default: False)")
        else:
            print("  ❌ use_modality_dropout parameter not found or incorrect default")
            return False

        # Check for modality_dropout_prob
        if "modality_dropout_prob: float = 0.3" in content:
            print("  ✅ modality_dropout_prob parameter found (default: 0.3)")
        else:
            print("  ❌ modality_dropout_prob parameter not found or incorrect default")
            return False

        # Check for documentation
        if "Modality Dropout" in content or "modality dropout" in content:
            print("  ✅ Documentation for modality dropout found")
        else:
            print("  ⚠️  Documentation might be missing")

        print("\n✅ Test 1 PASSED: Configuration parameters are correctly defined\n")
        return True

    except FileNotFoundError:
        print(f"  ❌ FAILED: {config_file} not found\n")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def test_forward_method_logic():
    """Test 2: Verify the modality dropout logic in forward method."""
    print("=" * 70)
    print("Test 2: Forward method logic verification")
    print("=" * 70)

    modeling_file = "src/lerobot/policies/act/modeling_act.py"

    try:
        with open(modeling_file, 'r') as f:
            content = f.read()

        # Check for modality dropout condition
        if "if self.config.use_modality_dropout and self.training:" in content:
            print("  ✅ Dropout condition checks config flag and training mode")
        else:
            print("  ❌ Dropout condition not found or incorrect")
            return False

        # Check for OBS_STATE presence check
        if "if OBS_STATE in batch:" in content:
            print("  ✅ Defensive check for OBS_STATE in batch")
        else:
            print("  ⚠️  OBS_STATE check might be missing")

        # Check for probability check
        if "torch.rand" in content and "modality_dropout_prob" in content:
            print("  ✅ Random probability check implemented")
        else:
            print("  ❌ Probability check not found")
            return False

        # Check for zeroing operation
        if "torch.zeros_like(batch[OBS_STATE])" in content:
            print("  ✅ State zeroing operation found")
        else:
            print("  ❌ State zeroing operation not found")
            return False

        # Check for shallow copy
        if content.count("batch = dict(batch)") >= 1:
            print("  ✅ Shallow copy to avoid modifying original batch")
        else:
            print("  ⚠️  Shallow copy might be missing")

        print("\n✅ Test 2 PASSED: Forward method logic is correctly implemented\n")
        return True

    except FileNotFoundError:
        print(f"  ❌ FAILED: {modeling_file} not found\n")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def test_code_structure():
    """Test 3: Verify code structure and positioning."""
    print("=" * 70)
    print("Test 3: Code structure and positioning")
    print("=" * 70)

    modeling_file = "src/lerobot/policies/act/modeling_act.py"

    try:
        with open(modeling_file, 'r') as f:
            lines = f.readlines()

        # Find the forward method
        forward_method_line = None
        for i, line in enumerate(lines):
            if "def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:" in line:
                forward_method_line = i
                break

        if forward_method_line is None:
            print("  ❌ FAILED: Could not find forward method\n")
            return False

        print(f"  ✅ Forward method found at line {forward_method_line + 1}")

        # Check if modality dropout is at the beginning of forward method
        dropout_line = None
        for i in range(forward_method_line, min(forward_method_line + 30, len(lines))):
            if "use_modality_dropout" in lines[i]:
                dropout_line = i
                break

        if dropout_line is None:
            print("  ❌ FAILED: Modality dropout logic not found in forward method\n")
            return False

        relative_position = dropout_line - forward_method_line
        print(f"  ✅ Modality dropout logic found at line {dropout_line + 1}")
        print(f"     (Position: {relative_position} lines after forward method definition)")

        if relative_position <= 15:
            print("  ✅ Logic is positioned early in the forward method (good practice)")
        else:
            print("  ⚠️  Logic might be positioned too late in the forward method")

        print("\n✅ Test 3 PASSED: Code structure is correct\n")
        return True

    except FileNotFoundError:
        print(f"  ❌ FAILED: {modeling_file} not found\n")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def test_backward_compatibility():
    """Test 4: Verify backward compatibility."""
    print("=" * 70)
    print("Test 4: Backward compatibility verification")
    print("=" * 70)

    config_file = "src/lerobot/policies/act/configuration_act.py"

    try:
        with open(config_file, 'r') as f:
            content = f.read()

        # Verify default is False
        if "use_modality_dropout: bool = False" in content:
            print("  ✅ Default value is False (preserves original behavior)")
        else:
            print("  ❌ FAILED: Default value is not False!")
            return False

        # Check that the parameter is properly typed
        if ": bool = False" in content and ": float = 0.3" in content:
            print("  ✅ Parameters are properly typed (bool and float)")
        else:
            print("  ⚠️  Type annotations might be missing")

        print("\n✅ Test 4 PASSED: Backward compatibility is preserved\n")
        return True

    except FileNotFoundError:
        print(f"  ❌ FAILED: {config_file} not found\n")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def test_documentation():
    """Test 5: Verify documentation quality."""
    print("=" * 70)
    print("Test 5: Documentation quality")
    print("=" * 70)

    config_file = "src/lerobot/policies/act/configuration_act.py"
    modeling_file = "src/lerobot/policies/act/modeling_act.py"

    doc_issues = []

    try:
        # Check configuration documentation
        with open(config_file, 'r') as f:
            config_content = f.read()

        if '"""' in config_content and "modality dropout" in config_content.lower():
            print("  ✅ Configuration parameters have docstrings")
        else:
            doc_issues.append("Configuration docstrings might be missing")

        # Check implementation documentation
        with open(modeling_file, 'r') as f:
            modeling_content = f.read()

        if "# Modality Dropout" in modeling_content or "# modality dropout" in modeling_content.lower():
            print("  ✅ Implementation has explanatory comments")
        else:
            doc_issues.append("Implementation comments might be missing")

        # Check for key concepts in documentation
        key_concepts = [
            ("visual reliance" in config_content.lower() or "visual reliance" in modeling_content.lower(),
             "Mentions visual reliance"),
            ("training" in config_content.lower() or "training" in modeling_content.lower(),
             "Explains training-only behavior"),
        ]

        for found, description in key_concepts:
            if found:
                print(f"  ✅ {description}")
            else:
                doc_issues.append(description)

        if not doc_issues:
            print("\n✅ Test 5 PASSED: Documentation is comprehensive\n")
            return True
        else:
            print("\n⚠️  Test 5 WARNING: Some documentation might be improved:")
            for issue in doc_issues:
                print(f"     - {issue}")
            print()
            return True  # Not a failure, just a warning

    except FileNotFoundError as e:
        print(f"  ❌ FAILED: File not found: {e}\n")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MODALITY DROPOUT STATIC VERIFICATION SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Configuration parameters", test_configuration_parameters),
        ("Forward method logic", test_forward_method_logic),
        ("Code structure", test_code_structure),
        ("Backward compatibility", test_backward_compatibility),
        ("Documentation quality", test_documentation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    total = len(results)
    passed_count = sum(1 for _, passed in results if passed)

    print(f"\nResult: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 All static checks passed! Code structure is correct.")
        print("   Next step: Run actual training to verify runtime behavior.")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Please review the code.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
