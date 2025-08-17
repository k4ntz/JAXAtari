#!/usr/bin/env python3
"""
Generalized script to compare the first rendered screen from JAXAtari vs ALE for any game.
Focuses on shape comparison and visual differences.
"""

import argparse
import importlib.util
import inspect
import numpy as np
import pygame
import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
from pathlib import Path
from typing import Any, Tuple, Optional

# Add the src directory to the path so we can import jaxatari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer


def load_game_environment(game_file_path: str) -> Tuple[JaxEnvironment, Optional[JAXGameRenderer]]:
    """
    Dynamically loads a game environment and the renderer from a .py file.
    It looks for a class that inherits from JaxEnvironment.
    """
    if not os.path.exists(game_file_path):
        raise FileNotFoundError(f"Game file not found: {game_file_path}")

    module_name = os.path.splitext(os.path.basename(game_file_path))[0]

    # Add the directory of the game file to sys.path to handle relative imports within the game file
    game_dir = os.path.dirname(os.path.abspath(game_file_path))
    if game_dir not in sys.path:
        sys.path.insert(0, game_dir)

    spec = importlib.util.spec_from_file_location(module_name, game_file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {game_file_path}")

    game_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(game_module)
    except Exception as e:
        if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
            sys.path.pop(0)
        raise ImportError(f"Could not execute module {module_name}: {e}")

    if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
        sys.path.pop(0)

    game = None
    renderer = None
    # Find the class that inherits from JaxEnvironment
    for name, obj in inspect.getmembers(game_module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            print(f"Found game environment: {name}")
            game = obj()  # Instantiate and return

        if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
            print(f"Found renderer: {name}")
            renderer = obj()

    if game is None:
        raise ImportError(f"No class found in {game_file_path} that inherits from JaxEnvironment")

    return game, renderer


def get_ale_frame(ale_game_name: str, seed: int = 42) -> Optional[np.ndarray]:
    """Get the first frame from ALE environment."""
    try:
        env = gym.make(f"ALE/{ale_game_name}", render_mode="rgb_array")
        obs, info = env.reset(seed=seed)
        frame = env.render()
        env.close()
        return frame
    except Exception as e:
        print(f"Error creating ALE {ale_game_name} environment: {e}")
        return None


def get_jaxatari_frame(jax_env: JaxEnvironment, seed: int = 42) -> Optional[np.ndarray]:
    """Get the first frame from JAXAtari environment."""
    try:
        import jax
        key = jax.random.PRNGKey(seed)
        obs, state = jax_env.reset(key)
        frame = jax_env.render(state)
        return frame
    except Exception as e:
        print(f"Error creating JAXAtari environment: {e}")
        return None


def compare_shapes(ale_frame: Optional[np.ndarray], jax_frame: Optional[np.ndarray], 
                   ale_game_name: str, jax_game_name: str):
    """Compare the shapes of the two frames."""
    print("\n=== SHAPE COMPARISON ===")
    
    if ale_frame is not None:
        print(f"ALE {ale_game_name} Frame shape: {ale_frame.shape}")
        print(f"ALE {ale_game_name} Frame dtype: {ale_frame.dtype}")
        print(f"ALE {ale_game_name} Frame min/max values: {ale_frame.min()}/{ale_frame.max()}")
    else:
        print(f"ALE {ale_game_name} Frame: None")
    
    if jax_frame is not None:
        print(f"JAXAtari {jax_game_name} Frame shape: {jax_frame.shape}")
        print(f"JAXAtari {jax_game_name} Frame dtype: {jax_frame.dtype}")
        print(f"JAXAtari {jax_game_name} Frame min/max values: {jax_frame.min()}/{jax_frame.max()}")
    else:
        print(f"JAXAtari {jax_game_name} Frame: None")
    
    if ale_frame is not None and jax_frame is not None:
        print(f"\nShape match: {ale_frame.shape == jax_frame.shape}")
        print(f"Dtype match: {ale_frame.dtype == jax_frame.dtype}")
        
        # Check if dimensions are compatible for comparison
        if len(ale_frame.shape) == len(jax_frame.shape):
            print(f"Dimension count match: True")
            for i, (ale_dim, jax_dim) in enumerate(zip(ale_frame.shape, jax_frame.shape)):
                print(f"  Dimension {i}: ALE={ale_dim}, JAXAtari={jax_dim}, Match={ale_dim == jax_dim}")
        else:
            print(f"Dimension count match: False")
            print(f"  ALE dimensions: {len(ale_frame.shape)}")
            print(f"  JAXAtari dimensions: {len(jax_frame.shape)}")


def normalize_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize frame to 0-255 range for display."""
    if frame is None:
        return None
    
    # Convert to float if needed
    if frame.dtype != np.float32 and frame.dtype != np.float64:
        frame = frame.astype(np.float32)
    
    # Normalize to 0-1 range
    if frame.max() > 1.0:
        frame = frame / 255.0
    
    # Convert back to 0-255 range
    return (frame * 255).astype(np.uint8)


def display_frames(ale_frame: Optional[np.ndarray], jax_frame: Optional[np.ndarray],
                   ale_game_name: str, jax_game_name: str, save_path: Optional[Path] = None):
    """Display both frames side by side."""
    print("\n=== VISUAL COMPARISON ===")
    
    # Normalize frames for display
    ale_norm = normalize_frame(ale_frame)
    jax_norm = normalize_frame(jax_frame)
    
    if ale_norm is None or jax_norm is None:
        print("Cannot display frames - one or both are None")
        return
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display ALE frame
    if ale_norm is not None:
        ax1.imshow(ale_norm)
        ax1.set_title(f'ALE {ale_game_name}\nShape: {ale_norm.shape}')
        ax1.axis('off')
    
    # Display JAXAtari frame
    if jax_norm is not None:
        ax2.imshow(jax_norm)
        ax2.set_title(f'JAXAtari {jax_game_name}\nShape: {jax_norm.shape}')
        ax2.axis('off')
    
    # Display difference (if shapes match)
    if (ale_norm is not None and jax_norm is not None and 
        ale_norm.shape == jax_norm.shape):
        diff = np.abs(ale_norm.astype(np.int16) - jax_norm.astype(np.int16))
        im = ax3.imshow(diff, cmap='hot')
        ax3.set_title(f'Difference\nMax diff: {diff.max()}')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, 'Cannot compute\ndifference\n(shape mismatch)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Difference')
        ax3.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"render_comparison_{jax_game_name}.png", dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path / f'render_comparison_{jax_game_name}.png'}")
    
    plt.show()


def analyze_content(ale_frame: Optional[np.ndarray], jax_frame: Optional[np.ndarray],
                   ale_game_name: str, jax_game_name: str):
    """Analyze the content of both frames."""
    print("\n=== CONTENT ANALYSIS ===")
    
    if ale_frame is not None:
        print(f"ALE {ale_game_name} Frame:")
        print(f"  Unique values: {len(np.unique(ale_frame))}")
        print(f"  Most common values: {np.bincount(ale_frame.flatten())[:5]}")
        print(f"  Non-zero pixels: {np.count_nonzero(ale_frame)}")
    
    if jax_frame is not None:
        print(f"JAXAtari {jax_game_name} Frame:")
        print(f"  Unique values: {len(np.unique(jax_frame))}")
        print(f"  Most common values: {np.bincount(jax_frame.flatten())[:5]}")
        print(f"  Non-zero pixels: {np.count_nonzero(jax_frame)}")
    
    if ale_frame is not None and jax_frame is not None and ale_frame.shape == jax_frame.shape:
        # Calculate similarity metrics
        mse = np.mean((ale_frame.astype(np.float32) - jax_frame.astype(np.float32)) ** 2)
        mae = np.mean(np.abs(ale_frame.astype(np.float32) - jax_frame.astype(np.float32)))
        
        print(f"\nSimilarity Metrics:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        
        # Check if frames are identical
        identical = np.array_equal(ale_frame, jax_frame)
        print(f"  Identical: {identical}")
        
        return {
            'mse': mse,
            'mae': mae,
            'identical': identical,
            'shape_match': True
        }
    else:
        print("Cannot compute similarity metrics - shapes don't match or frames are None")
        return {
            'mse': None,
            'mae': None,
            'identical': None,
            'shape_match': False
        }


def save_comparison_results(results: dict, ale_game_name: str, jax_game_name: str, 
                           save_path: Optional[Path] = None):
    """Save comparison results to a file."""
    if save_path is None:
        return
    
    save_path.mkdir(parents=True, exist_ok=True)
    results_file = save_path / f"comparison_results_{jax_game_name}.txt"
    
    with open(results_file, "w") as f:
        print(f"=== Render Comparison Results ===", file=f)
        print(f"ALE Game: {ale_game_name}", file=f)
        print(f"JAXAtari Game: {jax_game_name}", file=f)
        print(f"Timestamp: {Path(__file__).stat().st_mtime}", file=f)
        print("", file=f)
        
        if results['shape_match']:
            print("✅ Shapes match!", file=f)
            print(f"Mean Squared Error: {results['mse']:.2f}", file=f)
            print(f"Mean Absolute Error: {results['mae']:.2f}", file=f)
            print(f"Frames Identical: {results['identical']}", file=f)
        else:
            print("❌ Shapes do not match!", file=f)
            print("This indicates a potential issue with the rendering implementation.", file=f)
    
    print(f"Results saved to: {results_file}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare JAXAtari vs ALE renders for any game.")
    parser.add_argument("--jax-game-path", type=str, required=True, 
                       help="Path to the Python file for the JAX game environment.")
    parser.add_argument("--ale-game-name", type=str, required=True, 
                       help="Name of the ALE ROM for Gymnasium (e.g., Pong-v5, Breakout-v5).")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for both environments.")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Directory to save results and plots (optional).")
    parser.add_argument("--no-display", action="store_true", 
                       help="Don't display the matplotlib plot (just save it).")
    
    args = parser.parse_args()
    
    # Load JAX environment
    try:
        print(f"Loading JAX game from: {args.jax_game_path}")
        jax_env, renderer = load_game_environment(args.jax_game_path)
        jax_game_name = Path(args.jax_game_path).stem
    except Exception as e:
        print(f"Fatal: Could not load JAX game environment: {e}")
        sys.exit(1)
    
    # Create output directory only if specified
    output_path = Path(args.output_dir) if args.output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Comparing JAXAtari {jax_game_name} vs ALE {args.ale_game_name} - First Frame")
    print("=" * 60)
    
    # Get frames
    print(f"Loading ALE {args.ale_game_name} frame...")
    ale_frame = get_ale_frame(args.ale_game_name, args.seed)
    
    print(f"Loading JAXAtari {jax_game_name} frame...")
    jax_frame = get_jaxatari_frame(jax_env, args.seed)
    
    # Compare shapes
    compare_shapes(ale_frame, jax_frame, args.ale_game_name, jax_game_name)
    
    # Analyze content
    results = analyze_content(ale_frame, jax_frame, args.ale_game_name, jax_game_name)
    
    # Display frames
    if not args.no_display:
        display_frames(ale_frame, jax_frame, args.ale_game_name, jax_game_name, output_path)
    else:
        display_frames(ale_frame, jax_frame, args.ale_game_name, jax_game_name, output_path)
        plt.close()  # Close without showing
    
    # Save results only if output directory is specified
    if output_path:
        save_comparison_results(results, args.ale_game_name, jax_game_name, output_path)
    
    print("\n=== SUMMARY ===")
    if ale_frame is not None and jax_frame is not None:
        if ale_frame.shape == jax_frame.shape:
            print("✅ Shapes match!")
        else:
            print("❌ Shapes do not match!")
            print("This indicates a potential issue with the rendering implementation.")
    else:
        print("❌ Could not load one or both frames!")
    
    if output_path:
        print(f"\nComparison complete! Results saved to: {output_path}")
    else:
        print(f"\nComparison complete!")


if __name__ == "__main__":
    main() 