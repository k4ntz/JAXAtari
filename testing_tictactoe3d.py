#!/usr/bin/env python3
"""
Interactive 3D Tic-Tac-Toe Game - Play against AI
Run: python play_tictactoe3d.py
"""

import jax.numpy as jnp
from jaxatari.games.jax_tictactoe3d import JaxTicTactoe3D, TicTactoe3DConstants


def print_board(board):
    """Pretty print the 4x4x4 board layer by layer"""
    print("\n" + "="*60)
    for layer in range(4):
        print(f"\n📦 LAYER {layer} (z={layer}):")
        print("   ", " ".join(f"{i}" for i in range(4)))
        for y in range(4):
            row_str = f"y={y}: "
            for x in range(4):
                cell = board[x, y, layer]
                if cell == 1:
                    row_str += "🟢 "  # Player (green)
                elif cell == -1:
                    row_str += "🔴 "  # AI (red)
                else:
                    row_str += "⬜ "  # Empty
            print(row_str)
    print("\n" + "="*60)


def coords_to_action(x, y, z):
    """Convert (x, y, z) coordinates to action number (0-63)"""
    return x * 16 + y * 4 + z


def action_to_coords(action):
    """Convert action number (0-63) to (x, y, z) coordinates"""
    x = action // 16
    y = (action % 16) // 4
    z = action % 4
    return x, y, z


def get_player_move(state):
    """Get player's move from input"""
    while True:
        try:
            print("\n🟢 YOUR TURN (Player)")
            print("Enter move as: x y z (where each is 0-3)")
            print("Example: 0 0 0  or  2 1 3")
            user_input = input("Your move: ").strip()
            
            if user_input.lower() == 'help':
                print("\nBoard coordinates:")
                print("  x: 0-3 (left-right)")
                print("  y: 0-3 (top-bottom)")
                print("  z: 0-3 (layer/depth)")
                continue
            
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Invalid format! Use: x y z")
                continue
            
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            
            if not (0 <= x <= 3 and 0 <= y <= 3 and 0 <= z <= 3):
                print("❌ Coordinates must be 0-3!")
                continue
            
            action = coords_to_action(x, y, z)
            
            # Check if cell is empty
            if state.board[x, y, z] != 0:
                print(f"❌ Cell ({x},{y},{z}) is already occupied!")
                continue
            
            return action
        
        except ValueError:
            print("❌ Invalid input! Enter three numbers: x y z")
        except Exception as e:
            print(f"❌ Error: {e}")


def print_status(state, reward, done):
    """Print game status"""
    if state.winner == 1:
        print("\n🎉 YOU WIN! 🎉")
    elif state.winner == -1:
        print("\n😢 AI WINS!")
    elif state.winner == 2:
        print("\n🤝 DRAW GAME!")
    
    if done:
        print(f"Game Over! Final move count: {int(state.move_count)}")


def play_game():
    """Main game loop"""
    print("\n" + "="*60)
    print("🎮 3D TIC-TAC-TOE (JAX Version)")
    print("="*60)
    print("You are 🟢 (Player 1)")
    print("AI is 🔴 (Player 2)")
    print("\nType 'help' to see coordinate system")
    print("="*60)
    
    # Initialize environment
    env = JaxTicTactoe3D()
    obs, state = env.reset()
    
    move_count = 0
    
    while True:
        # Print current board
        print_board(jnp.array(state.board))
        
        if state.game_over:
            print_status(state, None, True)
            break
        
        # Player's turn
        player_action = get_player_move(state)
        print(f"\n✅ You played at {action_to_coords(player_action)}")
        
        obs, state, reward, done, info = env.step(state, player_action)
        move_count += 1
        
        print_board(jnp.array(state.board))
        
        if state.game_over:
            print(f"🟢 Player move #{move_count}")
            print_status(state, reward, done)
            break
        
        # AI's turn
        print(f"\n🔴 AI is thinking...")
        
        # AI uses first empty position (deterministic)
        flat_board = state.board.flatten()
        empty_positions = jnp.where(flat_board == 0, size=64, fill_value=64)[0]
        if int(empty_positions[0]) < 64:
            ai_action = int(empty_positions[0])
            print(f"🔴 AI played at {action_to_coords(ai_action)}")
            
            obs, state, reward, done, info = env.step(state, ai_action)
            move_count += 1
            
            print_board(jnp.array(state.board))
            
            if state.game_over:
                print(f"🔴 AI move #{move_count}")
                print_status(state, reward, done)
                break


def main():
    """Main entry point"""
    while True:
        try:
            play_game()
            
            # Ask to play again
            print("\n" + "="*60)
            play_again = input("Play again? (y/n): ").strip().lower()
            if play_again != 'y':
                print("Thanks for playing! 👋")
                break
        
        except KeyboardInterrupt:
            print("\n\nGame interrupted. Thanks for playing! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

