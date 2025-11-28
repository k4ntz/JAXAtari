#!/usr/bin/env python3
"""
Visualize node connections from the maze file.
"""
import sys
import os
sys.path.insert(0, 'src')

from jaxatari.games.pacmanTry.nodes import NodeGroup
from jaxatari.environment import JAXAtariAction as Action

def visualize_nodes():
    """Print a detailed visualization of all nodes and their connections."""
    maze_file_path = os.path.join('src/jaxatari/games/pacmanTry', 'maze1.txt')
    node_group = NodeGroup.from_maze_file(maze_file_path, tile_size=8)
    
    print("\n" + "="*80)
    print("NODE CONNECTION VISUALIZATION")
    print("="*80)
    print(f"\nTotal nodes: {len(node_group.nodeList)}\n")
    
    # Print detailed node information
    print("-"*80)
    print("DETAILED NODE INFORMATION")
    print("-"*80)
    
    for idx, node in enumerate(node_group.nodeList):
        x = int(float(node.position.x))
        y = int(float(node.position.y))
        # Use consistent naming: Node 0, Node 1, etc. (same as grid)
        node_name = str(idx) if idx < 10 else chr(ord('A') + (idx - 10)) if idx < 36 else '?'
        print(f"\nNode {idx:2d} ({node_name}): Position ({x:3d}, {y:3d})")
        
        connections = []
        for direction, neighbor in node.neighbors.items():
            if neighbor is not None:
                # Find neighbor index
                for n_idx, n in enumerate(node_group.nodeList):
                    n_x = float(n.position.x)
                    n_y = float(n.position.y)
                    if abs(n_x - float(neighbor.position.x)) < 0.001 and abs(n_y - float(neighbor.position.y)) < 0.001:
                        n_name = str(n_idx) if n_idx < 10 else chr(ord('A') + (n_idx - 10)) if n_idx < 36 else '?'
                        dir_name = {
                            Action.UP: "UP   ",
                            Action.DOWN: "DOWN ",
                            Action.LEFT: "LEFT ",
                            Action.RIGHT: "RIGHT"
                        }.get(direction, f"DIR{direction}")
                        connections.append(f"  {dir_name} -> Node {n_idx:2d} ({n_name}) at ({int(n_x):3d}, {int(n_y):3d})")
                        break
        
        if connections:
            print("  Connections:")
            for conn in connections:
                print(conn)
        else:
            print("  No connections")
    
    # Create a visual map
    print("\n" + "="*80)
    print("VISUAL MAP (showing node positions and connections)")
    print("="*80)
    
    # Find bounds
    min_x = min(int(float(n.position.x)) for n in node_group.nodeList)
    max_x = max(int(float(n.position.x)) for n in node_group.nodeList)
    min_y = min(int(float(n.position.y)) for n in node_group.nodeList)
    max_y = max(int(float(n.position.y)) for n in node_group.nodeList)
    
    # Create a grid representation - use actual maze dimensions
    grid_width = (max_x - min_x) // 8 + 1
    grid_height = (max_y - min_y) // 8 + 1
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Mark nodes
    node_positions = {}
    for idx, node in enumerate(node_group.nodeList):
        x = int(float(node.position.x))
        y = int(float(node.position.y))
        # Normalize to grid coordinates (relative to min)
        grid_x = (x - min_x) // 8
        grid_y = (y - min_y) // 8
        if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
            # Use node index (mod 100 to fit in 2 chars)
            if idx < 10:
                grid[grid_y][grid_x] = str(idx)
            elif idx < 100:
                # Can't fit 2 chars, use hex
                grid[grid_y][grid_x] = chr(ord('A') + (idx - 10))
            else:
                grid[grid_y][grid_x] = '?'
            node_positions[idx] = (grid_x, grid_y)
    
    # Draw connections
    for idx, node in enumerate(node_group.nodeList):
        x = int(float(node.position.x))
        y = int(float(node.position.y))
        # Use grid coordinates relative to min
        grid_x = (x - min_x) // 8
        grid_y = (y - min_y) // 8
        
        for direction, neighbor in node.neighbors.items():
            if neighbor is not None:
                # Find neighbor index
                for n_idx, n in enumerate(node_group.nodeList):
                    n_x = float(n.position.x)
                    n_y = float(n.position.y)
                    if abs(n_x - float(neighbor.position.x)) < 0.001 and abs(n_y - float(neighbor.position.y)) < 0.001:
                        n_grid_x = (int(n_x) - min_x) // 8
                        n_grid_y = (int(n_y) - min_y) // 8
                        
                        # Draw line between nodes
                        if direction == Action.RIGHT:
                            for dx in range(grid_x + 1, n_grid_x):
                                if 0 <= grid_y < grid_height and 0 <= dx < grid_width:
                                    if grid[grid_y][dx] == ' ':
                                        grid[grid_y][dx] = '-'
                                    elif grid[grid_y][dx] not in ['-', '|', '+']:
                                        grid[grid_y][dx] = '+'
                        elif direction == Action.LEFT:
                            for dx in range(n_grid_x + 1, grid_x):
                                if 0 <= grid_y < grid_height and 0 <= dx < grid_width:
                                    if grid[grid_y][dx] == ' ':
                                        grid[grid_y][dx] = '-'
                                    elif grid[grid_y][dx] not in ['-', '|', '+']:
                                        grid[grid_y][dx] = '+'
                        elif direction == Action.DOWN:
                            for dy in range(grid_y + 1, n_grid_y):
                                if 0 <= dy < grid_height and 0 <= grid_x < grid_width:
                                    if grid[dy][grid_x] == ' ':
                                        grid[dy][grid_x] = '|'
                                    elif grid[dy][grid_x] not in ['-', '|', '+']:
                                        grid[dy][grid_x] = '+'
                        elif direction == Action.UP:
                            for dy in range(n_grid_y + 1, grid_y):
                                if 0 <= dy < grid_height and 0 <= grid_x < grid_width:
                                    if grid[dy][grid_x] == ' ':
                                        grid[dy][grid_x] = '|'
                                    elif grid[dy][grid_x] not in ['-', '|', '+']:
                                        grid[dy][grid_x] = '+'
                        break
    
    # Print grid
    print("\nGrid visualization (Node names: 0-9 = Node 0-9, A-Z = Node 10-35, - and | show connections):")
    print("Y-axis (tile coordinates)")
    for y in range(grid_height):
        row_str = ''.join(grid[y])
        if any(c != ' ' for c in row_str):
            actual_y = min_y // 8 + y
            print(f"{actual_y:3d} | {row_str}")
    
    print(f"\nNode name mapping: Numbers 0-9 = Node 0-9, letters A-Z = Node 10-35")
    print(f"Grid shows {len([c for row in grid for c in row if c not in [' ', '-', '|', '+']])} nodes marked")
    
    # Print connection matrix
    print("\n" + "="*80)
    print("CONNECTION MATRIX (1 = connected, 0 = not connected)")
    print("="*80)
    num_nodes = len(node_group.nodeList)
    print(f"\n{'Node':>6}", end='')
    for i in range(num_nodes):
        print(f"{i:>4}", end='')
    print()
    
    for i in range(num_nodes):
        print(f"{i:>6}", end='')
        for j in range(num_nodes):
            # Check if node i connects to node j
            connected = False
            node_i = node_group.nodeList[i]
            for neighbor in node_i.neighbors.values():
                if neighbor is not None:
                    n_x = float(neighbor.position.x)
                    n_y = float(neighbor.position.y)
                    j_x = float(node_group.nodeList[j].position.x)
                    j_y = float(node_group.nodeList[j].position.y)
                    if abs(n_x - j_x) < 0.001 and abs(n_y - j_y) < 0.001:
                        connected = True
                        break
            print(f"{'1' if connected else '0':>4}", end='')
        print()
    
    print("\n" + "="*80)

if __name__ == "__main__":
    visualize_nodes()

