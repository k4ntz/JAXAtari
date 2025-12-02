from typing import NamedTuple, Optional, Dict
import jax.numpy as jnp
import numpy as np
import chex
import os

# Import Vector2 from the same directory
from .vector import Vector2
# Import Action enum
from jaxatari.environment import JAXAtariAction as Action


class Node(NamedTuple):
    """
    JAX-compatible node class representing a position in the maze.
    Translated from original Node class.
    """
    position: Vector2
    neighbor_indices: chex.Array  # JAX array: neighbor_indices[action] = node_index (-1 if no neighbor)
    
    @classmethod
    def create(cls, x, y):
        """Create a Node from coordinates (translated from __init__)."""
        # Initialize neighbor_indices with -1 (no neighbor) for all actions
        # Array size 18 to accommodate max Action value
        neighbor_indices = jnp.full(18, -1, dtype=jnp.int32)
        return cls(
            position=Vector2(
                x=jnp.array(x, dtype=jnp.float32),
                y=jnp.array(y, dtype=jnp.float32)
            ),
            neighbor_indices=neighbor_indices
        )


class NodeGroup(NamedTuple):
    """
    JAX-compatible group of nodes representing the maze graph.
    Translated from original NodeGroup class.
    """
    nodeList: list  # List of Node objects
    
    @classmethod
    def create_empty(cls):
        """Create an empty NodeGroup (translated from __init__)."""
        return cls(nodeList=[])
    
    @classmethod
    def from_maze_file(cls, maze_file_path, tile_size=8):
        """
        Create NodeGroup from maze file.
        Reads maze file and creates nodes at '+' positions, connecting them through '.' paths.
        If file doesn't exist, uses default map.
        
        Args:
            maze_file_path: Path to maze file (or None for default)
            tile_size: Size of each tile in pixels (default 8, matching TILE_SIZE in jax_pacman_simple)
        """
        # If file doesn't exist, use default map
        if maze_file_path is None or not os.path.exists(maze_file_path):
            return cls.create_default_map(tile_size)
        
        # Read maze file
        data = cls.read_maze_file(maze_file_path)
        
        # Create node table
        nodes_lut = {}
        cls.create_node_table(data, nodes_lut, tile_size)
        
        # Convert to list and create position->index mapping
        node_list = list(nodes_lut.values())
        position_to_index = {}
        for idx, node in enumerate(node_list):
            pos_key = (float(node.position.x), float(node.position.y))
            position_to_index[pos_key] = idx
        
        # Connect nodes (now using indices instead of Node objects)
        cls.connect_horizontally(data, nodes_lut, position_to_index, tile_size)
        cls.connect_vertically(data, nodes_lut, position_to_index, tile_size)
        
        # Update node_list with connected nodes
        node_list = list(nodes_lut.values())
        node_group = cls(nodeList=node_list)
        
        # Print node connections for debugging
        node_group.print_connections()
        
        return node_group
    
    @staticmethod
    def read_maze_file(textfile):
        """Read maze file (translated from readMazeFile)."""
        return np.loadtxt(textfile, dtype='<U1')
    
    @staticmethod
    def construct_key(col, row, tile_size):
        """Construct key from column and row (translated from constructKey)."""
        return col * tile_size, row * tile_size
    
    @staticmethod
    def create_node_table(data, nodes_lut, tile_size, xoffset=0, yoffset=0):
        """Create node table from maze data (translated from createNodeTable)."""
        node_symbols = ['+', 'H', 'o']
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    nodes_lut[(x, y)] = Node.create(x, y)
    
    @staticmethod
    def connect_horizontally(data, nodes_lut, position_to_index, tile_size, xoffset=0, yoffset=0):
        """Connect nodes horizontally (translated from connectHorizontally)."""
        node_symbols = ['+', 'H', 'o']
        path_symbols = ['.', 'H', 'o']
        
        for row in range(data.shape[0]):
            key = None
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    if key is None:
                        key = (x, y)
                    else:
                        other_key = (x, y)
                        # Get node indices from position
                        key_pos = (float(key[0]), float(key[1]))
                        other_key_pos = (float(other_key[0]), float(other_key[1]))
                        node1_idx = position_to_index.get(key_pos)
                        node2_idx = position_to_index.get(other_key_pos)
                        
                        if node1_idx is not None and node2_idx is not None:
                            # Connect nodes using indices
                            node1 = nodes_lut[key]
                            node2 = nodes_lut[other_key]
                            
                            # Update neighbor_indices array
                            node1_neighbors = node1.neighbor_indices.at[Action.RIGHT].set(node2_idx)
                            node2_neighbors = node2.neighbor_indices.at[Action.LEFT].set(node1_idx)
                            
                            node1 = Node(
                                position=node1.position,
                                neighbor_indices=node1_neighbors
                            )
                            node2 = Node(
                                position=node2.position,
                                neighbor_indices=node2_neighbors
                            )
                            nodes_lut[key] = node1
                            nodes_lut[other_key] = node2
                        key = other_key
                elif data[row][col] not in path_symbols:
                    # Hit a wall, reset chain
                    key = None
    
    @staticmethod
    def connect_vertically(data, nodes_lut, position_to_index, tile_size, xoffset=0, yoffset=0):
        """Connect nodes vertically (translated from connectVertically)."""
        node_symbols = ['+', 'H', 'o']
        path_symbols = ['.', 'H', 'o']
        dataT = data.transpose()
        
        for col in range(dataT.shape[0]):
            key = None
            for row in range(dataT.shape[1]):
                if dataT[col][row] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    if key is None:
                        key = (x, y)
                    else:
                        other_key = (x, y)
                        # Get node indices from position
                        key_pos = (float(key[0]), float(key[1]))
                        other_key_pos = (float(other_key[0]), float(other_key[1]))
                        node1_idx = position_to_index.get(key_pos)
                        node2_idx = position_to_index.get(other_key_pos)
                        
                        if node1_idx is not None and node2_idx is not None:
                            # Connect nodes using indices
                            node1 = nodes_lut[key]
                            node2 = nodes_lut[other_key]
                            
                            # Update neighbor_indices array
                            node1_neighbors = node1.neighbor_indices.at[Action.DOWN].set(node2_idx)
                            node2_neighbors = node2.neighbor_indices.at[Action.UP].set(node1_idx)
                            
                            node1 = Node(
                                position=node1.position,
                                neighbor_indices=node1_neighbors
                            )
                            node2 = Node(
                                position=node2.position,
                                neighbor_indices=node2_neighbors
                            )
                            nodes_lut[key] = node1
                            nodes_lut[other_key] = node2
                        key = other_key
                elif dataT[col][row] not in path_symbols:
                    # Hit a wall, reset chain
                    key = None
    
    def print_connections(self):
        """Print all node connections for debugging."""
        print("\n=== Node Connections ===")
        print(f"Total nodes: {len(self.nodeList)}\n")
        
        for idx, node in enumerate(self.nodeList):
            x = float(node.position.x)
            y = float(node.position.y)
            connections = []
            # Check all action directions
            for direction in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                neighbor_idx = int(node.neighbor_indices[direction])
                if neighbor_idx >= 0:
                    dir_name = {Action.UP: "UP", Action.DOWN: "DOWN", Action.LEFT: "LEFT", Action.RIGHT: "RIGHT"}.get(direction, f"DIR{direction}")
                    connections.append(f"{dir_name}->node{neighbor_idx}")
            
            if connections:
                print(f"Node {idx}: pos=({int(x)}, {int(y)}) -> {', '.join(connections)}")
            else:
                print(f"Node {idx}: pos=({int(x)}, {int(y)}) -> (no connections)")
        
        print("=" * 50 + "\n")
    
    @classmethod
    def create_default_map(cls, tile_size=8):
        """
        Create a default simple test map when no maze file is provided.
        Creates a simple grid of connected nodes.
        """
        nodes_lut = {}
        grid_size = 3
        spacing = 32  # 4 tiles apart
        
        for row in range(grid_size):
            for col in range(grid_size):
                x = col * spacing + 16
                y = row * spacing + 16
                nodes_lut[(x, y)] = Node.create(x, y)
        
        # Convert to list and create position->index mapping
        node_list = list(nodes_lut.values())
        position_to_index = {}
        for idx, node in enumerate(node_list):
            pos_key = (float(node.position.x), float(node.position.y))
            position_to_index[pos_key] = idx
        
        # Connect horizontally
        for row in range(grid_size):
            for col in range(grid_size - 1):
                x1 = col * spacing + 16
                y1 = row * spacing + 16
                x2 = (col + 1) * spacing + 16
                y2 = row * spacing + 16
                
                key1 = (x1, y1)
                key2 = (x2, y2)
                node1_idx = position_to_index[(float(x1), float(y1))]
                node2_idx = position_to_index[(float(x2), float(y2))]
                
                node1 = nodes_lut[key1]
                node2 = nodes_lut[key2]
                
                node1_neighbors = node1.neighbor_indices.at[Action.RIGHT].set(node2_idx)
                node2_neighbors = node2.neighbor_indices.at[Action.LEFT].set(node1_idx)
                
                node1 = Node(position=node1.position, neighbor_indices=node1_neighbors)
                node2 = Node(position=node2.position, neighbor_indices=node2_neighbors)
                nodes_lut[key1] = node1
                nodes_lut[key2] = node2
        
        # Connect vertically
        for col in range(grid_size):
            for row in range(grid_size - 1):
                x1 = col * spacing + 16
                y1 = row * spacing + 16
                x2 = col * spacing + 16
                y2 = (row + 1) * spacing + 16
                
                key1 = (x1, y1)
                key2 = (x2, y2)
                node1_idx = position_to_index[(float(x1), float(y1))]
                node2_idx = position_to_index[(float(x2), float(y2))]
                
                node1 = nodes_lut[key1]
                node2 = nodes_lut[key2]
                
                node1_neighbors = node1.neighbor_indices.at[Action.DOWN].set(node2_idx)
                node2_neighbors = node2.neighbor_indices.at[Action.UP].set(node1_idx)
                
                node1 = Node(position=node1.position, neighbor_indices=node1_neighbors)
                node2 = Node(position=node2.position, neighbor_indices=node2_neighbors)
                nodes_lut[key1] = node1
                nodes_lut[key2] = node2
        
        node_list = list(nodes_lut.values())
        return cls(nodeList=node_list)
    
