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
    neighbors: Dict[int, Optional['Node']]  # Action enum -> neighbor Node (or None)
    
    @classmethod
    def create(cls, x, y):
        """Create a Node from coordinates (translated from __init__)."""
        return cls(
            position=Vector2(
                x=jnp.array(x, dtype=jnp.float32),
                y=jnp.array(y, dtype=jnp.float32)
            ),
            neighbors={
                Action.UP: None,
                Action.DOWN: None,
                Action.LEFT: None,
                Action.RIGHT: None,
            }
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
        
        # Connect nodes
        cls.connect_horizontally(data, nodes_lut, tile_size)
        cls.connect_vertically(data, nodes_lut, tile_size)
        
        # Convert to list (maintain order for consistency)
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
        node_symbols = ['+']
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    nodes_lut[(x, y)] = Node.create(x, y)
    
    @staticmethod
    def connect_horizontally(data, nodes_lut, tile_size, xoffset=0, yoffset=0):
        """Connect nodes horizontally (translated from connectHorizontally)."""
        node_symbols = ['+']
        path_symbols = ['.']
        
        for row in range(data.shape[0]):
            key = None
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    if key is None:
                        key = (x, y)
                    else:
                        other_key = (x, y)
                        # Connect nodes
                        node1 = nodes_lut[key]
                        node2 = nodes_lut[other_key]
                        node1 = Node(
                            position=node1.position,
                            neighbors={
                                **node1.neighbors,
                                Action.RIGHT: node2
                            }
                        )
                        node2 = Node(
                            position=node2.position,
                            neighbors={
                                **node2.neighbors,
                                Action.LEFT: node1
                            }
                        )
                        nodes_lut[key] = node1
                        nodes_lut[other_key] = node2
                        key = other_key
                elif data[row][col] not in path_symbols:
                    # Hit a wall, reset chain
                    key = None
    
    @staticmethod
    def connect_vertically(data, nodes_lut, tile_size, xoffset=0, yoffset=0):
        """Connect nodes vertically (translated from connectVertically)."""
        node_symbols = ['+']
        path_symbols = ['.']
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
                        # Connect nodes
                        node1 = nodes_lut[key]
                        node2 = nodes_lut[other_key]
                        node1 = Node(
                            position=node1.position,
                            neighbors={
                                **node1.neighbors,
                                Action.DOWN: node2
                            }
                        )
                        node2 = Node(
                            position=node2.position,
                            neighbors={
                                **node2.neighbors,
                                Action.UP: node1
                            }
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
            for direction, neighbor in node.neighbors.items():
                if neighbor is not None:
                    # Find neighbor index
                    for n_idx, n in enumerate(self.nodeList):
                        n_x = float(n.position.x)
                        n_y = float(n.position.y)
                        if abs(n_x - float(neighbor.position.x)) < 0.001 and abs(n_y - float(neighbor.position.y)) < 0.001:
                            dir_name = {Action.UP: "UP", Action.DOWN: "DOWN", Action.LEFT: "LEFT", Action.RIGHT: "RIGHT"}.get(direction, f"DIR{direction}")
                            connections.append(f"{dir_name}->node{n_idx}")
                            break
            
            if connections:
                print(f"Node {idx}: pos=({int(x)}, {int(y)}) -> {', '.join(connections)}")
            else:
                print(f"Node {idx}: pos=({int(x)}, {int(y)}) -> (no connections)")
        
        print("=" * 50 + "\n")
    
