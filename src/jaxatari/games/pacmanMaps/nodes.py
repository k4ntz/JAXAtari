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
        
        Args:
            maze_file_path: Path to maze file (must exist)
            tile_size: Size of each tile in pixels (default 8, matching TILE_SIZE in jax_pacman_simple)
        
        Raises:
            FileNotFoundError: If maze file doesn't exist
        """
        # Check if file exists, raise error if not found
        if maze_file_path is None or not os.path.exists(maze_file_path):
            raise FileNotFoundError(f"Maze file not found: {maze_file_path}")
        
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
        cls.connect_portals(data, nodes_lut, position_to_index, tile_size)
        
        # Update node_list with connected nodes
        node_list = list(nodes_lut.values())
        node_group = cls(nodeList=node_list)
        
        # Print node connections for debugging
        node_group.print_connections()
        
        return node_group
    
    @staticmethod
    def read_maze_file(textfile):
        """Read maze file (translated from readMazeFile)."""
        with open(textfile, 'r') as f:
            lines = [list(line.strip().replace(' ', '')) for line in f.readlines()]
        return np.array(lines, dtype='<U1')
    
    @staticmethod
    def construct_key(col, row, tile_size):
        """Construct key from column and row (translated from constructKey)."""
        return col * tile_size, row * tile_size
    
    @staticmethod
    def create_node_table(data, nodes_lut, tile_size, xoffset=0, yoffset=32):
        """Create node table from maze data (translated from createNodeTable)."""
        node_symbols = ['+', 'H', 'o', 'D', 'P']
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row, tile_size)
                    y += yoffset
                    nodes_lut[(x, y)] = Node.create(x, y)
    
    @staticmethod
    def connect_horizontally(data, nodes_lut, position_to_index, tile_size, xoffset=0, yoffset=32):
        """Connect nodes horizontally (translated from connectHorizontally)."""
        node_symbols = ['+', 'H', 'o', 'D', 'P']
        path_symbols = ['.', 'H', 'o', 'D', 'P']
        
        for row in range(data.shape[0]):
            key = None
            for col in range(data.shape[1]):
                if data[row][col] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row, tile_size)
                    y += yoffset
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
    def connect_vertically(data, nodes_lut, position_to_index, tile_size, xoffset=0, yoffset=32):
        """Connect nodes vertically (translated from connectVertically)."""
        node_symbols = ['+', 'H', 'o', 'D', 'P']
        path_symbols = ['.', 'H', 'o', 'D', 'P']
        dataT = data.transpose()
        
        for col in range(dataT.shape[0]):
            key = None
            for row in range(dataT.shape[1]):
                if dataT[col][row] in node_symbols:
                    x, y = NodeGroup.construct_key(col + xoffset, row, tile_size)
                    y += yoffset
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

    @staticmethod
    def _connect_portal_pair_horizontal(nodes_lut, position_to_index, left_key, right_key):
        """Connect two portal nodes as a left-right (horizontal) pair."""
        left_pos = (float(left_key[0]), float(left_key[1]))
        right_pos = (float(right_key[0]), float(right_key[1]))
        left_idx = position_to_index.get(left_pos)
        right_idx = position_to_index.get(right_pos)
        if left_idx is None or right_idx is None:
            return
        left_node = nodes_lut[left_key]
        right_node = nodes_lut[right_key]
        left_node_neighbors = left_node.neighbor_indices.at[Action.LEFT].set(right_idx)
        right_node_neighbors = right_node.neighbor_indices.at[Action.RIGHT].set(left_idx)
        nodes_lut[left_key] = Node(position=left_node.position, neighbor_indices=left_node_neighbors)
        nodes_lut[right_key] = Node(position=right_node.position, neighbor_indices=right_node_neighbors)
        print(f"Connected Portal (horizontal): Node {left_idx} (Left) <-> Node {right_idx} (Right)")

    @staticmethod
    def _connect_portal_pair_vertical(nodes_lut, position_to_index, top_key, bottom_key):
        """Connect two portal nodes as a top-bottom (vertical) pair."""
        top_pos = (float(top_key[0]), float(top_key[1]))
        bottom_pos = (float(bottom_key[0]), float(bottom_key[1]))
        top_idx = position_to_index.get(top_pos)
        bottom_idx = position_to_index.get(bottom_pos)
        if top_idx is None or bottom_idx is None:
            return
        top_node = nodes_lut[top_key]
        bottom_node = nodes_lut[bottom_key]
        # Top node's UP neighbor -> bottom node; bottom node's DOWN neighbor -> top node
        top_node_neighbors = top_node.neighbor_indices.at[Action.UP].set(bottom_idx)
        bottom_node_neighbors = bottom_node.neighbor_indices.at[Action.DOWN].set(top_idx)
        nodes_lut[top_key] = Node(position=top_node.position, neighbor_indices=top_node_neighbors)
        nodes_lut[bottom_key] = Node(position=bottom_node.position, neighbor_indices=bottom_node_neighbors)
        print(f"Connected Portal (vertical): Node {top_idx} (Top) <-> Node {bottom_idx} (Bottom)")

    @staticmethod
    def connect_portals(data, nodes_lut, position_to_index, tile_size, xoffset=0, yoffset=0):
        """
        Connect portal nodes ('P') across the map. Supports left-right and up-down pairs.

        Pairing rules (same as 2-portal case, extended to many portals):
        - Two portals on the same row (same y) -> horizontal pair (LEFT <-> RIGHT).
        - Two portals on the same column (same x) -> vertical pair (UP <-> DOWN).

        Works for 2, 4, 6, ... portals as long as each pair aligns on a row or column.
        If there are 6 portals (e.g. 1 horizontal row + 2 vertical columns), all three
        pairs are connected. Previously only len==2 or len==4 was handled, so 6 portals
        skipped all connections and deadends broke.
        """
        # Find all 'P' nodes
        portal_keys = []
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] == 'P':
                    x, y = NodeGroup.construct_key(col + xoffset, row + yoffset, tile_size)
                    portal_keys.append((x, y))

        if not portal_keys:
            return
        if len(portal_keys) % 2 != 0:
            print(
                f"Warning: odd number of portal nodes ({len(portal_keys)}), "
                "skipping portal connections"
            )
            return

        # Group by same y (horizontal pairs) and same x (vertical pairs).
        # This subsumes the old len==2 and len==4 branches.
        by_x = {}
        by_y = {}
        for key in portal_keys:
            x, y = key
            by_x.setdefault(x, []).append(key)
            by_y.setdefault(y, []).append(key)
        for y, keys in by_y.items():
            if len(keys) == 2:
                keys_sorted = sorted(keys, key=lambda k: k[0])
                NodeGroup._connect_portal_pair_horizontal(
                    nodes_lut, position_to_index, keys_sorted[0], keys_sorted[1]
                )
            elif len(keys) > 2:
                print(
                    f"Warning: {len(keys)} portals on same row y={y}; "
                    "expected exactly 2 per row for horizontal pairing"
                )
        for x, keys in by_x.items():
            if len(keys) == 2:
                keys_sorted = sorted(keys, key=lambda k: k[1])
                NodeGroup._connect_portal_pair_vertical(
                    nodes_lut, position_to_index, keys_sorted[0], keys_sorted[1]
                )
            elif len(keys) > 2:
                print(
                    f"Warning: {len(keys)} portals on same column x={x}; "
                    "expected exactly 2 per column for vertical pairing"
                )
    
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
    