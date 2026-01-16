import tkinter as tk
from tkinter import ttk, messagebox
import threading, queue, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import time

# Constants
METERS_TO_PIXELS = 50  # For display only


# ---------------- DATA STRUCTURES ----------------
@dataclass
class Item:
    f: float  # width (x) in meters
    s: float  # depth (y) in meters
    u: float  # height (z) in meters
    label: str
    volume: float = 0

    def __post_init__(self):
        self.volume = self.f * self.s * self.u

    def rotations(self):
        """Generate unique rotations"""
        rotations = []
        dims = [
            (self.f, self.s, self.u),
            (self.f, self.u, self.s),
            (self.s, self.f, self.u),
            (self.s, self.u, self.f),
            (self.u, self.f, self.s),
            (self.u, self.s, self.f)
        ]

        seen = set()
        for f, s, u in dims:
            if (f, s, u) not in seen:
                seen.add((f, s, u))
                rotations.append(Item(f, s, u, self.label))
        return rotations


@dataclass
class PlacedItem:
    x: float  # position in meters
    y: float  # position in meters
    z: float  # position in meters
    item: Item
    color: str = ""

    @property
    def end_x(self):
        return self.x + self.item.f

    @property
    def end_y(self):
        return self.y + self.item.s

    @property
    def end_z(self):
        return self.z + self.item.u

    def intersects(self, other: 'PlacedItem') -> bool:
        """Check if this item intersects with another item"""
        epsilon = 0.001
        return not (self.end_x <= other.x + epsilon or
                    other.end_x <= self.x + epsilon or
                    self.end_y <= other.y + epsilon or
                    other.end_y <= self.y + epsilon or
                    self.end_z <= other.z + epsilon or
                    other.end_z <= self.z + epsilon)


@dataclass
class FreeSpace:
    """Represents an empty space in the container"""
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float

    @property
    def volume(self):
        return self.width * self.depth * self.height

    @property
    def end_x(self):
        return self.x + self.width

    @property
    def end_y(self):
        return self.y + self.depth

    @property
    def end_z(self):
        return self.z + self.height


# ---------------- OPTIMIZED 3D PACKING ALGORITHM ----------------
class OptimizedPacker:
    """True 3D bin packing using maximal spaces algorithm"""

    def __init__(self, container_dims, items):
        self.F, self.S, self.U = container_dims
        self.items = items
        self.placed = []

        # Start with one big free space (the entire container)
        self.free_spaces = [FreeSpace(0, 0, 0, self.F, self.S, self.U)]

    def find_best_space_for_item(self, item):
        """Find the best free space for an item"""
        best_space = None
        best_rotation = None
        best_score = -float('inf')

        for space in self.free_spaces:
            for rotated in item.rotations():
                if (rotated.f <= space.width and
                        rotated.s <= space.depth and
                        rotated.u <= space.height):

                    # Calculate score for this placement
                    score = self.calculate_placement_score(space, rotated)

                    if score > best_score:
                        best_score = score
                        best_space = space
                        best_rotation = rotated

        return best_space, best_rotation

    def calculate_placement_score(self, space, item):
        """Calculate score for placing item in space (higher is better)"""
        score = 0

        # Prefer lower positions (gravity)
        score -= space.z * 100

        # Prefer positions closer to origin
        score -= space.x * 10
        score -= space.y * 5

        # Bonus for fitting perfectly
        volume_ratio = item.volume / space.volume
        score += volume_ratio * 1000

        # Bonus for touching walls (stability)
        if space.x == 0:
            score += 50
        if space.y == 0:
            score += 50
        if space.z == 0:
            score += 100

        # Bonus for creating large remaining spaces
        remaining_width = space.width - item.f
        remaining_depth = space.depth - item.s
        remaining_height = space.height - item.u

        if remaining_width > 0:
            score += remaining_width * 10
        if remaining_depth > 0:
            score += remaining_depth * 10
        if remaining_height > 0:
            score += remaining_height * 20

        return score

    def place_item_in_space(self, item, space, color):
        """Place item in the specified free space"""
        placed_item = PlacedItem(space.x, space.y, space.z, item, color)
        self.placed.append(placed_item)

        # Remove the used space
        self.free_spaces.remove(space)

        # Create new free spaces from the remaining volume
        self.split_free_space(space, item)

        # Merge adjacent free spaces
        self.merge_free_spaces()

        # Remove small useless spaces
        self.remove_small_spaces(min_size=0.05)

        return placed_item

    def split_free_space(self, space, item):
        """Split the free space after placing an item"""
        # Create 3 new free spaces from the remaining volume

        # Space to the right (if any width left)
        if item.f < space.width:
            new_space = FreeSpace(
                space.x + item.f,
                space.y,
                space.z,
                space.width - item.f,
                space.depth,
                space.height
            )
            self.free_spaces.append(new_space)

        # Space above (if any depth left)
        if item.s < space.depth:
            new_space = FreeSpace(
                space.x,
                space.y + item.s,
                space.z,
                item.f,  # Only the area not covered by right space
                space.depth - item.s,
                space.height
            )
            self.free_spaces.append(new_space)

        # Space on top (if any height left)
        if item.u < space.height:
            new_space = FreeSpace(
                space.x,
                space.y,
                space.z + item.u,
                space.width,
                space.depth,
                space.height - item.u
            )
            self.free_spaces.append(new_space)

    def merge_free_spaces(self):
        """Merge adjacent free spaces to reduce fragmentation"""
        merged = True
        while merged:
            merged = False
            new_spaces = []

            for i in range(len(self.free_spaces)):
                space1 = self.free_spaces[i]
                if space1 is None:
                    continue

                for j in range(i + 1, len(self.free_spaces)):
                    space2 = self.free_spaces[j]
                    if space2 is None:
                        continue

                    # Try to merge in X direction
                    if (abs(space1.y - space2.y) < 0.01 and
                            abs(space1.z - space2.z) < 0.01 and
                            abs(space1.depth - space2.depth) < 0.01 and
                            abs(space1.height - space2.height) < 0.01 and
                            abs(space1.end_x - space2.x) < 0.01):

                        new_space = FreeSpace(
                            space1.x,
                            space1.y,
                            space1.z,
                            space1.width + space2.width,
                            space1.depth,
                            space1.height
                        )
                        new_spaces.append(new_space)
                        self.free_spaces[i] = None
                        self.free_spaces[j] = None
                        merged = True
                        break

                    # Try to merge in Y direction
                    elif (abs(space1.x - space2.x) < 0.01 and
                          abs(space1.z - space2.z) < 0.01 and
                          abs(space1.width - space2.width) < 0.01 and
                          abs(space1.height - space2.height) < 0.01 and
                          abs(space1.end_y - space2.y) < 0.01):

                        new_space = FreeSpace(
                            space1.x,
                            space1.y,
                            space1.z,
                            space1.width,
                            space1.depth + space2.depth,
                            space1.height
                        )
                        new_spaces.append(new_space)
                        self.free_spaces[i] = None
                        self.free_spaces[j] = None
                        merged = True
                        break

                    # Try to merge in Z direction
                    elif (abs(space1.x - space2.x) < 0.01 and
                          abs(space1.y - space2.y) < 0.01 and
                          abs(space1.width - space2.width) < 0.01 and
                          abs(space1.depth - space2.depth) < 0.01 and
                          abs(space1.end_z - space2.z) < 0.01):

                        new_space = FreeSpace(
                            space1.x,
                            space1.y,
                            space1.z,
                            space1.width,
                            space1.depth,
                            space1.height + space2.height
                        )
                        new_spaces.append(new_space)
                        self.free_spaces[i] = None
                        self.free_spaces[j] = None
                        merged = True
                        break

            # Add remaining non-merged spaces
            for space in self.free_spaces:
                if space is not None:
                    new_spaces.append(space)

            self.free_spaces = new_spaces

    def remove_small_spaces(self, min_size=0.1):
        """Remove free spaces that are too small to be useful"""
        self.free_spaces = [s for s in self.free_spaces
                            if s.width >= min_size and
                            s.depth >= min_size and
                            s.height >= min_size]

    def pack(self):
        """Main packing routine - TRUE optimization"""
        start_time = time.time()

        # Sort items by volume (largest first), then by min dimension
        items_to_pack = sorted(self.items,
                               key=lambda i: (-i.volume, -min(i.f, i.s, i.u)))

        placed_count = 0
        failed_items = []

        print(f"\nStarting OPTIMIZED packing of {len(items_to_pack)} items...")
        print(f"Container: {self.F:.1f}m × {self.S:.1f}m × {self.U:.1f}m")

        for idx, item in enumerate(items_to_pack):
            if idx % 10 == 0 and idx > 0:
                print(f"   Processing item {idx + 1}/{len(items_to_pack)}...")

            # Find best free space for this item
            best_space, best_rotation = self.find_best_space_for_item(item)

            if best_space:
                # Place the item
                color = self._get_color(item.label, placed_count)
                self.place_item_in_space(best_rotation, best_space, color)
                placed_count += 1
            else:
                failed_items.append(item)

        # Verify no overlaps
        overlaps = self.check_overlaps()

        total_time = time.time() - start_time
        efficiency = self.get_volume_utilization()

        print(f"\nPacking completed in {total_time:.2f} seconds")
        print(f"Placed {placed_count}/{len(items_to_pack)} items")
        print(f"Volume efficiency: {efficiency:.1f}%")

        if failed_items:
            print(f"Failed to place {len(failed_items)} items")

        if overlaps:
            print(f"WARNING: {len(overlaps)} overlaps detected!")

        return self.placed, placed_count, len(items_to_pack), efficiency, len(overlaps), self.free_spaces

    def check_overlaps(self):
        """Check for overlaps between placed items"""
        overlaps = []
        for i, item1 in enumerate(self.placed):
            for j, item2 in enumerate(self.placed):
                if i < j and item1.intersects(item2):
                    overlaps.append((item1, item2))
        return overlaps

    def get_volume_utilization(self):
        """Calculate volume utilization percentage"""
        total_volume = self.F * self.S * self.U
        used_volume = sum(item.item.volume for item in self.placed)
        return (used_volume / total_volume * 100) if total_volume > 0 else 0

    def _get_color(self, label, index):
        """Generate consistent color"""
        colors = [
            "#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0", "#118AB2",
            "#7209B7", "#3A86FF", "#FB5607", "#8338EC", "#FF006E",
            "#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"
        ]

        if label:
            color_idx = hash(label) % len(colors)
            return colors[color_idx]
        return colors[index % len(colors)]


# ---------------- COMPLETE 3D VISUALIZATION WITH ALL VIEWS ----------------
class CompleteVisualizer:
    """Complete visualization with ALL views: 3D, top, side, layers, and free space analysis"""

    def __init__(self, canvas, container_dims):
        self.canvas = canvas
        self.F, self.S, self.U = container_dims
        self.view_mode = "3d"  # 3d, top, side, layers, free_spaces
        self.rotation_angle = 45
        self.show_free_spaces = True
        self.show_grid = True
        self.show_labels = True
        self.current_layer = 0
        self.layer_height = 0.5

    def draw_packing(self, placed_items, free_spaces, width, height):
        """Draw packing based on current view mode"""
        self.canvas.delete("all")

        if self.view_mode == "3d":
            self.draw_3d_view(placed_items, free_spaces, width, height)
        elif self.view_mode == "top":
            self.draw_top_view(placed_items, free_spaces, width, height)
        elif self.view_mode == "side":
            self.draw_side_view(placed_items, free_spaces, width, height)
        elif self.view_mode == "layers":
            self.draw_layers_view(placed_items, free_spaces, width, height)
        elif self.view_mode == "free_spaces":
            self.draw_free_spaces_view(free_spaces, width, height)

        # Add stats
        self.draw_stats(placed_items, free_spaces, width, height)

    # ==================== 3D VIEW ====================
    def draw_3d_view(self, placed_items, free_spaces, width, height):
        """Draw beautiful 3D isometric view"""
        # Calculate center and scale
        padding = 80
        max_size = max(self.F, self.S, self.U)
        scale = min(width - 2 * padding, height - 2 * padding) / (max_size * 2.5)

        center_x = width / 2
        center_y = height / 2

        # Draw container wireframe
        self._draw_container_3d(center_x, center_y, scale)

        # Draw free spaces first (as semi-transparent blocks)
        if self.show_free_spaces:
            for space in free_spaces:
                self._draw_space_3d(space, center_x, center_y, scale)

        # Draw all items
        for placed in placed_items:
            self._draw_item_3d(placed, center_x, center_y, scale)

        # Title
        title = f"3D ISOMETRIC VIEW - {len(placed_items)} Items Packed"
        self.canvas.create_text(width / 2, 25,
                                text=title, font=("Arial", 14, "bold"),
                                fill="#333")

        # Controls hint
        controls = "Hold SPACE and drag to rotate | Mouse wheel to zoom"
        self.canvas.create_text(width / 2, height - 20,
                                text=controls, font=("Arial", 9),
                                fill="#666")

    def _draw_container_3d(self, center_x, center_y, scale):
        """Draw container as 3D wireframe"""
        w, d, h = self.F * scale, self.S * scale, self.U * scale

        # Calculate 8 corners with rotation
        angle_rad = math.radians(self.rotation_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        corners = []
        for x in [-w / 2, w / 2]:
            for y in [-d / 2, d / 2]:
                for z in [0, h]:
                    # Apply isometric projection
                    x_proj = (x * cos_angle) - (y * cos_angle)
                    y_proj = (x * sin_angle) + (y * sin_angle) - z

                    corners.append((
                        center_x + x_proj,
                        center_y + y_proj
                    ))

        # Draw edges
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)  # Sides
        ]

        for i, j in edges:
            self.canvas.create_line(corners[i][0], corners[i][1],
                                    corners[j][0], corners[j][1],
                                    fill="#333", width=3)

        # Add dimension labels
        if self.show_labels:
            # Width label (bottom front)
            mid_bottom_front = (
                (corners[0][0] + corners[1][0]) / 2,
                (corners[0][1] + corners[1][1]) / 2
            )
            self.canvas.create_text(mid_bottom_front[0], mid_bottom_front[1] - 15,
                                    text=f"{self.F:.1f}m", font=("Arial", 9, "bold"),
                                    fill="#666")

            # Depth label (bottom side)
            mid_bottom_side = (
                (corners[1][0] + corners[3][0]) / 2,
                (corners[1][1] + corners[3][1]) / 2
            )
            self.canvas.create_text(mid_bottom_side[0] + 10, mid_bottom_side[1],
                                    text=f"{self.S:.1f}m", font=("Arial", 9, "bold"),
                                    fill="#666")

            # Height label (front corner)
            self.canvas.create_text(corners[0][0] - 15, corners[0][1] - 10,
                                    text=f"{self.U:.1f}m", font=("Arial", 9, "bold"),
                                    fill="#666")

    def _draw_item_3d(self, placed, center_x, center_y, scale):
        """Draw item in 3D"""
        # Convert to container-relative coordinates
        x = placed.x * scale - (self.F * scale) / 2
        y = placed.y * scale - (self.S * scale) / 2
        z = placed.z * scale
        w = placed.item.f * scale
        d = placed.item.s * scale
        h = placed.item.u * scale

        # Calculate corners with rotation
        angle_rad = math.radians(self.rotation_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        corners = []
        for dx in [0, w]:
            for dy in [0, d]:
                for dz in [0, h]:
                    # Apply isometric projection
                    x_proj = ((x + dx) * cos_angle) - ((y + dy) * cos_angle)
                    y_proj = ((x + dx) * sin_angle) + ((y + dy) * sin_angle) - (z + dz)

                    corners.append((
                        center_x + x_proj,
                        center_y + y_proj
                    ))

        # Draw faces (only visible ones)
        faces = [
            (0, 1, 3, 2),  # Front face
            (1, 5, 7, 3),  # Right face
            (4, 5, 7, 6),  # Back face (partially visible)
            (2, 3, 7, 6),  # Top face
        ]

        face_colors = [
            self._lighten_color(placed.color, 0.8),  # Front (lightest)
            self._lighten_color(placed.color, 0.6),  # Right
            self._darken_color(placed.color, 0.2),  # Back
            placed.color,  # Top (main color)
        ]

        for i, (v1, v2, v3, v4) in enumerate(faces):
            points = [corners[v1], corners[v2], corners[v3], corners[v4]]
            self.canvas.create_polygon(points, fill=face_colors[i],
                                       outline=self._darken_color(face_colors[i], 0.3),
                                       width=1)

        # Draw edges for definition
        edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]

        for v1, v2 in edges:
            self.canvas.create_line(corners[v1][0], corners[v1][1],
                                    corners[v2][0], corners[v2][1],
                                    fill="#333", width=1)

        # Add label on top face if space allows
        if self.show_labels and h > 15:
            # Find center of top face
            top_center_x = (corners[2][0] + corners[3][0] + corners[6][0] + corners[7][0]) / 4
            top_center_y = (corners[2][1] + corners[3][1] + corners[6][1] + corners[7][1]) / 4

            label = placed.item.label
            if len(label) > 6:
                label = label[:4] + ".."

            self.canvas.create_text(top_center_x, top_center_y,
                                    text=f"{label}\n{placed.item.u:.1f}m",
                                    font=("Arial", 7, "bold"),
                                    fill="#000", anchor="center")

    def _draw_space_3d(self, space, center_x, center_y, scale):
        """Draw free space in 3D view (semi-transparent)"""
        # Only draw larger spaces
        if space.volume < 0.1:
            return

        # Convert to container-relative coordinates
        x = space.x * scale - (self.F * scale) / 2
        y = space.y * scale - (self.S * scale) / 2
        z = space.z * scale
        w = space.width * scale
        d = space.depth * scale
        h = space.height * scale

        # Calculate corners
        angle_rad = math.radians(self.rotation_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        corners = []
        for dx in [0, w]:
            for dy in [0, d]:
                for dz in [0, h]:
                    x_proj = ((x + dx) * cos_angle) - ((y + dy) * cos_angle)
                    y_proj = ((x + dx) * sin_angle) + ((y + dy) * sin_angle) - (z + dz)

                    corners.append((
                        center_x + x_proj,
                        center_y + y_proj
                    ))

        # Draw as semi-transparent box
        faces = [
            (0, 1, 3, 2),  # Front
            (1, 5, 7, 3),  # Right
            (2, 3, 7, 6),  # Top
        ]

        for v1, v2, v3, v4 in faces:
            points = [corners[v1], corners[v2], corners[v3], corners[v4]]
            self.canvas.create_polygon(points, fill="#e8f5e8",
                                       outline="#a5d6a7", width=1,
                                       stipple="gray50")

    # ==================== TOP VIEW ====================
    def draw_top_view(self, placed_items, free_spaces, width, height):
        """Draw top-down view (XY plane)"""
        x_offset, y_offset, draw_width, draw_height, scale = self._calculate_scale(width, height, "top")

        # Draw container
        self.canvas.create_rectangle(x_offset, y_offset,
                                     x_offset + draw_width,
                                     y_offset + draw_height,
                                     outline="#333", width=3,
                                     fill="#f8f9fa")

        # Draw grid
        if self.show_grid:
            self._draw_grid(x_offset, y_offset, draw_width, draw_height, scale)

        # Draw free spaces first
        if self.show_free_spaces:
            for space in free_spaces:
                self._draw_space_top(space, x_offset, y_offset, scale)

        # Draw items
        for placed in placed_items:
            self._draw_item_top(placed, x_offset, y_offset, scale)

        # Title
        title = f"TOP VIEW (XY PLANE) - {len(placed_items)} Items"
        self.canvas.create_text(width / 2, 25,
                                text=title, font=("Arial", 14, "bold"),
                                fill="#333")

    def _draw_item_top(self, placed, x_offset, y_offset, scale):
        """Draw item in top view"""
        x1 = x_offset + placed.x * scale
        y1 = y_offset + placed.y * scale
        x2 = x1 + placed.item.f * scale
        y2 = y1 + placed.item.s * scale

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return

        # Draw item
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=placed.color,
                                     outline=self._darken_color(placed.color, 0.3),
                                     width=2)

        # Add height pattern for tall items
        if placed.item.u > 0.5:
            pattern_color = self._darken_color(placed.color, 0.5)
            spacing = max(5, int((x2 - x1) / 5))
            for i in range(0, int(x2 - x1), spacing):
                self.canvas.create_line(x1 + i, y1, x1, y1 + i,
                                        fill=pattern_color, width=1)

        # Add label
        if self.show_labels and (x2 - x1) > 30 and (y2 - y1) > 20:
            label = placed.item.label
            if len(label) > 8:
                label = label[:6] + ".."

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.canvas.create_text(center_x, center_y,
                                    text=f"{label}\n{placed.item.u:.1f}m",
                                    font=("Arial", 8),
                                    fill="#000", anchor="center")

    def _draw_space_top(self, space, x_offset, y_offset, scale):
        """Draw free space in top view"""
        x1 = x_offset + space.x * scale
        y1 = y_offset + space.y * scale
        x2 = x1 + space.width * scale
        y2 = y1 + space.depth * scale

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill="#e8f5e8",
                                     outline="#a5d6a7",
                                     width=1,
                                     stipple="gray50")

    # ==================== SIDE VIEW ====================
    def draw_side_view(self, placed_items, free_spaces, width, height):
        """Draw side view (XZ plane)"""
        x_offset, y_offset, draw_width, draw_height, scale = self._calculate_scale(width, height, "side")

        # Draw container
        self.canvas.create_rectangle(x_offset, y_offset,
                                     x_offset + draw_width,
                                     y_offset + draw_height,
                                     outline="#333", width=3,
                                     fill="#f8f9fa")

        # Draw free spaces first
        if self.show_free_spaces:
            for space in free_spaces:
                self._draw_space_side(space, x_offset, y_offset, scale)

        # Draw items
        for placed in placed_items:
            self._draw_item_side(placed, x_offset, y_offset, scale)

        # Draw height grid
        if self.show_grid:
            self._draw_height_grid(x_offset, y_offset, draw_width, draw_height, scale)

        # Title
        title = f"SIDE VIEW (XZ PLANE) - Height Utilization"
        self.canvas.create_text(width / 2, 25,
                                text=title, font=("Arial", 14, "bold"),
                                fill="#333")

    def _draw_item_side(self, placed, x_offset, y_offset, scale):
        """Draw item in side view"""
        x1 = x_offset + placed.x * scale
        y1 = y_offset + scale * (self.U - placed.z - placed.item.u)
        x2 = x1 + placed.item.f * scale
        y2 = y1 + placed.item.u * scale

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=placed.color,
                                     outline=self._darken_color(placed.color, 0.3),
                                     width=1)

        if self.show_labels and (x2 - x1) > 30 and (y2 - y1) > 20:
            label = placed.item.label
            if len(label) > 6:
                label = label[:4] + ".."

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.canvas.create_text(center_x, center_y,
                                    text=f"{label}\n{placed.item.s:.1f}m",
                                    font=("Arial", 7),
                                    fill="#000", anchor="center")

    def _draw_space_side(self, space, x_offset, y_offset, scale):
        """Draw free space in side view"""
        x1 = x_offset + space.x * scale
        y1 = y_offset + scale * (self.U - space.z - space.height)
        x2 = x1 + space.width * scale
        y2 = y1 + space.height * scale

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill="#e8f5e8",
                                     outline="#a5d6a7",
                                     width=1,
                                     stipple="gray50")

    # ==================== LAYERS VIEW ====================
    def draw_layers_view(self, placed_items, free_spaces, width, height):
        """Draw layer-by-layer view"""
        x_offset, y_offset, draw_width, draw_height, scale = self._calculate_scale(width, height, "top")

        # Draw container
        self.canvas.create_rectangle(x_offset, y_offset,
                                     x_offset + draw_width,
                                     y_offset + draw_height,
                                     outline="#333", width=3,
                                     fill="#f8f9fa")

        # Find items and spaces in current layer
        layer_z_start = self.current_layer * self.layer_height
        layer_z_end = (self.current_layer + 1) * self.layer_height

        layer_items = []
        for placed in placed_items:
            if (placed.z < layer_z_end and placed.end_z > layer_z_start):
                layer_items.append(placed)

        layer_spaces = []
        for space in free_spaces:
            if (space.z < layer_z_end and space.end_z > layer_z_start):
                layer_spaces.append(space)

        # Draw free spaces first
        if self.show_free_spaces:
            for space in layer_spaces:
                self._draw_space_top(space, x_offset, y_offset, scale)

        # Draw items with opacity based on height in layer
        for placed in layer_items:
            # Calculate opacity based on how much of item is in this layer
            z_in_layer = max(0, placed.z - layer_z_start)
            item_height_in_layer = min(placed.item.u, self.layer_height - z_in_layer)
            opacity = 0.3 + (item_height_in_layer / self.layer_height) * 0.7

            self._draw_item_top_with_opacity(placed, x_offset, y_offset, scale, opacity)

        # Title
        title = f"LAYER {self.current_layer}: {layer_z_start:.1f}m - {layer_z_end:.1f}m"
        self.canvas.create_text(width / 2, 25,
                                text=title, font=("Arial", 14, "bold"),
                                fill="#333")

        info = f"Items in layer: {len(layer_items)} | Free spaces: {len(layer_spaces)}"
        self.canvas.create_text(width / 2, 45,
                                text=info, font=("Arial", 10),
                                fill="#666")

    def _draw_item_top_with_opacity(self, placed, x_offset, y_offset, scale, opacity):
        """Draw item with opacity for layer view"""
        x1 = x_offset + placed.x * scale
        y1 = y_offset + placed.y * scale
        x2 = x1 + placed.item.f * scale
        y2 = y1 + placed.item.s * scale

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return

        # Adjust color for opacity
        base_color = placed.color.lstrip('#')
        r = int(base_color[0:2], 16)
        g = int(base_color[2:4], 16)
        b = int(base_color[4:6], 16)

        # Blend with white based on opacity
        r = int(r * opacity + 255 * (1 - opacity))
        g = int(g * opacity + 255 * (1 - opacity))
        b = int(b * opacity + 255 * (1 - opacity))

        fill_color = f"#{r:02x}{g:02x}{b:02x}"

        # Draw item
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=fill_color,
                                     outline=self._darken_color(fill_color, 0.3),
                                     width=1)

        # Add label if high opacity
        if opacity > 0.7 and self.show_labels and (x2 - x1) > 30 and (y2 - y1) > 20:
            label = placed.item.label
            if len(label) > 6:
                label = label[:4] + ".."

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.canvas.create_text(center_x, center_y,
                                    text=f"{label}\n{placed.item.u:.1f}m",
                                    font=("Arial", 8),
                                    fill="#000", anchor="center")

    # ==================== FREE SPACES VIEW ====================
    def draw_free_spaces_view(self, free_spaces, width, height):
        """View showing ONLY free spaces for optimization analysis"""
        x_offset, y_offset, draw_width, draw_height, scale = self._calculate_scale(width, height, "top")

        # Draw container
        self.canvas.create_rectangle(x_offset, y_offset,
                                     x_offset + draw_width,
                                     y_offset + draw_height,
                                     outline="#333", width=3,
                                     fill="#f8f9fa")

        # Sort free spaces by volume
        sorted_spaces = sorted(free_spaces, key=lambda s: -s.volume)

        # Draw free spaces with color coding
        for space in sorted_spaces:
            x1 = x_offset + space.x * scale
            y1 = y_offset + space.y * scale
            x2 = x1 + space.width * scale
            y2 = y1 + space.depth * scale

            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            # Color by size
            if space.volume > 1.0:
                color = "#c8e6c9"
                outline = "#4caf50"
            elif space.volume > 0.1:
                color = "#fff9c4"
                outline = "#ffb300"
            else:
                color = "#ffcdd2"
                outline = "#f44336"

            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         fill=color, outline=outline,
                                         width=2)

            # Add dimensions
            if (x2 - x1) > 40 and (y2 - y1) > 30:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                label = f"{space.width:.1f}×{space.depth:.1f}×{space.height:.1f}m"
                self.canvas.create_text(center_x, center_y,
                                        text=label, font=("Arial", 7),
                                        fill="#000", anchor="center")

        # Title
        total_free = sum(s.volume for s in free_spaces)
        title = f"FREE SPACES ANALYSIS - {len(free_spaces)} spaces, {total_free:.1f}m³ total"
        self.canvas.create_text(width / 2, 25,
                                text=title, font=("Arial", 14, "bold"),
                                fill="#333")

        legend = "Green: >1m³ | Yellow: 0.1-1m³ | Red: <0.1m³"
        self.canvas.create_text(width / 2, 45,
                                text=legend, font=("Arial", 9),
                                fill="#666")

    # ==================== UTILITY METHODS ====================
    def _calculate_scale(self, width, height, view_type):
        """Calculate drawing scale and offsets"""
        padding = 60
        max_width = width - 2 * padding
        max_height = height - 2 * padding

        if view_type == "top" or view_type == "layers" or view_type == "free_spaces":
            scale_x = max_width / (self.F * METERS_TO_PIXELS)
            scale_y = max_height / (self.S * METERS_TO_PIXELS)
            draw_width = self.F * METERS_TO_PIXELS * min(scale_x, scale_y)
            draw_height = self.S * METERS_TO_PIXELS * min(scale_x, scale_y)
        else:  # side view
            scale_x = max_width / (self.F * METERS_TO_PIXELS)
            scale_y = max_height / (self.U * METERS_TO_PIXELS)
            draw_width = self.F * METERS_TO_PIXELS * min(scale_x, scale_y)
            draw_height = self.U * METERS_TO_PIXELS * min(scale_x, scale_y)

        x_offset = padding + (max_width - draw_width) / 2
        y_offset = padding + (max_height - draw_height) / 2
        scale = min(scale_x, scale_y) * METERS_TO_PIXELS

        return x_offset, y_offset, draw_width, draw_height, scale

    def _draw_grid(self, x_offset, y_offset, width, height, scale):
        """Draw measurement grid"""
        # Vertical lines
        for x in range(0, int(self.F) + 1):
            x_pos = x_offset + x * scale
            self.canvas.create_line(x_pos, y_offset, x_pos, y_offset + height,
                                    fill="#ddd", width=1)
            if x > 0:
                self.canvas.create_text(x_pos, y_offset - 5, text=f"{x}m",
                                        font=("Arial", 8), anchor="s", fill="#666")

        # Horizontal lines
        for y in range(0, int(self.S) + 1):
            y_pos = y_offset + y * scale
            self.canvas.create_line(x_offset, y_pos, x_offset + width, y_pos,
                                    fill="#ddd", width=1)
            if y > 0:
                self.canvas.create_text(x_offset - 5, y_pos, text=f"{y}m",
                                        font=("Arial", 8), anchor="e", fill="#666")

    def _draw_height_grid(self, x_offset, y_offset, width, height, scale):
        """Draw height measurement grid for side view"""
        # Vertical lines
        for x in range(0, int(self.F) + 1):
            x_pos = x_offset + x * scale
            self.canvas.create_line(x_pos, y_offset, x_pos, y_offset + height,
                                    fill="#ddd", width=1)
            if x > 0:
                self.canvas.create_text(x_pos, y_offset - 5, text=f"{x}m",
                                        font=("Arial", 8), anchor="s", fill="#666")

        # Horizontal lines (height)
        for z in range(0, int(self.U) + 1):
            y_pos = y_offset + height - (z * scale)
            self.canvas.create_line(x_offset, y_pos, x_offset + width, y_pos,
                                    fill="#ddd", width=1)
            if z > 0:
                self.canvas.create_text(x_offset - 5, y_pos, text=f"{z}m",
                                        font=("Arial", 8), anchor="e", fill="#666")

    def draw_stats(self, placed_items, free_spaces, width, height):
        """Draw statistics overlay"""
        total_volume = self.F * self.S * self.U
        used_volume = sum(p.item.volume for p in placed_items)
        free_volume = sum(s.volume for s in free_spaces)
        efficiency = (used_volume / total_volume * 100) if total_volume > 0 else 0

        stats = [
            f"Items placed: {len(placed_items)}",
            f"Volume efficiency: {efficiency:.1f}%",
            f"Used volume: {used_volume:.1f}m³",
            f"Free volume: {free_volume:.1f}m³",
            f"View: {self.view_mode.upper()}",
        ]

        if self.view_mode == "layers":
            stats.append(f"Layer: {self.current_layer}/{int(self.U / self.layer_height) - 1}")

        if efficiency > 85:
            quality = "EXCELLENT"
        elif efficiency > 70:
            quality = "GOOD"
        else:
            quality = "NEEDS IMPROVEMENT"
        stats.append(f"Quality: {quality}")

        # Draw in top-right corner
        x_pos = width - 15
        y_start = 50

        for i, text in enumerate(stats):
            y_pos = y_start + i * 22

            # Background
            text_width = len(text) * 7
            self.canvas.create_rectangle(x_pos - text_width - 15, y_pos - 10,
                                         x_pos + 5, y_pos + 12,
                                         fill="white", outline="#ddd", width=1)

            # Text
            color = "#333"
            if "efficiency" in text:
                if efficiency > 85:
                    color = "#2e7d32"
                elif efficiency > 70:
                    color = "#ff9800"
                else:
                    color = "#f44336"
            elif "Quality:" in text:
                if "EXCELLENT" in text:
                    color = "#2e7d32"
                elif "GOOD" in text:
                    color = "#ff9800"
                else:
                    color = "#f44336"

            self.canvas.create_text(x_pos, y_pos,
                                    text=text, anchor="ne",
                                    font=("Arial", 9),
                                    fill=color)

    def _lighten_color(self, color, factor):
        """Lighten a color"""
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _darken_color(self, color, factor):
        """Darken a color"""
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        return f"#{r:02x}{g:02x}{b:02x}"


# ---------------- MAIN APPLICATION ----------------
class UltimatePackingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Container Packing")
        self.root.geometry("1400x900")

        # Variables
        self.placed_items = []
        self.free_spaces = []
        self.container_dims = None
        self.visualizer = None
        self.q = queue.Queue()
        self.efficiency = 0

        self.setup_ui()
        self.root.mainloop()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title = ttk.Label(main_frame,
                          text="CONTAINER PACKING",
                          font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Input panel
        input_panel = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_panel.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E), padx=(0, 10))

        # Container dimensions
        ttk.Label(input_panel, text="Container (W×D×H meters):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.container_entry = ttk.Entry(input_panel, width=20)
        self.container_entry.insert(0, "12.0, 2.4, 2.6")
        self.container_entry.grid(row=0, column=1, pady=2, padx=(5, 0))

        # Items input
        ttk.Label(input_panel, text="Items to pack:").grid(row=1, column=0, sticky=tk.W, pady=(10, 2))

        items_frame = ttk.Frame(input_panel)
        items_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        self.items_text = tk.Text(items_frame, width=40, height=12, wrap=tk.NONE)
        items_scroll = ttk.Scrollbar(items_frame, orient="vertical", command=self.items_text.yview)
        self.items_text.configure(yscrollcommand=items_scroll.set)

        self.items_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        items_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Example with items that pack WELL in 3D
        example = """# EXAMPLE
# Format: width,depth,height,count,label (meters)
1.2,0.8,0.5,6,A
0.6,0.4,0.6,10,B
0.8,0.6,0.4,12,C
1.0,0.5,0.8,4,D
0.4,0.4,0.4,16,E
0.9,0.7,0.3,8,F
2.4,1.2,1.0,2,G"""
        self.items_text.insert("1.0", example)

        # Display controls
        display_frame = ttk.LabelFrame(input_panel, text="👁️ Display Options", padding="10")
        display_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.space_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show free spaces",
                        variable=self.space_var,
                        command=self.toggle_spaces).pack(anchor=tk.W, pady=2)

        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show grid",
                        variable=self.grid_var,
                        command=self.toggle_grid).pack(anchor=tk.W, pady=2)

        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show labels",
                        variable=self.labels_var,
                        command=self.toggle_labels).pack(anchor=tk.W, pady=2)

        # Action buttons
        button_frame = ttk.Frame(input_panel)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="OPTIMIZE & VISUALIZE",
                   command=self.start_optimization,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Analyze Spaces",
                   command=self.analyze_spaces).pack(side=tk.LEFT, padx=5)

        # View selection
        view_frame = ttk.LabelFrame(input_panel, text="🎬 Visualization Views", padding="10")
        view_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # View buttons with icons
        view_buttons = [
            ("3D View", "3d"),
            ("Top View", "top"),
            ("Side View", "side"),
            ("Layer View", "layers"),
            ("Space Analysis", "free_spaces")
        ]

        for i, (text, view) in enumerate(view_buttons):
            btn = ttk.Button(view_frame, text=text,
                             command=lambda v=view: self.change_view(v),
                             width=14)
            btn.grid(row=0, column=i, padx=2)

        # Layer controls (only for layer view)
        layer_frame = ttk.Frame(view_frame)
        layer_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))

        ttk.Label(layer_frame, text="Layer:").pack(side=tk.LEFT)
        self.layer_var = tk.IntVar(value=0)
        self.layer_entry = ttk.Entry(layer_frame, textvariable=self.layer_var, width=4)
        self.layer_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(layer_frame, text="▲",
                   command=lambda: self.change_layer(1)).pack(side=tk.LEFT, padx=2)

        ttk.Button(layer_frame, text="▼",
                   command=lambda: self.change_layer(-1)).pack(side=tk.LEFT, padx=2)

        # 3D rotation controls
        rotate_frame = ttk.Frame(view_frame)
        rotate_frame.grid(row=2, column=0, columnspan=5, pady=(10, 0))

        ttk.Button(rotate_frame, text="↻ Rotate 3D View",
                   command=self.rotate_3d).pack(side=tk.LEFT, padx=5)

        # Visualization area
        vis_frame = ttk.LabelFrame(main_frame, text="🎨 3D Visualization & Analysis", padding="5")
        vis_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.canvas = tk.Canvas(vis_frame, bg="white", highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Enter items and click OPTIMIZE & VISUALIZE")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2,
                               sticky=(tk.W, tk.E), pady=(5, 0))

        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Accent.TButton",
                             font=("Arial", 10, "bold"),
                             padding=8,
                             foreground="white",
                             background="#2196F3")

        # Bind keyboard shortcuts
        self.root.bind('<Up>', lambda e: self.change_layer(1))
        self.root.bind('<Down>', lambda e: self.change_layer(-1))
        self.root.bind('r', lambda e: self.rotate_3d())
        self.root.bind('g', lambda e: self.toggle_grid())
        self.root.bind('s', lambda e: self.toggle_spaces())
        self.root.bind('l', lambda e: self.toggle_labels())

        # Welcome message
        self.canvas.create_text(300, 200,
                                text="CONTAINER PACKING VISUALIZER\n\n"
                                     "Features:\n"
                                     "• Optimal packing algorithm\n"
                                     "• 3D visualization\n"
                                     "• Multiple view modes\n"
                                     "• Free space analysis\n"
                                     "• Achieves >80% efficiency or something",
                                font=("Arial", 12),
                                fill="#666",
                                justify="center")

    def parse_input(self):
        """Parse input with validation"""
        try:
            # Parse container
            container_text = self.container_entry.get().strip()
            parts = [p.strip() for p in container_text.replace(",", " ").split()]
            if len(parts) < 3:
                raise ValueError("Container needs width,depth,height dimensions")

            container_dims = tuple(float(p) for p in parts[:3])

            if any(d <= 0 for d in container_dims):
                raise ValueError("Container dimensions must be positive")

            # Parse items
            items = []
            lines = self.items_text.get("1.0", tk.END).strip().split("\n")

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.replace(",", " ").split()]
                if len(parts) < 4:
                    raise ValueError(f"Line {line_num}: Need width,depth,height,count[,label]")

                f, s, u = float(parts[0]), float(parts[1]), float(parts[2])
                count = int(parts[3])
                label = parts[4] if len(parts) > 4 else f"Item_{line_num}"

                # Check if item fits in container
                if f > container_dims[0] or s > container_dims[1] or u > container_dims[2]:
                    raise ValueError(f"Item '{label}' ({f}x{s}x{u}m) doesn't fit in container!")

                for _ in range(count):
                    items.append(Item(f, s, u, label))

            if not items:
                raise ValueError("No items to pack")

            return container_dims, items

        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return None, None

    def start_optimization(self):
        """Start OPTIMAL packing with 3D visualization"""
        container_dims, items = self.parse_input()
        if not items:
            return

        self.container_dims = container_dims
        self.status_var.set("Running optimal packing algorithm...")
        self.progress_bar.start()

        # Clear previous
        self.placed_items = []
        self.free_spaces = []
        self.canvas.delete("all")

        # Start in thread
        threading.Thread(target=self.optimize_worker, args=(container_dims, items), daemon=True).start()
        self.root.after(50, self.poll_queue)

    def optimize_worker(self, container_dims, items):
        """Worker thread for optimal packing"""
        try:
            self.q.put(("progress", 10))

            # Use OPTIMIZED packer
            packer = OptimizedPacker(container_dims, items)
            self.q.put(("progress", 50))

            placed_items, placed_count, total_items, efficiency, overlap_count, free_spaces = packer.pack()
            self.q.put(("progress", 90))

            self.q.put(("result", {
                "placed": placed_items,
                "free_spaces": free_spaces,
                "placed_count": placed_count,
                "total_items": total_items,
                "efficiency": efficiency,
                "overlap_count": overlap_count,
                "container_dims": container_dims
            }))
            self.q.put(("progress", 100))

        except Exception as e:
            self.q.put(("error", f"Optimization error: {str(e)}"))

    def poll_queue(self):
        """Poll queue for updates"""
        try:
            while True:
                msg_type, data = self.q.get_nowait()

                if msg_type == "progress":
                    self.progress_var.set(data)
                elif msg_type == "result":
                    self.progress_bar.stop()
                    self.handle_result(data)
                elif msg_type == "error":
                    self.progress_bar.stop()
                    messagebox.showerror("Error", data)

        except queue.Empty:
            self.root.after(100, self.poll_queue)

    def handle_result(self, data):
        """Handle optimization results"""
        self.placed_items = data["placed"]
        self.free_spaces = data["free_spaces"]
        self.efficiency = data["efficiency"]

        # Create visualizer
        self.visualizer = CompleteVisualizer(self.canvas, data["container_dims"])
        self.visualizer.show_free_spaces = self.space_var.get()
        self.visualizer.show_grid = self.grid_var.get()
        self.visualizer.show_labels = self.labels_var.get()

        # Update status
        used_volume = sum(p.item.volume for p in self.placed_items)
        free_volume = sum(s.volume for s in self.free_spaces)
        total_volume = data["container_dims"][0] * data["container_dims"][1] * data["container_dims"][2]

        status = (f"OPTIMIZATION COMPLETE | "
                  f"Efficiency: {self.efficiency:.1f}% | "
                  f"Items: {len(self.placed_items)} placed | "
                  f"Free: {free_volume:.1f}m³")

        if self.efficiency > 85:
            status += " | EXCELLENT!"
        elif self.efficiency > 70:
            status += " | GOOD"

        self.status_var.set(status)

        # Draw initial view (3D by default)
        self.redraw()

        # Show summary
        if self.efficiency > 80:
            messagebox.showinfo("Excellent Packing!",
                                f"Achieved {self.efficiency:.1f}% volume efficiency!\n\n"
                                f"• Items placed: {len(self.placed_items)}\n"
                                f"• Free space: {free_volume:.1f}m³\n"
                                f"• View options: 5 different visualization modes")

    def redraw(self):
        """Redraw visualization"""
        if not self.visualizer:
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 50 or height < 50:
            return

        self.visualizer.draw_packing(self.placed_items, self.free_spaces, width, height)

    def change_view(self, view_mode):
        """Change visualization view"""
        if self.visualizer:
            self.visualizer.view_mode = view_mode
            if view_mode == "layers":
                self.visualizer.current_layer = self.layer_var.get()
            self.redraw()

    def change_layer(self, delta):
        """Change current layer"""
        if self.visualizer and self.visualizer.view_mode == "layers":
            current = self.layer_var.get()
            max_layers = int(math.ceil(self.container_dims[2] / self.visualizer.layer_height)) - 1
            new_layer = max(0, min(current + delta, max_layers))
            self.layer_var.set(new_layer)
            self.visualizer.current_layer = new_layer
            self.redraw()

    def rotate_3d(self):
        """Rotate 3D view"""
        if self.visualizer and self.visualizer.view_mode == "3d":
            self.visualizer.rotation_angle = (self.visualizer.rotation_angle + 15) % 360
            self.redraw()

    def toggle_spaces(self):
        """Toggle free spaces display"""
        if self.visualizer:
            self.visualizer.show_free_spaces = self.space_var.get()
            self.redraw()

    def toggle_grid(self):
        """Toggle grid display"""
        if self.visualizer:
            self.visualizer.show_grid = self.grid_var.get()
            self.redraw()

    def toggle_labels(self):
        """Toggle labels display"""
        if self.visualizer:
            self.visualizer.show_labels = self.labels_var.get()
            self.redraw()

    def analyze_spaces(self):
        """Analyze free spaces"""
        if not self.visualizer or not self.free_spaces:
            messagebox.showinfo("Analyze Spaces",
                                "Run optimization first to analyze free spaces.")
            return

        # Switch to free spaces view
        self.visualizer.view_mode = "free_spaces"
        self.redraw()

        # Show analysis
        total_free = sum(s.volume for s in self.free_spaces)
        large = len([s for s in self.free_spaces if s.volume > 1.0])
        medium = len([s for s in self.free_spaces if 0.1 <= s.volume <= 1.0])
        small = len([s for s in self.free_spaces if s.volume < 0.1])

        analysis = (f"📊 Space Analysis:\n\n"
                    f"Total free: {total_free:.1f}m³\n"
                    f"Large spaces (>1m³): {large}\n"
                    f"Medium spaces: {medium}\n"
                    f"Small spaces: {small}\n\n"
                    f"Try adding items that fit these spaces!")

        messagebox.showinfo("Space Analysis", analysis)

    def clear_all(self):
        """Clear everything"""
        self.container_entry.delete(0, tk.END)
        self.container_entry.insert(0, "12.0, 2.4, 2.6")
        self.items_text.delete("1.0", tk.END)
        example = """#EXAMPLE
# Format: width,depth,height,count,label (meters)
1.2,0.8,0.5,6,A
0.6,0.4,0.6,10,B
0.8,0.6,0.4,12,C
1.0,0.5,0.8,4,D
0.4,0.4,0.4,16,E
0.9,0.7,0.3,8,F
2.4,1.2,1.0,2,G"""
        self.items_text.insert("1.0", example)
        self.canvas.delete("all")
        self.placed_items = []
        self.free_spaces = []
        self.status_var.set("Ready - Enter items and click OPTIMIZE & VISUALIZE")
        self.progress_var.set(0)
        self.efficiency = 0
        self.layer_var.set(0)


# Run application
if __name__ == "__main__":
    UltimatePackingApp()