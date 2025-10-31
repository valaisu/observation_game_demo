#!/usr/bin/env python3
"""
The prediction game - Main Entry Point

A farming game where players learn to predict chaotic weather patterns
and plan their actions accordingly to maximize food production.
"""

import pygame
from PIL import Image, ImageOps
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import random
import math
import os
import json
from datetime import datetime

# Game constants
GAME_WIDTH = 800
CONTROL_PANEL_WIDTH = 470
LEFT_PANEL_WIDTH = 470  # Same width as control panel
TOP_BAR_HEIGHT = 30
GAME_X_OFFSET = LEFT_PANEL_WIDTH  # Game world starts after left panel
WINDOW_WIDTH = LEFT_PANEL_WIDTH + GAME_WIDTH + CONTROL_PANEL_WIDTH
WINDOW_HEIGHT = 800 + TOP_BAR_HEIGHT
FPS = 4

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
LIGHT_GRAY = (200, 200, 200)
SKY_BLUE = (135, 206, 235)
GROUND_GREEN = (34, 139, 34)
DARK_BLUE = (0, 100, 200)  # For text highlights on light green background

class Tree:
    def __init__(self,
                 back_leaves_paths: list[str],
                 back_leaves_count: int,
                 front_leaves_paths: list[str],
                 front_leaves_count: int,
                 trunk_path: list[str],
                 location: tuple[int, int] = (100, 100),
                 leaf_bias: tuple[int, int] = (-1, 0),
                 img_scaling_factor: float = 2):
        self.back_leaves_paths = back_leaves_paths
        self.back_leaves_count = back_leaves_count
        self.front_leaves_paths = front_leaves_paths
        self.front_leaves_count = front_leaves_count
        self.trunk = []
        self.front_leaves = []
        self.front_leaves_phases = np.array([random.random()*3 for _ in range(front_leaves_count)])
        self.front_leaves_wind_multipliers = []
        self.back_leaves = []
        self.back_leaves_phases = np.array([random.random()*3 for _ in range(back_leaves_count)])
        self.back_leaves_wind_multipliers = []

        self.populate(self.trunk, trunk_path, img_scaling_factor)
        self.populate(self.back_leaves, self.back_leaves_paths, img_scaling_factor)
        self.populate(self.front_leaves, self.front_leaves_paths, img_scaling_factor)

        self.set_heights()

        self.location = location
        self.x = self.location[0]
        self.y = self.location[1]

        self.leaf_bias = leaf_bias
        self.leaf_bias_x = leaf_bias[0]
        self.leaf_bias_y = leaf_bias[1]

    def populate(self, dest: list[pygame.surface.Surface], paths: list[str], scaling_factor: int):
        for file in paths:
            img = Image.open(file).convert('RGBA')
            new_size = (int(img.size[0]*scaling_factor), int(img.size[1]*scaling_factor))
            img = img.resize(new_size, Image.NEAREST)
            pygame_img = pygame.image.fromstring(
                img.tobytes(), img.size, img.mode
            )
            dest.append(pygame_img)

    def set_heights(self):
        for img in self.front_leaves:
            self.front_leaves_wind_multipliers.append(self.find_relative_height(img))
        for img in self.back_leaves:
            self.back_leaves_wind_multipliers.append(self.find_relative_height(img))

    def find_relative_height(self, pygame_img: pygame.surface.Surface):
        arr = pygame.surfarray.array3d(pygame_img)
        tot_height = len(arr)
        for i, row in enumerate(arr):
            for j, element in enumerate(row):
                if not all([k==0 for k in element]):
                    return (tot_height-j)/tot_height
        return 1.0

    def update_phase(self, amplitude: float = math.pi/180):
        self.back_leaves_phases += np.full(self.back_leaves_count, amplitude)
        self.front_leaves_phases += np.full(self.front_leaves_count, amplitude)

    def render(self, screen: pygame.surface.Surface, wind_speed: float = 10):
        wind_multiplier = wind_speed / 5
        for i, l in enumerate(self.back_leaves):
            new_x = self.x + self.back_leaves_wind_multipliers[i] * (math.sin(self.back_leaves_phases[i])*wind_multiplier+self.leaf_bias_x)
            new_y = self.y+self.leaf_bias_y
            screen.blit(l, (new_x, new_y))
        screen.blit(self.trunk[0], (self.location))
        for i, l in enumerate(self.front_leaves):
            new_x = self.x + self.front_leaves_wind_multipliers[i] * (math.sin(self.front_leaves_phases[i])*wind_multiplier+self.leaf_bias_x)
            new_y = self.y+self.leaf_bias_y
            screen.blit(l, (new_x, new_y))

class Doorway:
    def __init__(self,
                 window_w: int,
                 window_h: int,
                 doorway_w: int,
                 doorway_h: int,
                 plank_paths: list[str],
                 plank_dim: tuple[int, int],
                 scale_factor: int,
                 x_offset: int = 0):

        n_planks_y = int(window_h/(plank_dim[1] * scale_factor) + 0.999)
        n_planks_x_per_side = int(int((window_w-doorway_w)/2) / (plank_dim[0] * scale_factor) + 0.999)
        lb = int((window_w-doorway_w)/2)
        left_offset = lb - n_planks_x_per_side * plank_dim[0] * scale_factor
        right_offset = lb + doorway_w
        n_horizontal_planks_x = int(doorway_w / (plank_dim[1] * scale_factor) + 0.999)
        n_horizontal_planks_y = int((window_h-doorway_h) / (plank_dim[0]*scale_factor) + 0.999)
        horizontal_planks_offset_y = (window_h-doorway_h) - n_horizontal_planks_y * (plank_dim[0]*scale_factor)

        self.vertical_plank_locs_left = []
        self.vertical_plank_locs_right = []
        self.horizontal_plank_locs = []

        for w in range(n_planks_x_per_side):
            for h in range(n_planks_y):
                offset_term = (w*scale_factor*873%59)*(w%2)
                self.vertical_plank_locs_left.append((x_offset + left_offset + w*plank_dim[0]*scale_factor, h*plank_dim[1]*scale_factor - offset_term))
        for w in range(n_planks_x_per_side):
            for h in range(n_planks_y):
                offset_term = (w*scale_factor*493%59)*(w%2)
                self.vertical_plank_locs_right.append((x_offset + right_offset + w*plank_dim[0]*scale_factor, h*plank_dim[1]*scale_factor - offset_term))

        for w in range(n_horizontal_planks_x):
            for h in range(n_horizontal_planks_y):
                offset_term = (w*scale_factor*647%59)*(h%2)
                self.horizontal_plank_locs.append((x_offset + lb + w*plank_dim[1]*scale_factor-offset_term, horizontal_planks_offset_y + h*plank_dim[0]*scale_factor))

        self.planks = []
        self.planks_rotated = []
        self.populate(self.planks, plank_paths, scale_factor)
        self.populate(self.planks_rotated, plank_paths, scale_factor, rotate_angle=90)

    def populate(self, dest: list[pygame.surface.Surface], paths: list[str], scaling_factor: int, rotate_angle: int = 0, flips: bool = True, mirrors: bool = True):
        for file in paths:
            img = Image.open(file).convert('RGBA')
            img = img.rotate(rotate_angle, expand=True)
            new_size = (img.size[0]*scaling_factor, img.size[1]*scaling_factor)
            img = img.resize(new_size, Image.NEAREST)
            pygame_img = pygame.image.fromstring(
                img.tobytes(), img.size, img.mode
            )
            dest.append(pygame_img)
            if flips:
                img = img.rotate(180, expand=True)
                pygame_img = pygame.image.fromstring(
                    img.tobytes(), img.size, img.mode
                )
                dest.append(pygame_img)
            if mirrors:
                img = ImageOps.flip(img)
                pygame_img = pygame.image.fromstring(
                    img.tobytes(), img.size, img.mode
                )
                dest.append(pygame_img)

    def render(self, screen: pygame.surface.Surface):
        for plank_pos in self.horizontal_plank_locs:
            screen.blit(self.planks_rotated[0], plank_pos)
        for plank_pos in self.vertical_plank_locs_left+self.vertical_plank_locs_right:
            screen.blit(self.planks[0], plank_pos)

class Clouds:
    def __init__(self, init_size: int = 7, init_n: int = 4, scaling_factor: int = 2):
        sizes = [3, 5, 8, 13, 17, 25, 35, 51, 70]
        irregularities = [0.60, 0.65, 0.70, 0.75, 0.80]
        self.all_clouds_paths = [[f'data/clouds_raw/cloud_s{s}_i{i}.png' for i in irregularities] for s in sizes]

        # Create multiple cloud image sets at different scales for depth layers
        # More aggressive scaling: Layer 0 (closest): 1.0, Layer 1: 0.6, Layer 2: 0.35, Layer 3: 0.2, Layer 4: 0.1, Layer 5: 0.05
        self.layer_scales = [1.0, 0.6, 0.35, 0.2, 0.1, 0.05]
        self.all_clouds_imgs_by_layer = []

        for layer_scale in self.layer_scales:
            layer_clouds = []
            effective_scale = max(1, int(scaling_factor * layer_scale))
            self.populate(layer_clouds, self.all_clouds_paths, effective_scale)
            self.all_clouds_imgs_by_layer.append(layer_clouds)

        # Store clouds per layer: each layer has its own list of clouds and locations
        self.cloud_layers = []  # List of (clouds, locations) tuples
        for _ in self.layer_scales:
            self.cloud_layers.append(([], []))

        self.add_clouds(init_n, init_size)

    def add_clouds(self, number, size):
        """Add clouds across all layers maintaining density"""
        horizon_y = int(0.325 * 800)  # Horizon is 260 pixels down from top bar

        for layer_idx, layer_scale in enumerate(self.layer_scales):
            # More clouds for smaller scales (farther away) to maintain visual density
            # Use moderate inverse scaling to get more clouds near horizon without overdoing it
            if layer_scale >= 0.6:
                # Close layers: normal scaling
                layer_cloud_count = int(number * layer_scale)
            else:
                # Far layers: moderate inverse scaling
                # Scale 0.35 -> ~2x, Scale 0.2 -> ~3x, Scale 0.1 -> ~5x, Scale 0.05 -> ~7x
                layer_cloud_count = int(number / (layer_scale ** 0.7))

            if layer_cloud_count == 0:
                layer_cloud_count = 1

            clouds_list, locs_list = self.cloud_layers[layer_idx]

            # Position clouds: farther layers (smaller scale) should be closer to horizon
            # Layer 0 (scale 1.0): y range 0-80
            # Layer 1 (scale 0.6): y range 60-140
            # Layer 2 (scale 0.35): y range 120-200
            # Layer 3 (scale 0.2): y range 180-240
            # Layer 4 (scale 0.1): y range 220-260
            # Layer 5 (scale 0.05): y range 245-260 (very close to horizon)

            y_min = int(horizon_y * (1 - layer_scale) * 0.85)
            y_max = min(horizon_y, int(y_min + 80 + (1 - layer_scale) * 100))

            for n in range(layer_cloud_count):
                clouds_list.append(self.all_clouds_imgs_by_layer[layer_idx][size][random.randint(0, 4)])
                # Distribute clouds more evenly across the width, with some randomness
                x = (n / layer_cloud_count * GAME_WIDTH + random.randint(-50, 50)) % GAME_WIDTH
                y = random.randint(y_min, y_max)
                locs_list.append((x, y))

    def render(self, screen: pygame.surface.Surface, x_offset: int = 0):
        """Render all cloud layers from back to front"""
        # Render layers from farthest to closest (back to front)
        for layer_idx in reversed(range(len(self.cloud_layers))):
            clouds_list, locs_list = self.cloud_layers[layer_idx]
            for n in range(len(clouds_list)):
                x, y = locs_list[n]
                screen.blit(clouds_list[n], (x + x_offset, y))

    def populate(self, dest: list[list[pygame.surface.Surface]], paths: list[list[str]], scaling_factor: int):
        for size in paths:
            container = []
            for file in size:
                img = Image.open(file).convert('RGBA')
                new_size = (max(1, img.size[0]*scaling_factor), max(1, img.size[1]*scaling_factor))
                img = img.resize(new_size, Image.NEAREST)
                pygame_img = pygame.image.fromstring(
                    img.tobytes(), img.size, img.mode
                )
                container.append(pygame_img)
            dest.append(container)

def load_image(file, scaling_factor, aspect_ratio=1):
    img = Image.open(file).convert('RGBA')
    new_size = (int(img.size[0]*scaling_factor*aspect_ratio), int(img.size[1]*scaling_factor))
    img = img.resize(new_size, Image.NEAREST)
    pygame_img = pygame.image.fromstring(
        img.tobytes(), img.size, img.mode
    )
    return pygame_img

def gradientRect(window, left_colour, right_colour, target_rect):
    colour_rect = pygame.Surface((2, 2))
    pygame.draw.line(colour_rect, left_colour, (0,0), (1,0))
    pygame.draw.line(colour_rect, right_colour, (0,1), (1,1))
    colour_rect = pygame.transform.smoothscale(colour_rect, (target_rect.width, target_rect.height))
    window.blit(colour_rect, target_rect)

class ActionType(Enum):
    PLAN_CROPS = "plant_crops"
    HARVEST_CROPS = "harvest_crops"
    REST = "rest"
    WATER_PLANTS = "water_plants"
    PUMP_WATER = "pump_water"

class Calendar:
    def __init__(self, start_day: int = 1, weeks_to_show: int = 5):
        self.start_day = start_day
        self.weeks_to_show = weeks_to_show
        self.days_per_week = 7
        self.total_days = weeks_to_show * self.days_per_week

        # Calendar layout in control panel (right side)
        self.calendar_x = LEFT_PANEL_WIDTH + GAME_WIDTH + 10
        self.calendar_y = TOP_BAR_HEIGHT + 320  # Moved up from 420 to 320 (-100 pixels)
        self.day_width = 62  # Increased from 50 to 62 (+12 pixels)
        self.day_height = 62  # Increased from 50 to 62 (+12 pixels)
        self.margin = 2

        # Colors for different actions
        self.action_colors = {
            ActionType.PLAN_CROPS: (100, 200, 100),    # Light green
            ActionType.WATER_PLANTS: (100, 150, 255),  # Light blue
            ActionType.HARVEST_CROPS: (220, 150, 255), # Light purple/magenta
            ActionType.REST: (200, 200, 200),          # Light gray
            ActionType.PUMP_WATER: (150, 200, 255),    # Cyan/light blue
            None: WHITE                                # No action
        }

        # Pre-create fonts to avoid creating them every frame
        self.calendar_font = None
        self.small_font = None

    def update_for_current_day(self, current_day: int):
        """Update calendar to always keep current day on the top row"""
        # Calculate the start of the week for the current day
        # This ensures current day is always in the first week (top row)
        current_day_in_week = (current_day - 1) % self.days_per_week
        week_start = current_day - current_day_in_week

        # Set start_day to the beginning of current week
        self.start_day = week_start

    def get_day_rect(self, day_number: int) -> pygame.Rect:
        """Get the rectangle for a specific day"""
        day_offset = day_number - self.start_day
        if day_offset < 0 or day_offset >= self.total_days:
            return None

        week = day_offset // self.days_per_week
        day_in_week = day_offset % self.days_per_week

        x = self.calendar_x + day_in_week * (self.day_width + self.margin)
        y = self.calendar_y + week * (self.day_height + self.margin)

        return pygame.Rect(x, y, self.day_width, self.day_height)

    def get_day_from_pos(self, pos: tuple) -> int:
        """Get day number from mouse position, or None if not on calendar"""
        mouse_x, mouse_y = pos

        if (mouse_x < self.calendar_x or
            mouse_y < self.calendar_y or
            mouse_x > self.calendar_x + 7 * (self.day_width + self.margin) or
            mouse_y > self.calendar_y + self.weeks_to_show * (self.day_height + self.margin)):
            return None

        rel_x = mouse_x - self.calendar_x
        rel_y = mouse_y - self.calendar_y

        day_in_week = rel_x // (self.day_width + self.margin)
        week = rel_y // (self.day_height + self.margin)

        if day_in_week >= self.days_per_week or week >= self.weeks_to_show:
            return None

        day_number = self.start_day + week * self.days_per_week + day_in_week
        return day_number

    def get_resource_changes(self, action: ActionType, days_ahead: int = 0) -> dict:
        """Get resource changes for an action, with planning discount applied"""
        # Calculate discount multiplier (2% per day, max 20% at 10 days)
        if days_ahead <= 0:
            discount_multiplier = 1.0
        else:
            discount = min(0.20, days_ahead * 0.02)
            discount_multiplier = 1.0 - discount

        changes = {}
        if action == ActionType.REST:
            changes['energy'] = +8.0  # Rest doesn't benefit from discount
        elif action == ActionType.PLAN_CROPS:
            changes['energy'] = -round(20.0 * discount_multiplier, 1)
        elif action == ActionType.WATER_PLANTS:
            changes['energy'] = -round(15.0 * discount_multiplier, 1)
            changes['water'] = -round(10.0 * discount_multiplier, 1)
        elif action == ActionType.HARVEST_CROPS:
            changes['energy'] = -round(25.0 * discount_multiplier, 1)
            changes['food'] = +30.0  # Food gain doesn't change
        elif action == ActionType.PUMP_WATER:
            changes['energy'] = -round(10.0 * discount_multiplier, 1)
            changes['water'] = +20.0  # Water gain doesn't change
        return changes

    def render(self, screen: pygame.Surface, current_day: int, planned_actions: List["PlannedAction"], current_day_action: Optional[ActionType] = None, current_day_days_planned: int = 0, painted_days: set = None, resource_icons: dict = None, kbd_selected_days: set = None):
        """Render the calendar"""
        # Initialize fonts if not done already
        if self.calendar_font is None:
            self.calendar_font = pygame.font.Font(None, 24)
        if self.small_font is None:
            self.small_font = pygame.font.Font(None, 18)

        # Calendar title
        title = self.calendar_font.render("Action Calendar", True, BLACK)
        screen.blit(title, (self.calendar_x, self.calendar_y - 40))

        # Day labels
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, label in enumerate(day_labels):
            x = self.calendar_x + i * (self.day_width + self.margin) + self.day_width // 2 - 15
            text = self.small_font.render(label, True, BLACK)
            screen.blit(text, (x, self.calendar_y - 20))

        # Create action lookup for faster rendering - store both action type and days_ahead
        action_lookup = {}
        for action in planned_actions:
            action_lookup[action.day_scheduled] = (action.action_type, action.days_ahead)

        # Add current day action if set (use stored days_planned value)
        if current_day_action:
            action_lookup[current_day] = (current_day_action, current_day_days_planned)

        # Render calendar days
        for day_offset in range(self.total_days):
            day_number = self.start_day + day_offset
            rect = self.get_day_rect(day_number)

            if rect:
                # Get action for this day (returns tuple of (action_type, days_ahead) or None)
                action_data = action_lookup.get(day_number)
                if action_data:
                    action, days_ahead = action_data
                else:
                    action, days_ahead = None, 0

                color = self.action_colors.get(action, WHITE)

                # Draw day background
                pygame.draw.rect(screen, color, rect)

                # Draw border (thicker for current day, special for painted days)
                if painted_days and day_number in painted_days:
                    # Paint mode preview - thick yellow border
                    border_width = 4
                    border_color = YELLOW
                elif day_number == current_day:
                    # Current day - thick red border
                    border_width = 3
                    border_color = RED
                else:
                    # Normal day - thin black border
                    border_width = 1
                    border_color = BLACK
                pygame.draw.rect(screen, border_color, rect, border_width)

                # Draw resource changes if action is set (using stored days_ahead)
                if action:
                    changes = self.get_resource_changes(action, days_ahead)
                    y_offset = rect.top + 5
                    for resource, change in changes.items():
                        if resource_icons:
                            # Use icons instead of letters
                            icon = resource_icons.get(resource)
                            if icon:
                                # Draw icon (20x20 - doubled from 10x10)
                                icon_calendar = pygame.transform.scale(icon, (20, 20))
                                screen.blit(icon_calendar, (rect.left + 2, y_offset))
                                # Draw value next to icon
                                sign = '+' if change > 0 else ''
                                value_text = f"{sign}{change:.1f}"
                                text_surface = self.small_font.render(value_text, True, BLACK)
                                screen.blit(text_surface, (rect.left + 24, y_offset + 3))
                            else:
                                # Fallback to text
                                symbol = resource[0].upper()
                                sign = '+' if change > 0 else ''
                                change_text = f"{symbol}: {sign}{change:.1f}"
                                text_surface = self.small_font.render(change_text, True, BLACK)
                                screen.blit(text_surface, (rect.left + 2, y_offset))
                        else:
                            # Fallback to text if no icons provided
                            symbol = resource[0].upper()  # E, W, or F
                            sign = '+' if change > 0 else ''
                            change_text = f"{symbol}: {sign}{change:.1f}"
                            text_surface = self.small_font.render(change_text, True, BLACK)
                            screen.blit(text_surface, (rect.left + 2, y_offset))
                        y_offset += 22  # Increased from 12 to 22 for larger icons

        # Draw keyboard selection boxes if active
        if kbd_selected_days:
            # Draw individual boxes for each selected day
            for day in kbd_selected_days:
                rect = self.get_day_rect(day)
                if rect:
                    # Draw thick blue outline for this selected day
                    pygame.draw.rect(screen, (0, 100, 255), rect, 4)

        # Legend
        legend_y = self.calendar_y + self.weeks_to_show * (self.day_height + self.margin) + 10

        legend_items = [
            ("Plant Crops", ActionType.PLAN_CROPS),
            ("Water Plants", ActionType.WATER_PLANTS),
            ("Harvest", ActionType.HARVEST_CROPS),
            ("Rest", ActionType.REST),
            ("Pump Water", ActionType.PUMP_WATER)
        ]

        for i, (label, action) in enumerate(legend_items):
            y = legend_y + i * 20
            # Color box
            color_rect = pygame.Rect(self.calendar_x, y, 15, 15)
            pygame.draw.rect(screen, self.action_colors[action], color_rect)
            pygame.draw.rect(screen, BLACK, color_rect, 1)
            # Label
            text = self.small_font.render(label, True, BLACK)
            screen.blit(text, (self.calendar_x + 20, y))

class NotesCalendar:
    """Calendar for player notes on the left panel"""
    def __init__(self, start_day: int = 1, weeks_to_show: int = 5):
        self.start_day = start_day
        self.weeks_to_show = weeks_to_show
        self.days_per_week = 7
        self.total_days = weeks_to_show * self.days_per_week

        # Calendar layout in left panel
        self.calendar_x = 10
        self.calendar_y = TOP_BAR_HEIGHT + 320  # Same height as right calendar
        self.day_width = 62
        self.day_height = 62
        self.margin = 2

        # Fonts
        self.calendar_font = None
        self.small_font = None

    def update_for_current_day(self, current_day: int):
        """Update calendar to keep current day on top row"""
        current_day_in_week = (current_day - 1) % self.days_per_week
        week_start = current_day - current_day_in_week
        self.start_day = week_start

    def get_day_rect(self, day_number: int) -> pygame.Rect:
        """Get the rectangle for a specific day"""
        day_offset = day_number - self.start_day
        if day_offset < 0 or day_offset >= self.total_days:
            return None

        week = day_offset // self.days_per_week
        day_in_week = day_offset % self.days_per_week

        x = self.calendar_x + day_in_week * (self.day_width + self.margin)
        y = self.calendar_y + week * (self.day_height + self.margin)

        return pygame.Rect(x, y, self.day_width, self.day_height)

    def get_day_from_pos(self, pos: tuple) -> int:
        """Get day number from mouse position, or None if not on calendar"""
        mouse_x, mouse_y = pos

        if (mouse_x < self.calendar_x or
            mouse_y < self.calendar_y or
            mouse_x > self.calendar_x + 7 * (self.day_width + self.margin) or
            mouse_y > self.calendar_y + self.weeks_to_show * (self.day_height + self.margin)):
            return None

        rel_x = mouse_x - self.calendar_x
        rel_y = mouse_y - self.calendar_y

        day_in_week = rel_x // (self.day_width + self.margin)
        week = rel_y // (self.day_height + self.margin)

        if day_in_week >= self.days_per_week or week >= self.weeks_to_show:
            return None

        day_number = self.start_day + week * self.days_per_week + day_in_week
        return day_number

    def render(self, screen: pygame.Surface, current_day: int, notes: dict, selected_day: int = None):
        """Render the notes calendar with RGB color values"""
        # Initialize fonts if not done already
        if self.calendar_font is None:
            self.calendar_font = pygame.font.Font(None, 24)
        if self.small_font is None:
            self.small_font = pygame.font.Font(None, 18)

        # Calendar title
        title = self.calendar_font.render("Notes Calendar", True, BLACK)
        screen.blit(title, (self.calendar_x, self.calendar_y - 40))

        # Day labels
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, label in enumerate(day_labels):
            x = self.calendar_x + i * (self.day_width + self.margin) + self.day_width // 2 - 15
            text = self.small_font.render(label, True, BLACK)
            screen.blit(text, (x, self.calendar_y - 20))

        # Render calendar days
        for day_offset in range(self.total_days):
            day_number = self.start_day + day_offset
            rect = self.get_day_rect(day_number)

            if rect:
                # Always draw white background
                pygame.draw.rect(screen, WHITE, rect)

                # If there's a note, draw three small slider bars
                if day_number in notes:
                    r, g, b = notes[day_number]

                    # Brighter pastel colors for the mini sliders (matching main sliders)
                    slider_colors = [
                        (255, 120, 120),  # Brighter pastel red
                        (120, 255, 120),  # Brighter pastel green
                        (120, 180, 255)   # Brighter pastel blue
                    ]
                    slider_values = [r, g, b]

                    # Draw three small horizontal bars
                    bar_width = 50
                    bar_height = 6
                    bar_spacing = 3
                    start_x = rect.left + (rect.width - bar_width) // 2
                    start_y = rect.top + (rect.height - (3 * bar_height + 2 * bar_spacing)) // 2

                    for i, (color, value) in enumerate(zip(slider_colors, slider_values)):
                        bar_y = start_y + i * (bar_height + bar_spacing)

                        # Bar background (white)
                        bar_bg_rect = pygame.Rect(start_x, bar_y, bar_width, bar_height)
                        pygame.draw.rect(screen, WHITE, bar_bg_rect)

                        # Bar fill based on value (0.0 to 1.0)
                        fill_width = int(value * bar_width)
                        if fill_width > 0:
                            bar_fill_rect = pygame.Rect(start_x, bar_y, fill_width, bar_height)
                            pygame.draw.rect(screen, color, bar_fill_rect)

                        # Bar border
                        pygame.draw.rect(screen, BLACK, bar_bg_rect, 1)

                # Draw border (different for current day and selected day)
                if day_number == current_day:
                    border_width = 3
                    border_color = RED
                elif selected_day is not None and day_number == selected_day:
                    border_width = 3
                    border_color = (50, 150, 255)  # Blue for selected day
                else:
                    border_width = 1
                    border_color = BLACK
                pygame.draw.rect(screen, border_color, rect, border_width)

@dataclass
class PlannedAction:
    action_type: ActionType
    day_scheduled: int
    day_planned: int  # Day when this was planned
    days_ahead: int  # How many days ahead this was planned (for fixed discount display)
    energy_bonus: float = 0.0

@dataclass
class WeatherState:
    temperature: float  # °C
    wind_speed: float   # m/s
    cloud_cover: float  # 0-100
    is_raining: bool = False

class GameState:
    def __init__(self):
        # Player resources
        self.water = 50.0
        self.food = 50.0
        self.energy = 100.0

        # Game progression
        self.current_day = 1
        self.field_water_level = 50.0  # 0-100
        self.crop_growth = 0.0  # 0-100
        self.has_crops_planted = False

        # Crop properties
        self.optimal_soil_moisture = 50.0  # Optimal moisture level for crop growth

        # Planning system
        self.planned_actions: List[PlannedAction] = []
        self.current_day_action: Optional[ActionType] = None  # Action selected for current day
        self.current_action_days_planned: int = 0  # How many days ahead current action was planned

        # Notes system - store RGB values (0.0-1.0) for each day
        self.notes: dict = {}  # {day: (r, g, b)}

        # Weather system (Rössler attractor)
        self.weather_state = np.array([3.0 + random.random(), -15.0 + random.random(), 0.3])  # T, W, C
        self.weather_time = 0.0
        self.weather_dt = 0.005 # so this controls how fast the weather changes. 0.005-> one cycle takes like maybe 15-20 days
        
        # Game parameters
        self.rain_threshold = 150.0  # Cloud cover threshold for rain
        self.energy_decay = 10.0  # Daily energy cost
        self.crop_water_consumption = 5.0  # Daily water consumption by crops

class WeatherSystem:
    """Chaotic weather system based on Rössler attractor"""

    def __init__(self):
        # Rössler system parameters
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7

        # Scale to observable weather units
        self.scale_T = 3.0
        self.scale_W = 2.0
        self.scale_C = 15.0
        self.offset_T = 20.0

    def derivatives(self, state, t):
        T, W, C = state

        # Convert to Rössler variables
        x = (T - self.offset_T) / self.scale_T
        y = W / self.scale_W
        z = C / self.scale_C

        # Rössler equations
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)

        # Convert back to weather units
        dT = dx * self.scale_T
        dW = dy * self.scale_W
        dC = dz * self.scale_C

        return np.array([dT, dW, dC])

    def rk4_step(self, state, t, dt):
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)

        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def update(self, state, time_step, dt):
        return self.rk4_step(state, time_step, dt)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("The prediction game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 72)

        # Menu state
        self.current_menu = "main"  # "main", "play", "game", "tutorial", or "highscores"
        self.selected_menu_item = 0

        # Tutorial state
        self.tutorial_step = 0
        # Interactive tutorial steps - each highlights a region of the game screen
        self.tutorial_steps = [
            {
                "title": "Resource Bar",
                "highlight_rect": pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT),  # Top bar
                "text": [
                    "This is your resource bar. It shows:",
                    "",
                    "ENERGY - Used to perform actions",
                    "WATER - For watering crops",
                    "FOOD - You consume 1 food per day",
                    "SOIL MOISTURE - Affects crop growth",
                    "",
                    "If FOOD reaches 0, the game ends!"
                ],
                "text_position": (WINDOW_WIDTH // 2, 200)
            },
            {
                "title": "Action Calendar",
                "highlight_rect": pygame.Rect(
                    LEFT_PANEL_WIDTH + GAME_WIDTH + 10,
                    TOP_BAR_HEIGHT + 320,
                    7 * 64,  # 7 days wide (62px + 2px margin)
                    5 * 64   # 5 weeks tall
                ),
                "text": [
                    "Plan your actions here!",
                    "",
                    "Select action (keys 1-5), then click days",
                    "Planning ahead gives energy discounts",
                    "",
                    "Press N to advance to the next day"
                ],
                "text_position": (WINDOW_WIDTH // 2, 150)
            },
            {
                "title": "Notes Calendar",
                "highlight_rect": pygame.Rect(
                    10,
                    TOP_BAR_HEIGHT + 320,
                    7 * 64,  # 7 days wide
                    5 * 64   # 5 weeks tall
                ),
                "text": [
                    "Track weather patterns with notes!",
                    "",
                    "Click a day, use RGB sliders to color-code",
                    "your observations, then save",
                    "",
                    "Notes are saved to /notes folder"
                ],
                "text_position": (WINDOW_WIDTH // 2, 150)
            },
            {
                "title": "Help Button",
                "highlight_rect": pygame.Rect(10, WINDOW_HEIGHT - 60, 50, 50),
                "text": [
                    "Click the '?' button anytime during",
                    "gameplay to see all controls",
                    "",
                    "Press ESC to exit tutorial"
                ],
                "text_position": (WINDOW_WIDTH // 2, 300)
            }
        ]

        # Help overlay state
        self.show_help = False
        self.help_button_rect = pygame.Rect(10, WINDOW_HEIGHT - 60, 50, 50)

        # Game over state
        self.game_over = False
        self.final_score = 0

        # Scores directory
        self.scores_dir = "scores"
        self.scores_file = os.path.join(self.scores_dir, "highscores.json")
        self.ensure_scores_dir()

        # Notes directory
        self.notes_dir = "notes"
        self.ensure_notes_dir()

        self.game_state = GameState()
        self.weather_system = WeatherSystem()
        self.running = True
        self.selected_action = None
        self.planning_day = 1

        # Paint mode for multi-day selection
        self.is_painting = False
        self.paint_start_day = None
        self.painted_days = set()

        # Keyboard selection for calendar - now a set of individual days
        self.kbd_selected_days = {1}  # Set of selected day numbers

        # Initialize calendars
        self.calendar = Calendar(start_day=1, weeks_to_show=5)
        self.notes_calendar = NotesCalendar(start_day=1, weeks_to_show=5)

        # RGB sliders for notes (values 0.0-1.0)
        self.slider_r = 0.5
        self.slider_g = 0.5
        self.slider_b = 0.5
        self.dragging_slider = None  # None, 'r', 'g', or 'b'
        self.selected_notes_day = 1  # Day selected in the notes calendar for editing

        # Initialize graphics
        self.init_graphics()

    def ensure_scores_dir(self):
        """Create scores directory if it doesn't exist"""
        if not os.path.exists(self.scores_dir):
            os.makedirs(self.scores_dir)

    def ensure_notes_dir(self):
        """Create notes directory if it doesn't exist"""
        if not os.path.exists(self.notes_dir):
            os.makedirs(self.notes_dir)

    def save_notes(self, days_survived: int, notes: dict):
        """Save notes to a separate file in the notes directory"""
        # Format notes as 2D list
        notes_list = []
        for day in range(1, days_survived + 1):
            if day in notes:
                # Convert to list and round to 2 decimals
                notes_list.append([round(v, 2) for v in notes[day]])
            else:
                notes_list.append([])

        # Create filename with timestamp and days survived
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"notes_{timestamp}_day{days_survived}.json"
        filepath = os.path.join(self.notes_dir, filename)

        # Save notes to file
        with open(filepath, 'w') as f:
            json.dump(notes_list, f, indent=2)

        print(f"Notes saved to {filepath}")

    def save_score(self, days_survived: int):
        """Save a score to the highscores file"""
        score_entry = {
            "days": days_survived,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Load existing scores
        scores = self.load_scores()
        scores.append(score_entry)

        # Sort by days survived (descending)
        scores.sort(key=lambda x: x["days"], reverse=True)

        # Keep only top 10
        scores = scores[:10]

        # Save to file
        with open(self.scores_file, 'w') as f:
            json.dump(scores, f, indent=2)

    def load_scores(self) -> List[Dict]:
        """Load scores from the highscores file"""
        if not os.path.exists(self.scores_file):
            return []

        try:
            with open(self.scores_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def init_graphics(self):
        """Initialize all graphics assets"""
        try:
            # Load background images
            self.sky = load_image('data/misc/sky.png', 2, aspect_ratio=2)
            self.field_empty = load_image('data/misc/field_empty.png', 2)

            # Load all plant growth stages
            self.plants = [
                load_image('data/plants/plants_1.png', 2),
                load_image('data/plants/plants_2.png', 2),
                load_image('data/plants/plants_3.png', 2)
            ]

            # Load base forest images
            self.forest_m_base = load_image('data/forest/forest_m.png', 2)
            self.forest_l_base = load_image('data/forest/forest_l.png', 2)
            self.forest_r_base = load_image('data/forest/forest_r.png', 2)

            # Cache for wind-transformed forests
            self.forest_m = self.forest_m_base
            self.forest_l = self.forest_l_base
            self.forest_r = self.forest_r_base
            self.last_wind_speed = 0.0

            # Load farmer clothing layers (all scaled by 6x)
            self.farmer_body = load_image('data/farmer/farmer_body.png', 6)
            self.farmer_shirt = load_image('data/farmer/farmer_shirt.png', 6)
            self.farmer_trousers = load_image('data/farmer/farmer_trousers.png', 6)
            self.farmer_hoodie = load_image('data/farmer/farmer_hoodie.png', 6)
            self.farmer_hat = load_image('data/farmer/farmer_hat.png', 6)
            self.farmer_wooly_hat = load_image('data/farmer/farmer_wooly_hat.png', 6)

            # Load wallpaper for side panels (293x400 -> scale and crop to fit panel widths)
            wallpaper_img_original = Image.open('data/misc/wallpaper.png').convert('RGBA')
            # Scale up to make it tall enough for the panels (830 pixels tall - 30 for top bar = 800)
            scale_factor = 800 / 400  # 2x scale to make it 800 tall
            new_width = int(293 * scale_factor)  # 586 pixels wide
            new_height = int(400 * scale_factor)  # 800 pixels tall
            wallpaper_img = wallpaper_img_original.resize((new_width, new_height), Image.NEAREST)

            # Create wallpaper for both panels (both are now 470 pixels wide)
            panel_margin = (new_width - LEFT_PANEL_WIDTH) // 2
            wallpaper_panel = wallpaper_img.crop((panel_margin, 0, panel_margin + LEFT_PANEL_WIDTH, 800))

            # Both panels use the same wallpaper (same width now)
            self.wallpaper_left = pygame.image.fromstring(
                wallpaper_panel.tobytes(), wallpaper_panel.size, wallpaper_panel.mode
            )
            self.wallpaper_right = pygame.image.fromstring(
                wallpaper_panel.tobytes(), wallpaper_panel.size, wallpaper_panel.mode
            )

            # Load resource icons (40x40 -> scale to 20x20)
            icon_scale = 0.5  # Scale from 40x40 to 20x20
            self.energy_icon = load_image('data/icons/energy_icon.png', icon_scale)
            self.water_icon = load_image('data/icons/water_icon.png', icon_scale)
            self.food_icon = load_image('data/icons/food_icon.png', icon_scale)
            self.soil_moisture_icon = load_image('data/icons/soil_moisture_icon.png', icon_scale)

            # Load menu sign images (130x60 -> scale to 260x120 for better visibility)
            sign_scale = 2.0  # Scale from 130x60 to 260x120
            self.play_sign = load_image('data/signs/play_sign.png', sign_scale)
            self.highscores_sign = load_image('data/signs/highscores_sign.png', sign_scale)
            self.tutorial_sign = load_image('data/signs/tutorial_sign.png', sign_scale)
            self.survival_sign = load_image('data/signs/Survival_sign.png', sign_scale)

            # Initialize adaptive cloud system
            self.clouds = Clouds(init_size=5, init_n=3, scaling_factor=2)
            self.cloud_update_counter = 0

            # Initialize doorway
            plank_paths = ['data/planks/plank_1.png', 'data/planks/plank_2.png']
            self.doorway = Doorway(GAME_WIDTH, WINDOW_HEIGHT, 600, 740, plank_paths, (11, 92), 5, GAME_X_OFFSET)

            # Initialize bottom planks (horizontal planks at the bottom)
            self.bottom_planks = []
            self.bottom_plank_locs = []
            self.init_bottom_planks(plank_paths, (11, 92), 5)

            # Graphics available flag
            self.graphics_available = True

        except Exception as e:
            print(f"Warning: Could not load graphics assets: {e}")
            self.graphics_available = False

    def init_bottom_planks(self, plank_paths: list[str], plank_dim: tuple[int, int], scale_factor: int):
        """Initialize horizontal planks at the bottom of the screen"""
        # Calculate how many horizontal planks we need
        plank_height = plank_dim[0] * scale_factor  # Rotated, so height uses original width
        plank_width = plank_dim[1] * scale_factor   # Rotated, so width uses original height

        # Number of planks needed to fill the width
        n_planks_x = int(GAME_WIDTH / plank_width) + 2

        # Create plank positions for bottom area (similar to top horizontal planks in doorway)
        for w in range(n_planks_x):
            offset_term = (w * scale_factor * 647 % 59) * (w % 2)
            y_pos = WINDOW_HEIGHT - plank_height
            x_pos = GAME_X_OFFSET + w * plank_width - offset_term
            self.bottom_plank_locs.append((x_pos, y_pos))

        # Load and rotate plank images
        for file in plank_paths:
            img = Image.open(file).convert('RGBA')
            img = img.rotate(90, expand=True)
            new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
            img = img.resize(new_size, Image.NEAREST)
            pygame_img = pygame.image.fromstring(
                img.tobytes(), img.size, img.mode
            )
            self.bottom_planks.append(pygame_img)

    def get_farmer_color(self, temperature: float) -> tuple:
        """Get farmer color based on temperature"""
        if temperature < 10:
            return (0, 0, 255)  # Blue for cold
        elif temperature < 20:
            return (0, 255, 0)  # Green for moderate
        elif temperature < 30:
            return (255, 255, 0)  # Yellow for warm
        else:
            return (255, 0, 0)  # Red for hot

    def get_current_weather(self) -> WeatherState:
        """Convert raw weather state to readable format"""
        state = self.game_state.weather_state
        is_raining = state[2] > self.game_state.rain_threshold

        return WeatherState(
            temperature=state[0],
            wind_speed=state[1],
            cloud_cover=state[2],
            is_raining=is_raining
        )

    def render_farmer(self, x: int, y: int, temperature: float):
        """Render farmer with appropriate clothing based on temperature"""
        # Always render body first
        self.screen.blit(self.farmer_body, (x, y))

        # Determine clothing based on temperature (-10 to +55°C)
        # More evenly sized temperature ranges (each ~13 degrees):
        # > 42°C: hat + shirt (13 degrees: 42-55)
        # 29-42°C: shirt (13 degrees)
        # 16-29°C: shirt + trousers (13 degrees)
        # 3-16°C: hoodie (13 degrees)
        # < 3°C: hoodie + wooly_hat (13 degrees: -10 to 3)

        if temperature > 42:
            # Very hot: hat + shirt
            self.screen.blit(self.farmer_shirt, (x, y))
            self.screen.blit(self.farmer_hat, (x, y))
        elif temperature > 29:
            # Hot: shirt only
            self.screen.blit(self.farmer_shirt, (x, y))
        elif temperature > 16:
            # Mild: shirt + trousers
            self.screen.blit(self.farmer_shirt, (x, y))
            self.screen.blit(self.farmer_trousers, (x, y))
        elif temperature > 3:
            # Cold: hoodie
            self.screen.blit(self.farmer_hoodie, (x, y))
        else:
            # Very cold: hoodie + wooly_hat
            self.screen.blit(self.farmer_hoodie, (x, y))
            self.screen.blit(self.farmer_wooly_hat, (x, y))

    def apply_wind_to_forests(self, wind_speed: float):
        """Transform forest images based on wind speed"""
        # Only update if wind changed significantly
        if abs(wind_speed - self.last_wind_speed) < 0.5:
            return

        self.last_wind_speed = wind_speed

        # Forest images are 800x800 pixels
        # Pivot: 0.55 * 800 = 440 pixels from bottom = y=360 from top
        # At y=360: displacement = 0 (stationary)
        # At y=800 (bottom): displacement = wind_speed pixels

        for base_img, attr_name in [(self.forest_m_base, 'forest_m'),
                                      (self.forest_l_base, 'forest_l'),
                                      (self.forest_r_base, 'forest_r')]:
            width, height = base_img.get_size()  # 800x800
            pivot_y = 360  # Stationary line
            bottom_y = height  # 800

            # Convert pygame surface to PIL Image
            img_str = pygame.image.tostring(base_img, 'RGBA')
            pil_img = Image.frombytes('RGBA', (width, height), img_str)

            # Affine transformation with PIL's inverse matrix
            # We want: x_dest = x_src + 2 * wind_speed * (y_src - pivot_y) / (bottom_y - pivot_y)
            # Simplify: displacement_factor = 2 * wind_speed / 440
            # x_dest = x_src + displacement_factor * (y_src - 360)
            # Inverse: x_src = x_dest - displacement_factor * (y_dest - 360)
            # x_src = x_dest - displacement_factor * y_dest + displacement_factor * 360

            bottom_displacement_multiplier = 2
            displacement_factor = (bottom_displacement_multiplier * wind_speed) / (bottom_y - pivot_y)

            affine_matrix = (
                1,  # a: x coefficient
                -displacement_factor,  # b: y coefficient
                displacement_factor * pivot_y,  # c: constant offset
                0,  # d: x coefficient for y
                1,  # e: y coefficient for y
                0   # f: constant offset for y
            )

            # Calculate new width to accommodate the shear
            max_displacement = abs(bottom_displacement_multiplier * wind_speed)
            new_width = int(width + max_displacement * 2)  # Add padding on both sides
            new_height = height

            # Apply transformation
            transformed = pil_img.transform(
                (new_width, new_height),
                Image.AFFINE,
                affine_matrix,
                resample=Image.BILINEAR
            )

            # Convert back to pygame surface
            pygame_img = pygame.image.fromstring(
                transformed.tobytes(), transformed.size, transformed.mode
            )

            # Store the transformed image
            setattr(self, attr_name, pygame_img)

    def calculate_energy_bonus(self, days_ahead: int) -> float:
        """Calculate energy bonus for planning ahead"""
        if days_ahead <= 0:
            return 0.0
        return 10.0 * (1.0 - np.exp(-days_ahead * 0.5))

    def can_execute_action(self, action: ActionType, days_planned_ahead: int = 0) -> bool:
        """Check if player has enough resources to execute an action (with discount applied)"""
        discount_multiplier = self.calculate_planning_discount(days_planned_ahead)

        if action == ActionType.REST:
            return True  # Rest always possible
        elif action == ActionType.PLAN_CROPS:
            energy_cost = 20.0 * discount_multiplier
            return self.game_state.energy >= energy_cost and not self.game_state.has_crops_planted
        elif action == ActionType.WATER_PLANTS:
            energy_cost = 15.0 * discount_multiplier
            water_cost = 10.0 * discount_multiplier
            return self.game_state.energy >= energy_cost and self.game_state.water >= water_cost
        elif action == ActionType.HARVEST_CROPS:
            energy_cost = 25.0 * discount_multiplier
            return self.game_state.energy >= energy_cost and self.game_state.crop_growth >= 100.0
        elif action == ActionType.PUMP_WATER:
            energy_cost = 10.0 * discount_multiplier
            return self.game_state.energy >= energy_cost
        return False

    def calculate_planning_discount(self, days_ahead: int) -> float:
        """Calculate resource cost discount for planning ahead (up to 20% for 10+ days)"""
        if days_ahead <= 0:
            return 1.0  # No discount
        discount = min(0.20, days_ahead * 0.02)  # 2% per day, max 20%
        return 1.0 - discount

    def execute_action(self, action: ActionType, days_planned_ahead: int = 0) -> bool:
        """Execute an action and return success status. days_planned_ahead provides discount."""
        discount_multiplier = self.calculate_planning_discount(days_planned_ahead)

        if action == ActionType.REST:
            self.game_state.energy = min(100.0, self.game_state.energy + 8.0)
            return True
        elif action == ActionType.PLAN_CROPS:
            energy_cost = round(20.0 * discount_multiplier, 1)
            if self.game_state.energy >= energy_cost and not self.game_state.has_crops_planted:
                self.game_state.energy -= energy_cost
                self.game_state.has_crops_planted = True
                self.game_state.crop_growth = 0.0
                if days_planned_ahead > 0:
                    print(f"  (Planned {days_planned_ahead} days ahead: {(1-discount_multiplier)*100:.0f}% discount, cost {energy_cost:.1f} energy)")
                return True
        elif action == ActionType.WATER_PLANTS:
            energy_cost = round(15.0 * discount_multiplier, 1)
            water_cost = round(10.0 * discount_multiplier, 1)
            if self.game_state.energy >= energy_cost and self.game_state.water >= water_cost:
                self.game_state.energy -= energy_cost
                self.game_state.water -= water_cost
                self.game_state.field_water_level = min(100.0, self.game_state.field_water_level + 25.0)
                if days_planned_ahead > 0:
                    print(f"  (Planned {days_planned_ahead} days ahead: {(1-discount_multiplier)*100:.0f}% discount, cost {energy_cost:.1f}E {water_cost:.1f}W)")
                return True
        elif action == ActionType.HARVEST_CROPS:
            energy_cost = round(25.0 * discount_multiplier, 1)
            if self.game_state.energy >= energy_cost and self.game_state.crop_growth >= 100.0:
                self.game_state.energy -= energy_cost
                self.game_state.food += 30.0
                self.game_state.has_crops_planted = False
                self.game_state.crop_growth = 0.0
                if days_planned_ahead > 0:
                    print(f"  (Planned {days_planned_ahead} days ahead: {(1-discount_multiplier)*100:.0f}% discount, cost {energy_cost:.1f} energy)")
                return True
        elif action == ActionType.PUMP_WATER:
            energy_cost = round(10.0 * discount_multiplier, 1)
            if self.game_state.energy >= energy_cost:
                self.game_state.energy -= energy_cost
                self.game_state.water = min(200.0, self.game_state.water + 20.0)
                if days_planned_ahead > 0:
                    print(f"  (Planned {days_planned_ahead} days ahead: {(1-discount_multiplier)*100:.0f}% discount, cost {energy_cost:.1f} energy)")
                return True
        return False

    def update_game_logic(self):
        """Update game state for each day"""
        # Store old calendar start to detect if it changed
        old_calendar_start = self.calendar.start_day

        # Update calendars for new day
        self.calendar.update_for_current_day(self.game_state.current_day)
        self.notes_calendar.update_for_current_day(self.game_state.current_day)

        # If calendar moved forward, adjust keyboard selection
        if self.calendar.start_day != old_calendar_start:
            offset = self.calendar.start_day - old_calendar_start
            # Shift all selected days by the offset and clamp to visible range
            new_selection = set()
            calendar_end = self.calendar.start_day + 34
            for day in self.kbd_selected_days:
                new_day = day + offset
                # Clamp to visible calendar range
                new_day = max(self.calendar.start_day, min(calendar_end, new_day))
                new_selection.add(new_day)
            self.kbd_selected_days = new_selection

        # Update weather (adjusted for the slower weather_dt)
        weather_steps = int(0.25 / self.game_state.weather_dt)  # About 1/4 day of weather progression
        for _ in range(weather_steps):
            self.game_state.weather_state = self.weather_system.update(
                self.game_state.weather_state,
                self.game_state.weather_time,
                self.game_state.weather_dt
            )
            self.game_state.weather_time += self.game_state.weather_dt

        weather = self.get_current_weather()

        # Apply rain effects
        if weather.is_raining:
            self.game_state.field_water_level = min(100.0, self.game_state.field_water_level + 15.0)
            self.game_state.water = min(200.0, self.game_state.water + 10.0)  # Collect rainwater (max 200)

        # Update crops based on soil moisture preference
        if self.game_state.has_crops_planted:
            # Calculate how far from optimal moisture we are
            moisture_diff = abs(self.game_state.field_water_level - self.game_state.optimal_soil_moisture)

            # Growth rate depends on how close to optimal moisture
            # At optimal (diff = 0): max growth = 5.0 per day (slower growth)
            # At diff = 50 (max deviation): min growth = 0.0 per day
            max_growth_rate = 5.0
            growth_rate = max_growth_rate * (1.0 - (moisture_diff / 50.0))
            growth_rate = max(0.0, growth_rate)  # Ensure non-negative

            self.game_state.crop_growth = min(100.0, self.game_state.crop_growth + growth_rate)
            self.game_state.field_water_level -= self.game_state.crop_water_consumption

        # Execute any planned actions for today (future actions that are now current)
        today_actions = [a for a in self.game_state.planned_actions if a.day_scheduled == self.game_state.current_day]
        for action in today_actions:
            # If no current day action is set, use the planned one
            if not self.game_state.current_day_action:
                self.game_state.current_day_action = action.action_type
                # Use the stored days_ahead from when it was originally planned
                self.game_state.current_action_days_planned = action.days_ahead
                print(f"Using planned action for today: {action.action_type.value} (planned {action.days_ahead} days ahead)")
            # Remove the planned action since it's now the current day action
            self.game_state.planned_actions.remove(action)

        # Daily resource changes
        self.game_state.food = max(0.0, self.game_state.food - 1.0)  # Farmer needs to eat daily
        self.game_state.field_water_level = max(0.0, self.game_state.field_water_level - 2.0)  # Natural evaporation

        # Check for game over condition
        if self.game_state.food <= 0.0:
            print(f"GAME OVER! You ran out of food on day {self.game_state.current_day}.")

            # Format and print all notes as a 2D list
            max_day = self.game_state.current_day
            notes_list = []
            for day in range(1, max_day + 1):
                if day in self.game_state.notes:
                    notes_list.append(list(self.game_state.notes[day]))
                else:
                    notes_list.append([])

            print("Player notes:")
            # only 2 decimals needed
            notes_list = [[f"{i:.2f}" for i in j] for j in notes_list]
            print(notes_list)

            # Set game over state
            self.game_over = True
            self.final_score = self.game_state.current_day
            self.selected_menu_item = 0  # Reset for game over menu

            # Save score to highscores
            self.save_score(self.final_score)

            # Save notes to notes folder
            self.save_notes(self.final_score, self.game_state.notes)

    def draw_ui(self):
        """Draw the game UI with graphics and control panel"""
        # Clear screen
        self.screen.fill(WHITE)

        # Draw game world (center)
        self.draw_game_world()

        # Draw control panel (right side)
        self.draw_control_panel()

        # Draw left panel with instructions (on top of game world)
        self.draw_left_panel()

        # Draw top bar LAST so nothing covers it
        self.draw_top_bar()

        # Draw help overlay if active (render last so it's on top)
        if self.show_help:
            self.draw_help_overlay()

        # Draw game over overlay if game is over (render last so it's on top of everything)
        if self.game_over:
            self.draw_game_over_overlay()

    def draw_left_panel(self):
        """Draw the left panel with notes calendar and RGB sliders"""
        # Draw wallpaper background
        if self.graphics_available:
            self.screen.blit(self.wallpaper_left, (0, TOP_BAR_HEIGHT))
        else:
            panel_rect = pygame.Rect(0, TOP_BAR_HEIGHT, LEFT_PANEL_WIDTH, WINDOW_HEIGHT - TOP_BAR_HEIGHT)
            pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.line(self.screen, BLACK, (LEFT_PANEL_WIDTH, TOP_BAR_HEIGHT), (LEFT_PANEL_WIDTH, WINDOW_HEIGHT), 2)

        # Draw notes calendar at the top
        self.notes_calendar.render(self.screen, self.game_state.current_day, self.game_state.notes, self.selected_notes_day)

        # Draw RGB sliders below the calendar, side-by-side
        slider_y_start = self.notes_calendar.calendar_y + self.notes_calendar.weeks_to_show * (self.notes_calendar.day_height + self.notes_calendar.margin) + 40
        slider_width = 130
        slider_height = 30
        slider_spacing = 10
        slider_x_start = 20

        # Brighter pastel colors for sliders
        slider_colors = [
            (255, 120, 120),  # Brighter pastel red
            (120, 255, 120),  # Brighter pastel green
            (120, 180, 255)   # Brighter pastel blue
        ]
        slider_values = [self.slider_r, self.slider_g, self.slider_b]

        self.slider_rects = []
        for i, (color, value) in enumerate(zip(slider_colors, slider_values)):
            x = slider_x_start + i * (slider_width + slider_spacing)

            # Slider background
            slider_rect = pygame.Rect(x, slider_y_start, slider_width, slider_height)
            self.slider_rects.append(slider_rect)
            pygame.draw.rect(self.screen, WHITE, slider_rect)
            pygame.draw.rect(self.screen, BLACK, slider_rect, 2)

            # Slider fill
            fill_width = int(value * slider_width)
            fill_rect = pygame.Rect(x, slider_y_start, fill_width, slider_height)
            pygame.draw.rect(self.screen, color, fill_rect)

            # Slider handle
            handle_x = x + fill_width
            handle_rect = pygame.Rect(handle_x - 3, slider_y_start - 3, 6, slider_height + 6)
            pygame.draw.rect(self.screen, BLACK, handle_rect)

        # Save button below sliders, positioned on the right side of the left panel
        save_button_y = slider_y_start + slider_height + 20
        save_button_width = 150
        save_button_x = LEFT_PANEL_WIDTH - save_button_width - 20  # Right side with margin
        save_button_rect = pygame.Rect(save_button_x, save_button_y, save_button_width, 40)
        self.save_button_rect = save_button_rect  # Store for click detection

        pygame.draw.rect(self.screen, (180, 230, 180), save_button_rect)  # Pastel green
        pygame.draw.rect(self.screen, BLACK, save_button_rect, 2)
        save_text = self.small_font.render("Save Note", True, BLACK)
        save_text_rect = save_text.get_rect(center=save_button_rect.center)
        self.screen.blit(save_text, save_text_rect)

        # Draw help button (circle with "?")
        pygame.draw.circle(self.screen, (150, 220, 150),
                          (self.help_button_rect.centerx, self.help_button_rect.centery), 25)
        pygame.draw.circle(self.screen, BLACK,
                          (self.help_button_rect.centerx, self.help_button_rect.centery), 25, 2)
        help_text = self.font.render("?", True, BLACK)
        help_text_rect = help_text.get_rect(center=(self.help_button_rect.centerx, self.help_button_rect.centery))
        self.screen.blit(help_text, help_text_rect)

    def draw_help_overlay(self):
        """Draw the help overlay window"""
        # Semi-transparent dark overlay for background
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Help window (pastel green)
        help_window_width = 600
        help_window_height = 550
        help_x = (WINDOW_WIDTH - help_window_width) // 2
        help_y = (WINDOW_HEIGHT - help_window_height) // 2
        help_rect = pygame.Rect(help_x, help_y, help_window_width, help_window_height)

        pygame.draw.rect(self.screen, (180, 230, 180), help_rect)  # Pastel green
        pygame.draw.rect(self.screen, BLACK, help_rect, 3)

        # Title
        title = self.font.render("How to Execute Actions", True, BLACK)
        self.screen.blit(title, (help_x + 20, help_y + 20))

        # Instructions
        instructions_y = help_y + 70
        instructions = [
            "Select Action (1-5 keys):",
            "1: Plant Crops",
            "2: Water Plants",
            "3: Harvest Crops",
            "4: Rest",
            "5: Pump Water",
            "",
            "Keyboard Controls:",
            "Arrow keys: Move selection box",
            "Shift + Arrows: Resize selection box",
            "Enter: Assign action to selected days",
            "N: Advance to next day",
            "",
            "Mouse Controls:",
            "Click calendar: Set action for one day",
            "Drag calendar: Set action for multiple days",
        ]

        for i, line in enumerate(instructions):
            text = self.small_font.render(line, True, BLACK)
            self.screen.blit(text, (help_x + 20, instructions_y + i * 25))

        # Close instruction
        close_y = instructions_y + len(instructions) * 25 + 30
        close_text = self.small_font.render("Click anywhere or press ESC to close", True, BLACK)
        self.screen.blit(close_text, (help_x + 20, close_y))

    def draw_top_bar(self):
        """Draw the top resource bar across the whole window"""
        # Top bar background - bright cyan sky color
        top_bar_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (100, 200, 255), top_bar_rect)  # Bright cyan
        pygame.draw.line(self.screen, BLACK, (0, TOP_BAR_HEIGHT), (WINDOW_WIDTH, TOP_BAR_HEIGHT), 2)

        # Resource bars in top bar
        bar_width = 120
        bar_height = 20
        bar_y = 5
        spacing = 150

        # Energy bar (red) with icon
        if self.graphics_available:
            self.draw_compact_resource_bar(10, bar_y, bar_width, bar_height,
                                         self.game_state.energy, 100, (255, 100, 100), self.energy_icon)
        else:
            self.draw_compact_resource_bar(10, bar_y, bar_width, bar_height,
                                         self.game_state.energy, 100, (255, 100, 100), "Energy")

        # Water bar (blue) - increased capacity to 200 with icon
        if self.graphics_available:
            self.draw_compact_resource_bar(10 + spacing, bar_y, bar_width, bar_height,
                                         self.game_state.water, 200, (100, 150, 255), self.water_icon)
        else:
            self.draw_compact_resource_bar(10 + spacing, bar_y, bar_width, bar_height,
                                         self.game_state.water, 200, (100, 150, 255), "Water")

        # Food bar (green) with icon
        if self.graphics_available:
            self.draw_compact_resource_bar(10 + spacing * 2, bar_y, bar_width, bar_height,
                                         self.game_state.food, 100, (100, 200, 100), self.food_icon)
        else:
            self.draw_compact_resource_bar(10 + spacing * 2, bar_y, bar_width, bar_height,
                                         self.game_state.food, 100, (100, 200, 100), "Food")

        # Field water bar (cyan) with icon
        if self.graphics_available:
            self.draw_compact_resource_bar(10 + spacing * 3, bar_y, bar_width, bar_height,
                                         self.game_state.field_water_level, 100, (100, 255, 200), self.soil_moisture_icon)
        else:
            self.draw_compact_resource_bar(10 + spacing * 3, bar_y, bar_width, bar_height,
                                         self.game_state.field_water_level, 100, (100, 255, 200), "Field")

        # Day counter on the right
        day_text = f"Day {self.game_state.current_day}"
        day_surface = self.small_font.render(day_text, True, BLACK)
        day_x = WINDOW_WIDTH - day_surface.get_width() - 10
        self.screen.blit(day_surface, (day_x, 8))

    def draw_compact_resource_bar(self, x, y, width, height, current, maximum, color, label):
        """Draw a compact resource bar for the top bar"""
        # Background
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height))
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height), 1)

        # Fill bar based on percentage
        fill_width = int((current / maximum) * (width - 2))
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, fill_width, height - 2))

        # Draw icon or text label with value
        if isinstance(label, str):
            # Text label (fallback)
            label_text = f"{label}: {current:.1f}"
            text_surface = self.small_font.render(label_text, True, BLACK)
            text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
            self.screen.blit(text_surface, text_rect)
        else:
            # Icon (pygame surface) - draw icon on left, value on right
            icon_y = y + (height - label.get_height()) // 2
            self.screen.blit(label, (x + 3, icon_y))
            # Draw value text
            value_text = f"{current:.1f}"
            text_surface = self.small_font.render(value_text, True, BLACK)
            # Position text to the right of center
            text_x = x + width // 2 + 10
            text_rect = text_surface.get_rect(center=(text_x, y + height // 2))
            self.screen.blit(text_surface, text_rect)

    def draw_game_world(self):
        """Draw the main game world graphics"""
        weather = self.get_current_weather()

        if self.graphics_available:
            # Sky background - use actual sky image (offset by left panel and top bar)
            self.screen.blit(self.sky, (LEFT_PANEL_WIDTH, TOP_BAR_HEIGHT))

            # Render weather-dependent clouds
            self.render_weather_clouds(weather)

            # Render field_empty (400x400 scaled 2x = 800x800, offset by left panel and top bar)
            self.screen.blit(self.field_empty, (GAME_X_OFFSET, TOP_BAR_HEIGHT))

            # Render plant on top of field based on growth stage (only if planted and growing)
            if self.game_state.has_crops_planted and self.game_state.crop_growth >= 15.0:
                # Determine which plant image to show based on growth
                if self.game_state.crop_growth < 40.0:
                    plant_stage = 0  # plants_1.png (15-39% growth)
                elif self.game_state.crop_growth < 70.0:
                    plant_stage = 1  # plants_2.png (40-69% growth)
                else:
                    plant_stage = 2  # plants_3.png (70-100% growth)

                self.screen.blit(self.plants[plant_stage], (GAME_X_OFFSET, TOP_BAR_HEIGHT))

                # Display growth percentage on the field
                growth_text = f"{self.game_state.crop_growth:.1f}%"
                growth_surface = self.font.render(growth_text, True, WHITE)
                # Position text in the center-bottom of the field area
                text_x = GAME_X_OFFSET + 400 - growth_surface.get_width() // 2
                text_y = TOP_BAR_HEIGHT + 700
                # Draw black outline for better visibility
                outline_color = BLACK
                for dx in [-2, 0, 2]:
                    for dy in [-2, 0, 2]:
                        if dx != 0 or dy != 0:
                            outline_surface = self.font.render(growth_text, True, outline_color)
                            self.screen.blit(outline_surface, (text_x + dx, text_y + dy))
                # Draw the white text on top
                self.screen.blit(growth_surface, (text_x, text_y))

            # Apply wind transformation to forests
            self.apply_wind_to_forests(weather.wind_speed)

            # Render forest_m first (behind other forests) (400x400 scaled 2x = 800x800, offset by left panel and top bar)
            self.screen.blit(self.forest_m, (GAME_X_OFFSET, TOP_BAR_HEIGHT))

            # Render forest_l (400x400 scaled 2x = 800x800, offset by left panel and top bar)
            self.screen.blit(self.forest_l, (GAME_X_OFFSET, TOP_BAR_HEIGHT))

            # Render forest_r (400x400 scaled 2x = 800x800, offset by left panel and top bar)
            self.screen.blit(self.forest_r, (GAME_X_OFFSET, TOP_BAR_HEIGHT))

            # Rain effect (render before farmer and doorway)
            if weather.is_raining:
                self.draw_rain_effect()

            # Apply darkness overlay based on cloud coverage (before planks and farmer)
            self.apply_cloud_darkness(weather.cloud_cover)

            # Render doorway (on top of rain and darkness)
            self.doorway.render(self.screen)

            # Render bottom planks (horizontal planks at bottom)
            for plank_pos in self.bottom_plank_locs:
                self.screen.blit(self.bottom_planks[0], plank_pos)

            # Render farmer on top of everything with temperature-appropriate clothing
            self.render_farmer(GAME_X_OFFSET + 100, TOP_BAR_HEIGHT + 300, weather.temperature)

        else:
            # Fallback simple graphics
            self.draw_simple_graphics(weather)

    def apply_cloud_darkness(self, cloud_cover):
        """Apply a darkness overlay based on cloud coverage"""
        # Cloud coverage ranges from 0 to ~400 (max)
        # Calculate darkness alpha (0 = no darkness, 255 = fully dark)
        # At cloud_cover = 0: alpha = 0
        # At cloud_cover = 400: alpha = ~126 (scaled by 0.7)
        max_cloud_cover = 400
        max_alpha = 180

        alpha = min(max_alpha, int((cloud_cover / max_cloud_cover) * max_alpha * 0.7))

        if alpha > 0:
            # Create a semi-transparent black overlay for game world only
            darkness_overlay = pygame.Surface((GAME_WIDTH, WINDOW_HEIGHT - TOP_BAR_HEIGHT))
            darkness_overlay.set_alpha(alpha)
            darkness_overlay.fill((0, 0, 0))
            self.screen.blit(darkness_overlay, (GAME_X_OFFSET, TOP_BAR_HEIGHT))

    def render_weather_clouds(self, weather):
        """Render clouds based on current weather conditions"""
        # Generate clouds once per day
        if self.cloud_update_counter != self.game_state.current_day:
            self.update_clouds_for_weather(weather.cloud_cover)
            self.cloud_update_counter = self.game_state.current_day

        self.clouds.render(self.screen, GAME_X_OFFSET)

    def update_clouds_for_weather(self, cloud_cover):
        """Update cloud configuration based on cloud cover value"""
        # Clear existing clouds from all layers
        for layer_idx in range(len(self.clouds.cloud_layers)):
            self.clouds.cloud_layers[layer_idx] = ([], [])

        # Determine cloud parameters based on cloud cover
        # Rain threshold is 150, so we need lots of clouds above that value
        if cloud_cover < 0.4:
            # Clear sky - no clouds
            num_clouds = 0
            cloud_size_index = 0
        elif cloud_cover < 1.0:
            # Light clouds - small clouds only
            num_clouds = 2 + int((cloud_cover - 0.4) * 5)  # 2-5 clouds
            cloud_size_index = min(1, int((cloud_cover - 0.4) * 2))  # sizes 3,5
        elif cloud_cover < 10:
            # Some clouds - small to medium
            num_clouds = 5 + int((cloud_cover - 1) * 1.5)  # 5-18 clouds
            cloud_size_index = 1 + int((cloud_cover - 1) / 3)  # sizes 5,8,13
        elif cloud_cover < 50:
            # Moderate clouds
            num_clouds = 10 + int((cloud_cover - 10) / 2)  # 10-30 clouds
            cloud_size_index = 3 + int((cloud_cover - 10) / 15)  # sizes 13,17,25
        elif cloud_cover < 100:
            # Heavy clouds
            num_clouds = 25 + int((cloud_cover - 50) / 1.5)  # 25-58 clouds
            cloud_size_index = 4 + int((cloud_cover - 50) / 20)  # sizes 17,25,35
        elif cloud_cover < 150:
            # Very heavy clouds approaching rain
            num_clouds = 50 + int((cloud_cover - 100))  # 50-100 clouds
            cloud_size_index = min(7, 5 + int((cloud_cover - 100) / 20))  # sizes 25,35,51
        else:
            # Rainy sky - completely overcast
            num_clouds = 100 + int((cloud_cover - 150) / 2)  # 100+ clouds
            cloud_size_index = min(8, 6 + int((cloud_cover - 150) / 50))  # sizes 35,51,70

        # Add clouds with determined parameters
        if num_clouds > 0:
            self.clouds.add_clouds(num_clouds, cloud_size_index)

    def draw_rain_effect(self):
        """Draw simple rain lines"""
        for _ in range(50):
            x = random.randint(GAME_X_OFFSET, GAME_X_OFFSET + GAME_WIDTH)
            y = random.randint(TOP_BAR_HEIGHT, WINDOW_HEIGHT)
            pygame.draw.line(self.screen, BLUE, (x, y), (x - 5, y + 15), 1)

    def draw_simple_graphics(self, weather):
        """Fallback graphics when assets aren't available"""
        # Sky (offset by top bar)
        pygame.draw.rect(self.screen, SKY_BLUE, (0, TOP_BAR_HEIGHT, GAME_WIDTH, 320))

        # Ground (offset by top bar)
        pygame.draw.rect(self.screen, GROUND_GREEN, (0, TOP_BAR_HEIGHT + 320, GAME_WIDTH, 480))

        # Simple tree (offset by top bar)
        pygame.draw.rect(self.screen, BROWN, (350, TOP_BAR_HEIGHT + 250, 20, 100))
        pygame.draw.circle(self.screen, GREEN, (360, TOP_BAR_HEIGHT + 250), 40)

        # Field (offset by top bar)
        if self.game_state.has_crops_planted:
            pygame.draw.rect(self.screen, GREEN, (40, TOP_BAR_HEIGHT + 500, 200, 100))

        # Farmer (offset by top bar)
        farmer_color = self.get_farmer_color(weather.temperature)
        pygame.draw.circle(self.screen, farmer_color, (100, TOP_BAR_HEIGHT + 400), 20)

        # Rain
        if weather.is_raining:
            self.draw_rain_effect()

    def draw_resource_bar(self, x, y, width, height, current, maximum, color, label):
        """Draw a resource bar with label"""
        # Background
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height))
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height), 2)

        # Fill bar based on percentage
        fill_width = int((current / maximum) * (width - 4))
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (x + 2, y + 2, fill_width, height - 4))

        # Label and value
        label_text = self.small_font.render(f"{label}: {current:.1f}/{maximum:.1f}", True, BLACK)
        self.screen.blit(label_text, (x, y - 20))

    def draw_control_panel(self):
        """Draw the control panel on the right side"""
        panel_x = GAME_X_OFFSET + GAME_WIDTH
        # Draw wallpaper background
        if self.graphics_available:
            self.screen.blit(self.wallpaper_right, (panel_x, TOP_BAR_HEIGHT))
        else:
            panel_rect = pygame.Rect(panel_x, TOP_BAR_HEIGHT, CONTROL_PANEL_WIDTH, WINDOW_HEIGHT - TOP_BAR_HEIGHT)
            pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.line(self.screen, BLACK, (panel_x, TOP_BAR_HEIGHT), (panel_x, WINDOW_HEIGHT), 2)

        # Title
        title = self.font.render("The prediction game", True, BLACK)
        self.screen.blit(title, (panel_x + 10, TOP_BAR_HEIGHT + 10))

        # Crop status
        crop_y = TOP_BAR_HEIGHT + 60
        if self.game_state.has_crops_planted:
            crop_text = f"Crop Growth: {self.game_state.crop_growth:.1f}/100"
            if self.game_state.crop_growth >= 100.0:
                crop_text += " (Ready!)"
        else:
            crop_text = "No crops planted"

        crop_surface = self.small_font.render(crop_text, True, BLACK)
        self.screen.blit(crop_surface, (panel_x + 10, crop_y))

        # Current day action status
        current_action_y = TOP_BAR_HEIGHT + 200
        if self.game_state.current_day_action:
            current_action_text = f"Today's Action: {self.game_state.current_day_action.value}"
            current_action_color = DARK_BLUE
        else:
            current_action_text = "Today's Action: NOT SET (Required!)"
            current_action_color = RED

        current_action_surface = self.small_font.render(current_action_text, True, current_action_color)
        self.screen.blit(current_action_surface, (panel_x + 10, current_action_y))

        # Render calendar with resource icons
        resource_icons = None
        if self.graphics_available:
            resource_icons = {
                'energy': self.energy_icon,
                'water': self.water_icon,
                'food': self.food_icon
            }
        self.calendar.render(self.screen, self.game_state.current_day, self.game_state.planned_actions, self.game_state.current_day_action, self.game_state.current_action_days_planned, self.painted_days, resource_icons, self.kbd_selected_days)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle menu navigation
            if self.current_menu in ["main", "play"]:
                self.handle_menu_input(event)
                continue

            # Handle tutorial
            if self.current_menu == "tutorial":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.current_menu = "main"
                        self.selected_menu_item = 0
                        self.tutorial_step = 0
                    elif event.key == pygame.K_RIGHT:
                        if self.tutorial_step < len(self.tutorial_steps) - 1:
                            self.tutorial_step += 1
                    elif event.key == pygame.K_LEFT:
                        if self.tutorial_step > 0:
                            self.tutorial_step -= 1
                continue

            # Handle highscores menu
            if self.current_menu == "highscores":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.current_menu = "main"
                    self.selected_menu_item = 0
                continue

            # Handle game over overlay
            if self.game_over:
                self.handle_menu_input(event)
                continue

            # Handle game events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Close help overlay if open
                    if self.show_help:
                        self.show_help = False
                elif event.key == pygame.K_1:
                    self.selected_action = ActionType.PLAN_CROPS
                elif event.key == pygame.K_2:
                    self.selected_action = ActionType.WATER_PLANTS
                elif event.key == pygame.K_3:
                    self.selected_action = ActionType.HARVEST_CROPS
                elif event.key == pygame.K_4:
                    self.selected_action = ActionType.REST
                elif event.key == pygame.K_5:
                    self.selected_action = ActionType.PUMP_WATER
                elif event.key == pygame.K_LEFT:
                    self.handle_kbd_selection_move(-1, event.mod & pygame.KMOD_SHIFT)
                elif event.key == pygame.K_RIGHT:
                    self.handle_kbd_selection_move(1, event.mod & pygame.KMOD_SHIFT)
                elif event.key == pygame.K_UP:
                    self.handle_kbd_selection_move(-7, event.mod & pygame.KMOD_SHIFT)
                elif event.key == pygame.K_DOWN:
                    self.handle_kbd_selection_move(7, event.mod & pygame.KMOD_SHIFT)
                elif event.key == pygame.K_RETURN:
                    self.handle_kbd_selection_assign()
                elif event.key == pygame.K_n:
                    # Advance to next day - only if current day action is set and can be executed
                    if self.game_state.current_day_action:
                        # Check if player has enough resources to execute the action (with discount)
                        if self.can_execute_action(self.game_state.current_day_action, self.game_state.current_action_days_planned):
                            # Execute current day action with planning discount
                            action_executed = self.execute_action(
                                self.game_state.current_day_action,
                                self.game_state.current_action_days_planned
                            )
                            if action_executed:
                                print(f"Executed today's action: {self.game_state.current_day_action.value}")
                            else:
                                print(f"Failed to execute action: {self.game_state.current_day_action.value} (insufficient resources)")

                            # Clear current day action and planning info
                            self.game_state.current_day_action = None
                            self.game_state.current_action_days_planned = 0

                            # Advance to next day
                            self.game_state.current_day += 1
                            self.selected_notes_day = self.game_state.current_day  # Update selected day
                            self.update_game_logic()
                            print(f"Advanced to day {self.game_state.current_day}")
                        else:
                            print(f"Cannot advance day: Insufficient resources for {self.game_state.current_day_action.value}!")
                    else:
                        print("Cannot advance day: You must select an action for today first!")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.handle_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.handle_mouse_up(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self.is_painting:
                    self.handle_mouse_drag(event.pos)
                elif self.dragging_slider:
                    self.handle_slider_drag(event.pos)

    def handle_calendar_click(self, mouse_pos):
        """Handle mouse clicks on the calendar"""
        clicked_day = self.calendar.get_day_from_pos(mouse_pos)

        if clicked_day is not None and self.selected_action is not None:
            # Check if day is in the future or current day
            if clicked_day >= self.game_state.current_day:
                if clicked_day == self.game_state.current_day:
                    # Set action for current day (with 0 days planning ahead)
                    self.game_state.current_day_action = self.selected_action
                    self.game_state.current_action_days_planned = 0
                    print(f"Set action for today: {self.selected_action.value}")
                else:
                    # Remove any existing action for this future day
                    self.game_state.planned_actions = [
                        action for action in self.game_state.planned_actions
                        if action.day_scheduled != clicked_day
                    ]

                    # Calculate energy bonus for planning ahead
                    days_ahead = clicked_day - self.game_state.current_day
                    energy_bonus = self.calculate_energy_bonus(days_ahead)

                    # Add new planned action with days_ahead stored
                    planned_action = PlannedAction(
                        action_type=self.selected_action,
                        day_scheduled=clicked_day,
                        day_planned=self.game_state.current_day,
                        days_ahead=days_ahead,
                        energy_bonus=energy_bonus
                    )
                    self.game_state.planned_actions.append(planned_action)

                    print(f"Planned {self.selected_action.value} for day {clicked_day} (+{energy_bonus:.1f} energy bonus)")
            else:
                print(f"Cannot plan for past day {clicked_day}")
        elif clicked_day is not None and self.selected_action is None:
            print("Select an action first (1-4 keys)")

    def handle_mouse_down(self, mouse_pos):
        """Handle mouse button down - start painting or single click"""
        # Check if help button was clicked
        if self.help_button_rect.collidepoint(mouse_pos):
            self.show_help = not self.show_help
            return

        # If help overlay is open, check if clicked outside the help window to close it
        if self.show_help:
            help_width = 600
            help_height = 550
            help_x = (WINDOW_WIDTH - help_width) // 2
            help_y = (WINDOW_HEIGHT - help_height) // 2
            help_rect = pygame.Rect(help_x, help_y, help_width, help_height)

            if not help_rect.collidepoint(mouse_pos):
                self.show_help = False
            return

        # Check if notes calendar was clicked
        clicked_notes_day = self.notes_calendar.get_day_from_pos(mouse_pos)
        if clicked_notes_day is not None:
            self.selected_notes_day = clicked_notes_day
            # Load existing note values for this day if they exist
            if clicked_notes_day in self.game_state.notes:
                r, g, b = self.game_state.notes[clicked_notes_day]
                self.slider_r = r
                self.slider_g = g
                self.slider_b = b
            else:
                # Reset sliders to default for days without notes
                self.slider_r = 0.5
                self.slider_g = 0.5
                self.slider_b = 0.5
            print(f"Selected day {clicked_notes_day} for note editing")
            return

        # Check if save button was clicked
        if hasattr(self, 'save_button_rect') and self.save_button_rect.collidepoint(mouse_pos):
            # Save current slider values to notes for selected day
            self.game_state.notes[self.selected_notes_day] = (self.slider_r, self.slider_g, self.slider_b)
            print(f"Saved note for day {self.selected_notes_day}: RGB({self.slider_r:.2f}, {self.slider_g:.2f}, {self.slider_b:.2f})")
            return

        # Check if any slider was clicked
        if hasattr(self, 'slider_rects'):
            for i, rect in enumerate(self.slider_rects):
                if rect.collidepoint(mouse_pos):
                    self.dragging_slider = ['r', 'g', 'b'][i]
                    self.handle_slider_drag(mouse_pos)
                    return

        clicked_day = self.calendar.get_day_from_pos(mouse_pos)

        if clicked_day is not None and self.selected_action is not None:
            # Start paint mode
            self.is_painting = True
            self.paint_start_day = clicked_day
            self.painted_days = {clicked_day}

            # Apply action to the first day
            self.apply_action_to_day(clicked_day, self.selected_action)
        elif clicked_day is not None and self.selected_action is None:
            print("Select an action first (1-4 keys)")

    def handle_mouse_up(self, mouse_pos):
        """Handle mouse button up - finish painting or slider dragging"""
        if self.is_painting:
            print(f"Applied {self.selected_action.value} to {len(self.painted_days)} days")

        # Reset paint mode
        self.is_painting = False
        self.paint_start_day = None
        self.painted_days = set()

        # Reset slider dragging
        self.dragging_slider = None

    def handle_mouse_drag(self, mouse_pos):
        """Handle mouse drag - continue painting"""
        if not self.is_painting or not self.selected_action:
            return

        clicked_day = self.calendar.get_day_from_pos(mouse_pos)

        if clicked_day is not None and clicked_day not in self.painted_days:
            # Add this day to painted days
            self.painted_days.add(clicked_day)

            # Apply action to this day
            self.apply_action_to_day(clicked_day, self.selected_action)

    def handle_slider_drag(self, mouse_pos):
        """Handle slider dragging to adjust RGB values"""
        if not self.dragging_slider or not hasattr(self, 'slider_rects'):
            return

        # Find which slider is being dragged
        slider_index = ['r', 'g', 'b'].index(self.dragging_slider)
        slider_rect = self.slider_rects[slider_index]

        # Calculate new value based on mouse position
        relative_x = mouse_pos[0] - slider_rect.x
        new_value = max(0.0, min(1.0, relative_x / slider_rect.width))

        # Update the corresponding slider
        if self.dragging_slider == 'r':
            self.slider_r = new_value
        elif self.dragging_slider == 'g':
            self.slider_g = new_value
        elif self.dragging_slider == 'b':
            self.slider_b = new_value

    def apply_action_to_day(self, day, action):
        """Apply an action to a specific day"""
        if day == self.game_state.current_day:
            # Set action for current day (with 0 days planning ahead)
            self.game_state.current_day_action = action
            self.game_state.current_action_days_planned = 0
            print(f"Set action for today: {action.value}")
        elif day > self.game_state.current_day:
            # Remove any existing action for this future day
            self.game_state.planned_actions = [
                act for act in self.game_state.planned_actions
                if act.day_scheduled != day
            ]

            # Calculate energy bonus for planning ahead
            days_ahead = day - self.game_state.current_day
            energy_bonus = self.calculate_energy_bonus(days_ahead)

            # Add new planned action with days_ahead stored
            planned_action = PlannedAction(
                action_type=action,
                day_scheduled=day,
                day_planned=self.game_state.current_day,
                days_ahead=days_ahead,
                energy_bonus=energy_bonus
            )
            self.game_state.planned_actions.append(planned_action)

    def handle_kbd_selection_move(self, delta, shift_held):
        """Move or extend/shrink the keyboard selection

        Args:
            delta: Change in days (positive = right/down, negative = left/up)
            shift_held: True if shift is held (extend/shrink mode), False for move mode
        """
        calendar_start = self.calendar.start_day
        calendar_end = calendar_start + 34  # 5 weeks = 35 days

        if shift_held:
            if delta > 0:
                # Positive delta (Right/Down): GROW selection
                # Add days at delta offset from each currently selected day
                new_days = set()
                for day in self.kbd_selected_days:
                    new_day = day + delta
                    # Only add if within visible calendar range
                    if calendar_start <= new_day <= calendar_end:
                        new_days.add(new_day)
                # Only update if we actually added new days within range
                if new_days:
                    self.kbd_selected_days.update(new_days)
            else:
                # Negative delta (Left/Up): SHRINK selection
                # Remove days that are at -delta offset from any selected day
                days_to_remove = set()
                for day in self.kbd_selected_days:
                    check_day = day + delta  # This is negative, so we're checking backwards
                    # If a day exists delta positions before this day, remove this day
                    if check_day in self.kbd_selected_days:
                        days_to_remove.add(day)
                # Remove the days (keep at least one day selected)
                if len(self.kbd_selected_days) > len(days_to_remove):
                    self.kbd_selected_days -= days_to_remove
        else:
            # Move mode: check if entire selection would stay within visible range
            new_selection = set()
            all_in_range = True
            for day in self.kbd_selected_days:
                new_day = day + delta
                new_selection.add(new_day)
                # Check if this day would be out of range
                if new_day < calendar_start or new_day > calendar_end:
                    all_in_range = False

            # Only apply the move if entire selection stays within visible range
            if all_in_range:
                self.kbd_selected_days = new_selection

    def handle_kbd_selection_assign(self):
        """Assign the selected action to all days in the keyboard selection"""
        if self.selected_action is None:
            print("Select an action first (1-5 keys)")
            return

        # Apply action to all selected days
        for day in self.kbd_selected_days:
            if day >= self.game_state.current_day:
                self.apply_action_to_day(day, self.selected_action)

        num_days = len(self.kbd_selected_days)
        print(f"Applied {self.selected_action.value} to {num_days} day(s)")

    def draw_main_menu(self):
        """Draw the main menu"""
        # Green grass-like background that works well with brown signs
        self.screen.fill((120, 180, 100))

        # Title
        title = self.title_font.render("The prediction game", True, (80, 50, 20))  # Dark brown
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)

        # Menu options using sign graphics
        sign_images = [self.play_sign, self.highscores_sign]
        y_start = 350

        for i, sign in enumerate(sign_images):
            sign_rect = sign.get_rect(center=(WINDOW_WIDTH // 2, y_start + i * 150))

            # Draw a subtle highlight around the selected option
            if i == self.selected_menu_item:
                highlight_rect = pygame.Rect(
                    sign_rect.x - 10, sign_rect.y - 10,
                    sign_rect.width + 20, sign_rect.height + 20
                )
                pygame.draw.rect(self.screen, (255, 255, 150), highlight_rect, 5, border_radius=10)

            self.screen.blit(sign, sign_rect)

    def draw_play_menu(self):
        """Draw the play menu"""
        # Green grass-like background that works well with brown signs
        self.screen.fill((120, 180, 100))

        # Title
        title = self.title_font.render("Select Mode", True, (80, 50, 20))  # Dark brown
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)

        # Menu options using sign graphics
        sign_images = [self.tutorial_sign, self.survival_sign]
        y_start = 350

        for i, sign in enumerate(sign_images):
            sign_rect = sign.get_rect(center=(WINDOW_WIDTH // 2, y_start + i * 150))

            # Draw a subtle highlight around the selected option
            if i == self.selected_menu_item:
                highlight_rect = pygame.Rect(
                    sign_rect.x - 10, sign_rect.y - 10,
                    sign_rect.width + 20, sign_rect.height + 20
                )
                pygame.draw.rect(self.screen, (255, 255, 150), highlight_rect, 5, border_radius=10)

            self.screen.blit(sign, sign_rect)

    def draw_tutorial(self):
        """Draw interactive tutorial with highlighted screen regions"""
        # Draw the full game interface in the background
        # Draw in correct order: left panel, game world, control panel, then top bar on top
        self.draw_left_panel()
        self.draw_game_world()
        self.draw_control_panel()
        self.draw_top_bar()

        # Get current tutorial step
        step = self.tutorial_steps[self.tutorial_step]
        highlight_rect = step["highlight_rect"]

        # Draw the overlay in pieces, excluding the highlighted region
        # Top section (above highlight)
        if highlight_rect.top > 0:
            top_overlay = pygame.Surface((WINDOW_WIDTH, highlight_rect.top))
            top_overlay.set_alpha(200)
            top_overlay.fill((0, 0, 0))
            self.screen.blit(top_overlay, (0, 0))

        # Bottom section (below highlight)
        if highlight_rect.bottom < WINDOW_HEIGHT:
            bottom_height = WINDOW_HEIGHT - highlight_rect.bottom
            bottom_overlay = pygame.Surface((WINDOW_WIDTH, bottom_height))
            bottom_overlay.set_alpha(200)
            bottom_overlay.fill((0, 0, 0))
            self.screen.blit(bottom_overlay, (0, highlight_rect.bottom))

        # Left section (left of highlight, within highlight's vertical range)
        if highlight_rect.left > 0:
            left_overlay = pygame.Surface((highlight_rect.left, highlight_rect.height))
            left_overlay.set_alpha(200)
            left_overlay.fill((0, 0, 0))
            self.screen.blit(left_overlay, (0, highlight_rect.top))

        # Right section (right of highlight, within highlight's vertical range)
        if highlight_rect.right < WINDOW_WIDTH:
            right_width = WINDOW_WIDTH - highlight_rect.right
            right_overlay = pygame.Surface((right_width, highlight_rect.height))
            right_overlay.set_alpha(200)
            right_overlay.fill((0, 0, 0))
            self.screen.blit(right_overlay, (highlight_rect.right, highlight_rect.top))

        # Draw border around highlighted area
        pygame.draw.rect(self.screen, (255, 255, 100), highlight_rect, 3)  # Yellow border

        # Draw explanatory text box
        text_x, text_y = step["text_position"]
        box_width = 700
        box_height = 70 + len(step["text"]) * 30  # Extended by 30px
        box_x = text_x - box_width // 2
        box_y = text_y - 40

        # Text background box
        text_box = pygame.Surface((box_width, box_height))
        text_box.set_alpha(230)
        text_box.fill((40, 40, 40))
        self.screen.blit(text_box, (box_x, box_y))
        pygame.draw.rect(self.screen, (255, 255, 100), (box_x, box_y, box_width, box_height), 2)

        # Title
        title = self.font.render(step["title"], True, (255, 255, 100))
        title_rect = title.get_rect(center=(text_x, text_y))
        self.screen.blit(title, title_rect)

        # Text lines
        line_y = text_y + 35
        for line in step["text"]:
            text_surface = self.small_font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(text_x, line_y))
            self.screen.blit(text_surface, text_rect)
            line_y += 30

        # Navigation hint at bottom
        nav_text = f"Step {self.tutorial_step + 1}/{len(self.tutorial_steps)} - Use LEFT/RIGHT arrows to navigate, ESC to exit"
        nav_surface = self.small_font.render(nav_text, True, (255, 255, 100))
        nav_rect = nav_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
        # Background for navigation text
        nav_bg = pygame.Surface((nav_surface.get_width() + 20, nav_surface.get_height() + 10))
        nav_bg.set_alpha(200)
        nav_bg.fill((0, 0, 0))
        self.screen.blit(nav_bg, (nav_rect.x - 10, nav_rect.y - 5))
        self.screen.blit(nav_surface, nav_rect)

    def draw_highscores_menu(self):
        """Draw the highscores menu"""
        # Green grass-like background
        self.screen.fill((120, 180, 100))

        # Title
        title = self.title_font.render("Highscores", True, (80, 50, 20))  # Dark brown
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)

        # Load scores
        scores = self.load_scores()

        if not scores:
            # No scores yet
            no_scores_text = self.font.render("No scores yet! Play to set a record.", True, (80, 50, 20))
            no_scores_rect = no_scores_text.get_rect(center=(WINDOW_WIDTH // 2, 300))
            self.screen.blit(no_scores_text, no_scores_rect)
        else:
            # Display scores
            y_start = 200
            for i, score in enumerate(scores[:10]):  # Top 10
                rank_text = f"{i+1}."
                days_text = f"{score['days']} days"
                date_text = score['date']

                # Rank
                rank_surface = self.font.render(rank_text, True, (80, 50, 20))
                self.screen.blit(rank_surface, (WINDOW_WIDTH // 2 - 300, y_start + i * 50))

                # Days survived
                days_surface = self.font.render(days_text, True, (80, 50, 20))
                self.screen.blit(days_surface, (WINDOW_WIDTH // 2 - 200, y_start + i * 50))

                # Date
                date_surface = self.small_font.render(date_text, True, (100, 70, 40))
                self.screen.blit(date_surface, (WINDOW_WIDTH // 2 + 50, y_start + i * 50))

        # Back button
        back_text = self.font.render("Press ESC to return to menu", True, (80, 50, 20))
        back_rect = back_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
        self.screen.blit(back_text, back_rect)

    def draw_game_over_overlay(self):
        """Draw the game over overlay with score and options"""
        # Semi-transparent dark overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((40, 20, 10))  # Dark brown
        self.screen.blit(overlay, (0, 0))

        # Game Over title
        game_over_text = self.title_font.render("GAME OVER", True, (220, 80, 60))  # Red-ish
        game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(game_over_text, game_over_rect)

        # Score display
        score_text = self.font.render(f"You survived {self.final_score} days", True, (255, 220, 150))
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 300))
        self.screen.blit(score_text, score_rect)

        # Menu options
        menu_options = ["Return to Main Menu", "Quit Game"]
        y_start = 450

        for i, option in enumerate(menu_options):
            # Create a background rectangle for each option
            option_text = self.font.render(option, True, (255, 255, 255))
            option_rect = option_text.get_rect(center=(WINDOW_WIDTH // 2, y_start + i * 80))

            # Background box
            bg_rect = pygame.Rect(
                option_rect.x - 20, option_rect.y - 10,
                option_rect.width + 40, option_rect.height + 20
            )

            # Highlight selected option
            if i == self.selected_menu_item:
                pygame.draw.rect(self.screen, (180, 140, 80), bg_rect, border_radius=10)
                pygame.draw.rect(self.screen, (255, 220, 100), bg_rect, 3, border_radius=10)
            else:
                pygame.draw.rect(self.screen, (80, 60, 40), bg_rect, border_radius=10)

            self.screen.blit(option_text, option_rect)

    def handle_menu_input(self, event):
        """Handle menu navigation and selection"""
        if event.type == pygame.KEYDOWN:
            if self.current_menu == "main":
                menu_items = 2  # Play, Highscores
            elif self.current_menu == "play":
                menu_items = 2  # Tutorial, Survival
            elif self.game_over:
                menu_items = 2  # Return to Main Menu, Quit Game
            else:
                return

            if event.key == pygame.K_UP:
                self.selected_menu_item = (self.selected_menu_item - 1) % menu_items
            elif event.key == pygame.K_DOWN:
                self.selected_menu_item = (self.selected_menu_item + 1) % menu_items
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                if self.game_over:
                    self.handle_game_over_selection()
                else:
                    self.select_menu_item()

    def handle_game_over_selection(self):
        """Handle game over menu selection"""
        if self.selected_menu_item == 0:  # Return to Main Menu
            self.game_over = False
            self.current_menu = "main"
            self.selected_menu_item = 0
            # Reset all game state for next playthrough
            self.game_state = GameState()
            self.weather_system = WeatherSystem()
            self.selected_action = None
            self.planning_day = 1
            self.is_painting = False
            self.paint_start_day = None
            self.painted_days = set()
            self.kbd_selected_days = {1}
            self.calendar = Calendar(start_day=1, weeks_to_show=5)
            self.notes_calendar = NotesCalendar(start_day=1, weeks_to_show=5)
            self.slider_r = 0.5
            self.slider_g = 0.5
            self.slider_b = 0.5
            self.dragging_slider = None
            self.selected_notes_day = 1
            # Reset wind transformation state
            self.forest_m = self.forest_m_base
            self.forest_l = self.forest_l_base
            self.forest_r = self.forest_r_base
            self.last_wind_speed = 0.0
        elif self.selected_menu_item == 1:  # Quit Game
            self.running = False

    def select_menu_item(self):
        """Handle menu item selection"""
        if self.current_menu == "main":
            if self.selected_menu_item == 0:  # Play
                self.current_menu = "play"
                self.selected_menu_item = 0
            elif self.selected_menu_item == 1:  # Highscores
                self.current_menu = "highscores"
                self.selected_menu_item = 0
        elif self.current_menu == "play":
            if self.selected_menu_item == 0:  # Tutorial
                self.current_menu = "tutorial"
                self.tutorial_slide = 0
                self.selected_menu_item = 0
            elif self.selected_menu_item == 1:  # Survival
                self.current_menu = "game"
                self.selected_menu_item = 0

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()

            if self.current_menu == "game":
                self.draw_ui()
            elif self.current_menu == "main":
                self.draw_main_menu()
            elif self.current_menu == "play":
                self.draw_play_menu()
            elif self.current_menu == "tutorial":
                self.draw_tutorial()
            elif self.current_menu == "highscores":
                self.draw_highscores_menu()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

def main():
    """Entry point for the game"""
    print("Starting The prediction game...")
    print("Use number keys to select actions, arrow keys to plan ahead,")
    print("SPACE to execute immediately, ENTER to plan for future, N to advance day")

    game = Game()
    game.run()

if __name__ == "__main__":
    main()