from ast import Tuple
import colorsys
import random
import sys
import threading
import time
from math import floor
from typing import List
import numpy as np

import pygame
from button import Button

from cell import Cell
from checkbox import Checkbox
from client import Client
from globals import (BG_COLOR, BLACK, CELL_EDGE, COOL, PAUSE_IS_PRESSED, DEAD, DISP,
                     DISP_H, DISP_W, FONT_SIZE, FONTS, GEAR, HOURGLASS, IMAGES, LOSE,
                     LRB_BORDER, MINE, PAUSE_FONT_SIZE, PAUSED, PLAYING,
                     PAUSE_WAS_PRESSED, RESET, SETTING_BUTTON_PRESSED, SETTINGS, SHOCKED, SMILE,
                     TOP_BORDER)
from grid import Grid
from screens import Screen
from solver import Solver, pct
from sprites import SPRITES
from textbox import Textbox

pygame.init()


class Game(Client):

    font = pygame.font.Font(FONTS + "Font.ttf", FONT_SIZE)
    pause_font = pygame.font.Font(FONTS + "Font.ttf", PAUSE_FONT_SIZE)
    pause_label_font = pygame.font.Font(FONTS + "Font.ttf", int(CELL_EDGE * 0.75))
    prob_font = pygame.font.Font(FONTS + "Font.ttf", int(CELL_EDGE * 0.42))
    prob_font_small = pygame.font.Font(FONTS + "Font.ttf", int(CELL_EDGE * 0.34))

    # AUTO SOLVER pacing (seconds): one certain move, showing the probabilities
    # before acting on them, and highlighting a guess before clicking it.
    SOLVER_STEP = 0.15
    SOLVER_PROB = 0.6
    SOLVER_GUESS = 0.4

    def __init__(self) -> None:
        super().__init__()
        self.running = True
        self.elapsed_time = 0
        self.flagged_cells = 0
        self.displayed_screen = Screen.MAIN
        self.timerEvent = threading.Event()
        self.timer_thread = threading.Thread(target=self.timer)
        self.start_button = Button(DISP_W // 2, TOP_BORDER * 2, "START")
        self.bot_button = Button(DISP_W // 2, TOP_BORDER * 2 + 127, "AUTO SOLVER")
        self.quit_button = Button(DISP_W // 2, TOP_BORDER * 2 + 255, "QUIT")
        self.settings_btn = Cell(
            SETTINGS["width"] - 0.75, -2.75, value=GEAR, hidden=False)
        self.reset_btn = Cell(
            SETTINGS["width"] / 2 - 0.5, -2, value=RESET, hidden=False)
        self.textboxes: List[Textbox] = []
        self.checkboxes: List[Checkbox] = []
        for i, setting in enumerate(SETTINGS):
            setting_type = type(SETTINGS[setting])
            if setting_type is bool:
                self.checkboxes.append(Checkbox(setting, TOP_BORDER * 1.3 + (Textbox.box_height + 20) * i))
            else:
                self.textboxes.append(Textbox(setting, TOP_BORDER * 1.3 + (Textbox.box_height + 20) * i))
               
        self.grid = Grid()
        self.solving = False      # the solver thread is playing
        self.solver_board = False # the board belongs to an AUTO SOLVER run
        self.solver_status = ""
        self.probs = None         # (x, y) -> mine probability, drawn on hidden cells
        self.highlight = None     # (x, y) of the guess about to be clicked
        self._prob_labels = {}

    def timer(self):
        while self.running:
            while self.grid.enabled and self.grid.clicked_cell:
                self.timerEvent.wait()
                self.elapsed_time += 1
                time.sleep(1)

    def save_settings(self):
        """Tries to save the user altered settings
        """
        error = False
        changed = False
        for box in self.textboxes:
            setting = box.name.lower().replace(" ", "_")
            setting_type = type(SETTINGS[setting])
            if box.text != "":
                try:
                    new_setting = setting_type(box.text)
                    if new_setting != SETTINGS[setting]:
                        SETTINGS[setting] = new_setting
                        changed = True
                except:
                    box.text = "ERR"
                    error = True
            else:
                box.text = str(SETTINGS[setting])

        for box in self.checkboxes:
            setting = box.name.lower().replace(" ", "_")
            if not box.active == SETTINGS[setting]:
                SETTINGS[setting] = box.active
                changed = True

        if error:
            return
        if changed:
            super().set_settings()
            self.running = False
            return

    def reveal(self, x: int, y: int):
        """This method reveals the cell at the given coordinates if possible.

        Args:
            x (int): The x coordinate of the cell.
            y (int): The y coordinate of the cell.
        """
        global RESET

        if not self.grid.clicked_cell and SETTINGS["easy_start"]:
            self.grid.easy_start(x, y)

        self.grid.clicked_cell = self.grid.contents[y][x]
        mine_loc = self.grid.reveal_next(x, y)
        if mine_loc:
            RESET = DEAD
            self.reset_btn.value = DEAD
            self.grid.enabled = False
            return mine_loc

    def flag(self, x: int, y: int):
        """This method Flags/Unflags the cell at the given coordinates.

        Args:
            x (int): The x coordinate of the cell to flag.
            y (int): The y coordinate of the cell to
        """
        if self.grid.contents[y][x].hidden:
            if self.grid.contents[y][x].flagged:
                self.grid.contents[y][x].flagged = False
                self.flagged_cells -= 1
                self.grid.remove_flags(x, y)
            else:
                self.grid.contents[y][x].flagged = True
                self.flagged_cells += 1
                self.grid.add_flags(x, y)

        if self.flagged_cells == self.grid.mines:
            self.endgame_handler()

    def solve(self):
        """AUTO SOLVER: plays the whole game from this thread, one visible move
        at a time (see solver.py for how the moves are chosen)."""
        grid = self.grid
        solver = Solver(grid, SETTINGS["easy_start"])
        stats = {"moves": 0, "guesses": 0, "last_guess": None}
        self.solver_status = "Solver: starting"
        print("Solver moves:")

        def wait(seconds):
            """Sleeps, holding the schedule while paused. False = solver stopped."""
            end = time.monotonic() + seconds
            while True:
                if not self.solving or self.grid is not grid or not self.running:
                    return False
                if self.displayed_screen == Screen.GAME and grid.enabled:
                    if time.monotonic() >= end:
                        return True
                else:
                    end += 0.02
                time.sleep(0.02)

        def finish(summary, status):
            self.solving = False
            self.probs = None
            self.highlight = None
            self.solver_status = status
            print(summary)

        def ended():
            if self.reset_btn.value == COOL:
                g = stats["guesses"]
                finish("Won in %i moves (%i %s)" % (stats["moves"], g, "guess" if g == 1 else "guesses"), "Solver: won")
            elif self.reset_btn.value == DEAD:
                last = stats["last_guess"]
                finish("Lost on guess #%i (%s%% risk)" % (stats["guesses"], pct(last.p)) if last else "Lost",
                       "Solver: lost on a guess")
            elif grid.troll_mode:
                finish("Gave up: troll mode (too many mines for a safe start)", "Solver gives up: troll mode")
            else:
                return False
            return True

        while not ended():
            if not wait(0):
                return
            result = solver.next_moves()
            if not result.moves:
                finish("Stopped: no moves left", "Solver: stopped")
                return
            self.probs = result.probs
            if result.probs:
                self.solver_status = "Solver: reading the odds"
                if not wait(Game.SOLVER_PROB):
                    return
            for move in result.moves:
                if move.guess:
                    self.highlight = (move.x, move.y)
                    self.solver_status = "Solver: guessing (%s%% risk)" % pct(move.p)
                    if not wait(Game.SOLVER_GUESS):
                        return
                    self.highlight = None
                cell = grid.contents[move.y][move.x]
                if not cell.hidden or cell.flagged:
                    continue  # already opened by an earlier move
                stats["moves"] += 1
                where = "(%i, %i)" % (move.x + 1, move.y + 1)
                if move.guess:
                    stats["guesses"] += 1
                    stats["last_guess"] = move
                    print("#%i Guess %s - %s" % (stats["moves"], where, move.reason))
                else:
                    self.solver_status = "Solver: certain move"
                    print("#%i %s %s - %s" % (stats["moves"], "Flag" if move.type == "flag" else "Reveal", where, move.reason))
                if move.type == "flag":
                    self.flag(move.x, move.y)
                else:
                    self.reveal(move.x, move.y)
                if ended() or not wait(Game.SOLVER_STEP):
                    return
            self.probs = None

    def not_over(self):
        maxy, maxx = self.grid.contents.shape
        for x in range(maxx):
            for y in range(maxy):
                if not self.grid.contents[y][x].flagged and self.grid.contents[y][x].hidden:
                    return True
        return False
    

# ----------------------Event-Handlers---------------------- #
 # Cursor Events:

    def find_affected_cell(self, event: pygame.event.Event):
        """Reacts to player left click

        Args:
            game (Minesweeper.Game): The game to react to.
            event (pygame.event.Event, optional): The event associated with the left click.
        """
        global SETTING_BUTTON_PRESSED, RESET


        match (self.displayed_screen):
            case Screen.MAIN:
                if not self.start_button.clicked and self.start_button.collidepoint(event.pos):
                    self.displayed_screen = Screen.GAME
                    self.grid.enabled = True
                    self.start_button.clicked = True
                else:
                    self.start_button.clicked = False
                    if not self.quit_button.clicked and self.quit_button.collidepoint(event.pos):
                        self.running = False
                        self.quit_button.clicked = True
                    else:
                        self.quit_button.clicked = False
                        if not self.bot_button.clicked and self.bot_button.collidepoint(event.pos):
                            self.displayed_screen = Screen.GAME
                            self.bot_button.clicked = True
                            self.grid.enabled = True  # so the timer runs and Esc pauses the solver
                            self.solver_board = True
                            self.solving = True
                            threading.Thread(target=self.solve, daemon=True).start()


            case Screen.GAME:
                x = (event.pos[0] - LRB_BORDER) / CELL_EDGE
                y = (event.pos[1] - TOP_BORDER) / CELL_EDGE
                if self.grid.enabled and TOP_BORDER < event.pos[1] < DISP_H - LRB_BORDER and LRB_BORDER < event.pos[0] < DISP_W - LRB_BORDER:
                    if self.solving:
                        return  # the board is the solver's while it plays
                    x = floor(x)
                    y = floor(y)
                    actions = {
                        1: self.reveal,
                        3: self.flag,
                    }
                    try:
                        actions[event.button](x, y)
                    except:
                        pass

                elif self.settings_btn.collidepoint(event.pos[0], event.pos[1]):
                    if self.grid.troll_mode:
                        self.grid.troll()
                        for y in range(SETTINGS["height"]):
                            for x in range(SETTINGS["width"]):
                                self.grid.contents[y][x].hidden = False
                    else:
                        SETTING_BUTTON_PRESSED = True
                elif self.reset_btn.collidepoint(event.pos[0], event.pos[1]):
                    if self.grid.troll_mode:
                        SETTINGS["mines%"] = 15
                        super().set_settings()
                        self.running = False # TODO
                    else:
                        self.reset()

            case Screen.SETTINGS:
                for box in self.textboxes:
                    if box.collidepoint(event.pos):
                        box.active = True
                        box.text = ""
                    else:
                        box.active = False

                for box in self.checkboxes:
                    if box.collidepoint(event.pos):
                        box.active = not box.active

                if self.settings_btn.collidepoint(event.pos[0], event.pos[1]):
                    if self.grid.troll_mode:
                        self.grid.troll()
                        for y in range(SETTINGS["height"]):
                            for x in range(SETTINGS["width"]):
                                self.grid.contents[y][x].hidden = False
                    else:
                        SETTING_BUTTON_PRESSED = True


#-----#

    def highlight_cell(self, event: pygame.event.Event):
        """Reacts to cursor movement

        Args:
            game (Minesweeper.Game): The game to react in.
            event (pygame.event.Event): The event associated with the left click.
        """
        global RESET

        
        RESET = self.reset_btn.value
        if self.reset_btn.collidepoint(event.pos[0], event.pos[1]):
            RESET = SHOCKED

# --------------------------------------------- #
 # Draw events:

    def draw_panel(self):
        mines_left = Game.font.render(
            str(self.grid.mines - self.flagged_cells), 0, BLACK)
        time_elapsed = Game.font.render(str(self.elapsed_time), 0, BLACK)
        DISP.blits([(SPRITES[MINE], (LRB_BORDER, LRB_BORDER)),
                    (mines_left, mines_left.get_rect(
                        center=(LRB_BORDER + CELL_EDGE * 1.6, LRB_BORDER + CELL_EDGE / 1.75))),
                    (SPRITES[HOURGLASS], (LRB_BORDER, LRB_BORDER + CELL_EDGE)),
                    (time_elapsed, time_elapsed.get_rect(center=(LRB_BORDER + CELL_EDGE * 1.6, LRB_BORDER + CELL_EDGE * 1.6)))])
        DISP.blit(SPRITES[RESET], self.reset_btn)
        self.settings_btn.draw()

#-----#

    def draw_main(self):

        DISP.fill(BG_COLOR)
        match (self.displayed_screen):
            case Screen.MAIN:
                self.draw_main_menu()
            case Screen.GAME:
                self.draw_panel()
                self.grid.draw()
                if self.solver_board:
                    self.draw_solver_overlay()
            case Screen.SETTINGS:
                self.draw_panel()
                self.draw_settings()
            case Screen.PAUSE:
                # The board stays on screen, frozen, under the PAUSED label.
                self.draw_panel()
                self.grid.draw()
                if self.solver_board:
                    self.draw_solver_overlay()
                self.draw_pause_menu()




#-----#

    def draw_settings(self):
        text = Game.pause_font.render("SETTINGS", 0, BLACK)
        DISP.blit(text, text.get_rect(
            center=(DISP_W // 2, TOP_BORDER + SETTINGS["scale"] * 10)))
        for box in self.textboxes:
            box.draw()
        
        for box in self.checkboxes:
            box.draw()
 

#-----#

    def prob_label(self, text: str):
        """Rendered (outline, fill) surfaces for a probability label, cached."""
        if text not in self._prob_labels:
            p = 0.0 if text == "<1" else 1.0 if text == ">99" else int(text) / 100
            r, g, b = colorsys.hls_to_rgb((1 - p) / 3, 0.55, 0.95)
            font = Game.prob_font if len(text) <= 2 else Game.prob_font_small
            self._prob_labels[text] = (font.render(text, True, BLACK),
                                       font.render(text, True, (int(r * 255), int(g * 255), int(b * 255))))
        return self._prob_labels[text]

    def draw_solver_overlay(self):
        """Mine probabilities on the hidden cells, the next guess, and the status."""
        probs = self.probs
        if probs:
            o = max(1, int(SETTINGS["scale"] * 1.5))
            for (x, y), p in probs.items():
                cell = self.grid.contents[y][x]
                if not cell.hidden or cell.flagged:
                    continue
                outline, fill = self.prob_label(pct(p))
                cx, cy = cell.center
                for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o), (-o, -o), (o, o), (-o, o), (o, -o)):
                    DISP.blit(outline, outline.get_rect(center=(cx + dx, cy + dy)))
                DISP.blit(fill, fill.get_rect(center=(cx, cy)))
        if self.highlight:
            x, y = self.highlight
            cell = self.grid.contents[y][x]
            lw = max(2, int(CELL_EDGE / 12))
            pygame.draw.rect(DISP, BLACK, cell, lw + 2)
            pygame.draw.rect(DISP, (255, 212, 0), cell.inflate(-2, -2), lw)
        note = self.solver_status
        if note and self.displayed_screen != Screen.PAUSE:  # the PAUSED label goes there
            text = Game.font.render(note, 0, BLACK)
            if text.get_width() > DISP_W - 2 * LRB_BORDER:
                short = note.replace("Solver gives up: ", "").replace("Solver: ", "").replace(" risk)", "").replace("(", "")
                text = Game.font.render(short, 0, BLACK)
            DISP.blit(text, text.get_rect(center=(DISP_W // 2, self.status_y())))

    def status_y(self):
        """Vertical centre of the strip between the smiley and the grid."""
        return (self.reset_btn.bottom + TOP_BORDER) / 2

    def draw_pause_menu(self):
        # Drawn in the strip between the smiley and the grid (where the solver's
        # status goes), so the frozen board stays fully visible.
        text = Game.pause_label_font.render("PAUSED", 0, BLACK)
        DISP.blit(text, text.get_rect(center=(DISP_W // 2, self.status_y())))
        

    def draw_main_menu(self):
        self.start_button.draw()
        self.bot_button.draw()
        self.quit_button.draw()




# --------------------------------------------- #
  # Game Events:

    def pause_handler(self):

        global PAUSE_IS_PRESSED, PAUSE_WAS_PRESSED, SETTING_BUTTON_PRESSED

        keys = pygame.key.get_pressed()
        if RESET == SMILE and self.displayed_screen != Screen.MAIN:
            if keys[pygame.K_ESCAPE] or SETTING_BUTTON_PRESSED:
                if PAUSE_IS_PRESSED:
                    PAUSE_WAS_PRESSED = True
                PAUSE_IS_PRESSED = True
            else:
                PAUSE_WAS_PRESSED = False
                PAUSE_IS_PRESSED = False

            if PAUSE_IS_PRESSED and not PAUSE_WAS_PRESSED:
                self.grid.enabled = not self.grid.enabled
                if self.displayed_screen == Screen.GAME:
                    if SETTING_BUTTON_PRESSED:
                        self.displayed_screen = Screen.SETTINGS
                    else:
                        self.displayed_screen = Screen.PAUSE
                if self.timerEvent.is_set():
                    self.timerEvent.clear()
                elif keys[pygame.K_ESCAPE] and self.displayed_screen == Screen.PAUSE:
                    self.timerEvent.set()
                    self.displayed_screen = Screen.GAME
                elif SETTING_BUTTON_PRESSED and self.displayed_screen == Screen.SETTINGS:
                    self.timerEvent.set()
                    self.displayed_screen = Screen.GAME
                    self.save_settings()
        elif keys[pygame.K_ESCAPE] and RESET == SHOCKED:
            print("LOL")

        SETTING_BUTTON_PRESSED = False


#-----#

    def endgame_handler(self):

        global RESET

        b = False
        for list in self.grid.contents:
            for cell in list:
                if cell.value == MINE and not cell.flagged:
                    b = True
                    break

            if b:
                break

        else:
            RESET = COOL
            self.reset_btn.value = COOL
            self.grid.enabled = False
            for list in self.grid.contents:
                for cell in list:
                    if cell.value != MINE:
                        cell.hidden = False

#-----#

    def event_handler(self):
        
        global RESET

        for event in pygame.event.get():
            match event.type:
                case pygame.MOUSEBUTTONDOWN:
                    self.find_affected_cell(event)
                case pygame.MOUSEMOTION:
                    if self.timerEvent.is_set():
                        self.highlight_cell(event)
                case pygame.KEYDOWN:
                    if self.displayed_screen == Screen.SETTINGS:
                        for box in self.textboxes:
                            box.text_handler(event.key, event.unicode)
                case pygame.QUIT:
                    self.running = False

        pygame.display.update()

#-----#

    def reset(self):
        """Resets the Game
        """
        self.reset_btn.value = SMILE
        self.elapsed_time = 0
        self.flagged_cells = 0
        self.grid.clicked_cell = None
        self.grid.troll_mode = False
        self.solving = False
        self.solver_board = False
        self.solver_status = ""
        self.probs = None
        self.highlight = None
        self.grid = Grid()
        self.grid.enabled = True

# ----------------------- #

    def run(self):
        """Starts the game threads & game loop
        """
        self.running = True
        if not self.timer_thread.is_alive():
            self.timer_thread.start()
            self.timerEvent.set()

        # fps = 0
        # temp = self.elapsed_time
        while self.running:
            self.event_handler()
            self.draw_main()
            self.pause_handler()
            # fps += 1
            # if temp != self.elapsed_time:
            #     temp = self.elapsed_time
            #     print(fps)
            #     fps = 0

        pygame.quit()


