import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------- persistent state ----------

_fig = None
_ax = None
_bg_drawn = False
_agent_dot = None
_path_line = None
_title_obj = None
_last_grid_id = None


def _draw_background(ax, grid, goal):
    """Draw the static parts once: city image, grid lines, goal, legend."""
    global _bg_drawn, _last_grid_id

    size = grid.shape[0]

    display = np.ones((size, size, 3))
    for r in range(size):
        for c in range(size):
            if grid[r, c] == 1:
                display[r, c] = [0.22, 0.22, 0.26]
            else:
                display[r, c] = [0.93, 0.93, 0.90]

    ax.imshow(display, origin='upper')

    for i in range(size + 1):
        ax.axhline(i - 0.5, color='#555555', linewidth=0.3)
        ax.axvline(i - 0.5, color='#555555', linewidth=0.3)

    ax.add_patch(mpatches.FancyBboxPatch(
        (goal[1] - 0.4, goal[0] - 0.4), 0.8, 0.8,
        boxstyle='round,pad=0.05', facecolor='#2ecc71',
        edgecolor='black', linewidth=1.5, zorder=3))
    ax.text(goal[1], goal[0], '★', ha='center', va='center',
            fontsize=14, color='white', fontweight='bold', zorder=4)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(labelsize=6)

    legend_elements = [
        mpatches.Patch(facecolor='#393940', label='Wall'),
        mpatches.Patch(facecolor='#edede6', label='Road'),
        mpatches.Patch(facecolor='#2ecc71', label='Goal ★'),
        mpatches.Patch(facecolor='#e74c3c', label='Agent'),
        mpatches.Patch(facecolor='#3498db', label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    _bg_drawn = True
    _last_grid_id = id(grid)


def show_grid(grid, goal, agent_pos=None, path=None, title='City Grid', pause=0.02):
    """
    Live-update the matplotlib window without any flashing.
    Uses draw + flush_events + time.sleep instead of plt.pause().
    """
    global _fig, _ax, _bg_drawn, _agent_dot, _path_line, _title_obj, _last_grid_id

    # --- create / recover figure ---
    if _fig is None or not plt.fignum_exists(_fig.number):
        plt.ion()
        _fig, _ax = plt.subplots(figsize=(7, 7))
        _fig.canvas.draw()          # initial full draw
        _fig.show()
        _bg_drawn = False
        _agent_dot = None
        _path_line = None
        _title_obj = None

    # --- draw static background if needed ---
    if not _bg_drawn or _last_grid_id != id(grid):
        _ax.clear()
        _agent_dot = None
        _path_line = None
        _title_obj = None
        _draw_background(_ax, grid, goal)
        _fig.canvas.draw()           # full draw for background

    # --- update path line ---
    if _path_line is not None:
        _path_line.remove()
        _path_line = None

    if path and len(path) > 1:
        rows, cols = zip(*path)
        _path_line, = _ax.plot(cols, rows, '-', color='#3498db',
                               linewidth=2, alpha=0.6, zorder=2)

    # --- update agent dot ---
    if _agent_dot is not None:
        _agent_dot.remove()
        _agent_dot = None

    if agent_pos is not None:
        _agent_dot, = _ax.plot(agent_pos[1], agent_pos[0], 'o', color='#e74c3c',
                               markersize=12, markeredgecolor='black',
                               markeredgewidth=1.5, zorder=5)

    # --- update title ---
    if _title_obj is not None:
        _title_obj.set_text(title)
    else:
        _title_obj = _ax.set_title(title, fontsize=13, fontweight='bold')

    # --- flush WITHOUT plt.pause (that causes the flash) ---
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()
    time.sleep(pause)


def reset_view():
    """Call between episodes to clear agent/path for the next run."""
    global _agent_dot, _path_line
    if _agent_dot is not None:
        _agent_dot.remove()
        _agent_dot = None
    if _path_line is not None:
        _path_line.remove()
        _path_line = None