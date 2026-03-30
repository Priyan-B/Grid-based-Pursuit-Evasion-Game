"""
Matplotlib-based live visualisation for Stage 3 grid world.
Adds police cars and CCTV cameras on top of stage 2 rendering.

Usage from training:
    from utils.visualize_stage3 import show_grid_s3, reset_view_s3
"""

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
_trap_patches = []
_traffic_patches = []
_police_patches = []
_cctv_patches = []


def _draw_background(ax, grid, goal, cctv_cells=None):
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

    # CCTV cells — drawn as part of background (they don't move)
    if cctv_cells:
        for (cr, cc) in cctv_cells:
            rect = mpatches.FancyBboxPatch(
                (cc - 0.45, cr - 0.45), 0.9, 0.9,
                boxstyle='round,pad=0.02', facecolor='#9b59b6',
                edgecolor='black', linewidth=0.8, alpha=0.35, zorder=2)
            ax.add_patch(rect)
            ax.text(cc, cr, '\u25A3', ha='center', va='center',
                    fontsize=9, color='#6c3483', zorder=2)

    # Goal
    ax.add_patch(mpatches.FancyBboxPatch(
        (goal[1] - 0.4, goal[0] - 0.4), 0.8, 0.8,
        boxstyle='round,pad=0.05', facecolor='#2ecc71',
        edgecolor='black', linewidth=1.5, zorder=3))
    ax.text(goal[1], goal[0], '\u2605', ha='center', va='center',
            fontsize=14, color='white', fontweight='bold', zorder=4)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(labelsize=6)

    legend_elements = [
        mpatches.Patch(facecolor='#393940', label='Wall'),
        mpatches.Patch(facecolor='#edede6', label='Road'),
        mpatches.Patch(facecolor='#2ecc71', label='Goal \u2605'),
        mpatches.Patch(facecolor='#e74c3c', label='Thief'),
        mpatches.Patch(facecolor='#3498db', label='Path'),
        mpatches.Patch(facecolor='#e67e22', label='Trap \u25C6'),
        mpatches.Patch(facecolor='#f1c40f', label='Traffic'),
        mpatches.Patch(facecolor='#2c3e50', label='Police'),
        mpatches.Patch(facecolor='#9b59b6', label='CCTV', alpha=0.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    global _bg_drawn, _last_grid_id
    _bg_drawn = True
    _last_grid_id = id(grid)


def _clear_dynamic(lst):
    for p in lst:
        p.remove()
    lst.clear()


def show_grid_s3(grid, goal, agent_pos=None, path=None,
                 traps=None, traffic_positions=None,
                 police_positions=None, cctv_cells=None,
                 title='Stage 3', pause=0.02):
    global _fig, _ax, _bg_drawn, _agent_dot, _path_line, _title_obj
    global _last_grid_id, _trap_patches, _traffic_patches
    global _police_patches, _cctv_patches

    if _fig is None or not plt.fignum_exists(_fig.number):
        plt.ion()
        _fig, _ax = plt.subplots(figsize=(7, 7))
        _fig.canvas.draw()
        _fig.show()
        _bg_drawn = False
        _agent_dot = None
        _path_line = None
        _title_obj = None
        _trap_patches = []
        _traffic_patches = []
        _police_patches = []
        _cctv_patches = []

    if not _bg_drawn or _last_grid_id != id(grid):
        _ax.clear()
        _agent_dot = None
        _path_line = None
        _title_obj = None
        _trap_patches = []
        _traffic_patches = []
        _police_patches = []
        _cctv_patches = []
        _draw_background(_ax, grid, goal, cctv_cells)
        _fig.canvas.draw()

    # ── traps ──
    _clear_dynamic(_trap_patches)
    if traps:
        for (tr, tc) in traps:
            p = _ax.plot(tc, tr, 'D', color='#e67e22', markersize=9,
                         markeredgecolor='black', markeredgewidth=1.0,
                         zorder=4)[0]
            _trap_patches.append(p)

    # ── traffic ──
    _clear_dynamic(_traffic_patches)
    if traffic_positions:
        for (tr, tc) in traffic_positions:
            rect = mpatches.FancyBboxPatch(
                (tc - 0.3, tr - 0.3), 0.6, 0.6,
                boxstyle='round,pad=0.05', facecolor='#f1c40f',
                edgecolor='black', linewidth=1.0, zorder=4)
            _ax.add_patch(rect)
            _traffic_patches.append(rect)

    # ── police ──
    _clear_dynamic(_police_patches)
    if police_positions:
        for (pr, pc) in police_positions:
            p = _ax.plot(pc, pr, 's', color='#2c3e50', markersize=11,
                         markeredgecolor='white', markeredgewidth=1.5,
                         zorder=6)[0]
            _police_patches.append(p)

    # ── path ──
    if _path_line is not None:
        _path_line.remove()
        _path_line = None
    if path and len(path) > 1:
        rows, cols = zip(*path)
        _path_line, = _ax.plot(cols, rows, '-', color='#3498db',
                               linewidth=2, alpha=0.6, zorder=2)

    # ── agent ──
    if _agent_dot is not None:
        _agent_dot.remove()
        _agent_dot = None
    if agent_pos is not None:
        _agent_dot, = _ax.plot(agent_pos[1], agent_pos[0], 'o',
                               color='#e74c3c', markersize=12,
                               markeredgecolor='black',
                               markeredgewidth=1.5, zorder=7)

    # ── title ──
    if _title_obj is not None:
        _title_obj.set_text(title)
    else:
        _title_obj = _ax.set_title(title, fontsize=13, fontweight='bold')

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()
    time.sleep(pause)


def save_frame(path="logs/stage3_frame.png", dpi=120):
    global _fig
    if _fig is not None:
        _fig.savefig(path, dpi=dpi, bbox_inches='tight')


def reset_view_s3():
    global _agent_dot, _path_line
    global _trap_patches, _traffic_patches, _police_patches
    if _agent_dot is not None:
        _agent_dot.remove()
        _agent_dot = None
    if _path_line is not None:
        _path_line.remove()
        _path_line = None
    _clear_dynamic(_trap_patches)
    _clear_dynamic(_traffic_patches)
    _clear_dynamic(_police_patches)