# =============================================================================
# SCRATCH CELL — five alternative styles for E1's distribution figure.
# Paste this into a throwaway cell in NB02, right after the existing E1
# generation cell (so `df`, `C`, `tiers`, `xlabs`, `pcts` already exist).
# Does NOT touch or replace the real Fig 02-A cell -- purely exploratory.
# Delete this cell once you've picked a style, or keep it as a design-notes
# scratchpad; either way it never gets called by save_fig() or anything
# downstream.
# =============================================================================
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
axes = axes.flatten()
colors = [C['green'], C['orange'], C['red']]

# -----------------------------------------------------------------------
# 1) Horizontal bar -- same information as your current chart, just
#    rotated. Cheapest possible change; instantly reads as "not a Results
#    bar chart" purely because Results bars are all vertical.
# -----------------------------------------------------------------------
ax = axes[0]
y = np.arange(len(tiers))
bars = ax.barh(y, pcts.values, 0.55, color=colors, alpha=0.88, edgecolor='white')
for bar, v in zip(bars, pcts.values):
    ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%',
            va='center', fontsize=12, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels([lab.split('\n')[0] for lab in xlabs], fontsize=11, fontweight='bold')
ax.set_xlim(0, max(pcts.values) * 1.25)
ax.set_xlabel('% of intervals', fontsize=12, fontweight='bold')
ax.set_title('1. Horizontal bar', fontsize=13, fontweight='bold')

# -----------------------------------------------------------------------
# 2) Donut -- best fit conceptually, since these percentages ARE parts of
#    one whole (they sum to 100%), which a bar chart never visually states
#    but a donut states automatically just by being a circle.
# -----------------------------------------------------------------------
ax = axes[1]
wedges, _, autotexts = ax.pie(
    pcts.values, colors=colors, autopct='%1.1f%%', pctdistance=0.78,
    startangle=90, wedgeprops=dict(width=0.42, edgecolor='white', linewidth=1.5),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
    at.set_color('white')
ax.legend(wedges, [lab.split('\n')[0] for lab in xlabs],
          loc='center', fontsize=10, frameon=False)
ax.set_title('2. Donut', fontsize=13, fontweight='bold')

# -----------------------------------------------------------------------
# 3) Single 100%-stacked horizontal bar -- one bar, segmented. Very
#    space-efficient (works well as a thin strip under a paragraph rather
#    than a full figure), and like the donut it visually states "this is a
#    complete partition" rather than "these are separate quantities".
# -----------------------------------------------------------------------
ax = axes[2]
left = 0
for lab, val, color in zip(xlabs, pcts.values, colors):
    ax.barh(0, val, left=left, color=color, edgecolor='white', height=0.6)
    if val > 6:  # skip label if the slice is too thin to hold text
        ax.text(left + val / 2, 0, f'{lab.split(chr(10))[0]}\n{val:.1f}%',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    left += val
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel('% of intervals', fontsize=12, fontweight='bold')
ax.set_title('3. 100% stacked bar', fontsize=13, fontweight='bold')

# -----------------------------------------------------------------------
# 4) Waffle / icon grid -- 100 squares, one per percentage point. Most
#    intuitive read for a non-specialist committee member ("out of 100
#    typical intervals, X are background..."), at the cost of being the
#    least conventional choice for an engineering thesis.
# -----------------------------------------------------------------------
ax = axes[3]
counts = np.round(pcts.values).astype(int)
counts[-1] = 100 - counts[:-1].sum()  # force exact total of 100 after rounding
grid_colors = []
for c_, n in zip(colors, counts):
    grid_colors += [c_] * max(n, 0)
n_cols = 10
for i, color in enumerate(grid_colors[:100]):
    row, col = divmod(i, n_cols)
    ax.add_patch(mpatches.Rectangle((col, 9 - row), 0.88, 0.88, color=color))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
legend_patches = [mpatches.Patch(color=c_, label=f'{lab.split(chr(10))[0]} ({v:.1f}%)')
                   for c_, lab, v in zip(colors, xlabs, pcts.values)]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          ncol=1, fontsize=9, frameon=False)
ax.set_title('4. Waffle (1 square = 1%)', fontsize=13, fontweight='bold')

# -----------------------------------------------------------------------
# 5) Lollipop -- minimalist stem-and-dot instead of solid bars. Reads as
#    "lighter weight, descriptive" rather than "here's a measured result",
#    which is a useful signal difference from Results' solid PDR bars.
# -----------------------------------------------------------------------
ax = axes[4]
x = np.arange(len(tiers))
ax.vlines(x, 0, pcts.values, color=colors, linewidth=2.5, alpha=0.85)
ax.scatter(x, pcts.values, color=colors, s=220, zorder=5, edgecolor='white', linewidth=1.5)
for xi, v in zip(x, pcts.values):
    ax.text(xi, v + 2, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([lab.split('\n')[0] for lab in xlabs], fontsize=11, fontweight='bold')
ax.set_ylabel('% of intervals', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(pcts.values) * 1.3)
ax.set_title('5. Lollipop', fontsize=13, fontweight='bold')

axes[5].axis('off')  # spare panel

fig.suptitle('E1: CO\u2082 Tier Distribution \u2014 Five Alternatives (scratch comparison)',
             fontsize=15, fontweight='bold', y=1.0)
fig.tight_layout()
plt.show()  # deliberately NOT save_fig() -- this is a comparison scratchpad, not a thesis figure
