"""Regenerate committed PNGs; Matplotlib is a development-only dependency."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/nswe-matplotlib')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from equation_guide import GUIDE_ROWS
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral'})
output=Path(__file__).resolve().parents[1]/'assets'/'equations'
output.mkdir(parents=True,exist_ok=True)
for key,title,equation,explanation in GUIDE_ROWS:
    fig=plt.figure(figsize=(5.1,.85),dpi=200)
    label=fig.text(.02,.5,'$'+equation+'$',ha='left',va='center',fontsize=21,color='#19384a')
    fig.canvas.draw()
    width=label.get_window_extent(fig.canvas.get_renderer()).width
    if width>980:label.set_fontsize(21*980/width)
    fig.savefig(output/(key+'.png'),transparent=True,dpi=200)
    plt.close(fig)
print(f'Rendered {len(GUIDE_ROWS)} equation images')
