"""Final system_load: shared frozen initial geometry, independent resource initialization."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
SUPPORT=HERE.parent/'final_support_plainfp_o4_i5_s8005'
sys.path.insert(0,str(SUPPORT))
from run_final import main
if __name__=='__main__':
    args=sys.argv[1:]
    if any(a=='--figure' or a.startswith('--figure=') for a in args):raise SystemExit('This entry is dedicated to system_load')
    sys.argv=[str(SUPPORT/'run_final.py'),'--figure','system_load','--output',str(HERE/'results'/'manual_plainfp_o4_i5'),*args]
    main()
