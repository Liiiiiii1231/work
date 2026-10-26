"""Entry for the final tradeoff experiment."""
from pathlib import Path
import runpy,sys
support=Path(__file__).resolve().parent.parent/'tradeoff_support_plainfp_o4_i5_s8005'
sys.path.insert(0,str(support))
runpy.run_path(str(support/'run_tradeoff_final.py'),run_name='__main__')
