#!"c:\users\cgx\onedrive - teesside university\desktop\mlprojectend2end\crop_recom\scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'kaggle==1.5.13','console_scripts','kaggle'
__requires__ = 'kaggle==1.5.13'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('kaggle==1.5.13', 'console_scripts', 'kaggle')()
    )
