import pandas as pd
import os
from django.conf import settings
BASE_DIR=settings.BASE_DIR
print BASE_DIR
print "Here im in ttt"
df = pd.read_csv(os.path.join(BASE_DIR,'save2csv.csv'))
print df