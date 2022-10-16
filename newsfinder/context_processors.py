import os
from datetime import datetime


def export_vars(request):
    data = {}
    data['TODAYS_DATE'] = datetime.strptime(os.environ['TODAYS_DATE'], "%Y-%m-%d").strftime("%B %d, %Y")
    return data
