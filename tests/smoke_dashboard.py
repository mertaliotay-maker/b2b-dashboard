"""Optional UI regression: python tests/smoke_dashboard.py from project root.

Reads the real Excel with the shared loader once; reuses that exact DataFrame
across reruns. This is a local Streamlit test, not a browser/cloud deployment test.
"""
from pathlib import Path
import os
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import data_loader


def main():
    clean = data_loader.load_data()
    original_loader = data_loader.load_data
    data_loader.load_data = lambda: clean
    records = []
    app = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=240)

    def check(label):
        exceptions = [x.value for x in app.exception]
        records.append({'check':label, 'exceptions':exceptions,
                        'plot_count':len(app.get('plotly_chart')),
                        'warnings':[x.value for x in app.warning]})
        if exceptions:
            raise AssertionError(f'{label}: {exceptions}')
        print(f'{label}: OK',flush=True)

    try:
        app.run()
        check('Overview default')
        for name in ['Sales Analysis','Customers & Marketing','Supply Chain & Forecast']:
            app.sidebar.radio[0].set_value(name).run()
            check(name)
        for months in [6,12,3]:
            app.selectbox(key='fs_horizon').set_value(months).run()
            check(f'Forecast {months}m')
        app.selectbox(key='fs_target').set_value('Net revenue (TL)').run()
        check('Revenue target')
        app.selectbox(key='fs_service').set_value(99).run()
        check('99% service level')
        app.number_input(key='fs_lead_std').set_value(3.).run()
        check('Variable lead time')
        app.selectbox(key='fs_lookback').set_value(366).run()
        check('Full-history inventory')
        app.multiselect(key='fs_classes').set_value(['AZ']).run()
        check('AZ stock filter')
        app.sidebar.date_input[0].set_value((pd.Timestamp('2020-12-01').date(),pd.Timestamp('2020-12-31').date())).run()
        check('Short date range')
        assert any('at least 40' in x.value for x in app.warning)
        app.sidebar.date_input[0].set_value((pd.Timestamp('2020-12-31').date(),pd.Timestamp('2020-12-31').date())).run()
        check('Single-day range')
        app.sidebar.multiselect[0].set_value(['1000 Parça Puzzle']).run()
        check('Category plus single-day filter')
    finally:
        data_loader.load_data = original_loader
    report = {'mode':'Local Streamlit AppTest, shared real Excel loader called once, same result reused for UI reruns.', 'checks':records}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/streamlit_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')


if __name__ == '__main__':
    main()
