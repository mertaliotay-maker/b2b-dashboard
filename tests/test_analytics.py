"""Meaningful checks for time leakage, zero-sale periods and inventory math."""
import unittest
import numpy as np
import pandas as pd
from statistics import NormalDist
from analytics.forecasting import weekly_series, metrics, evaluate_models, forecast_model, prediction_band
from analytics.inventory import inventory_table


def transactions():
    dates = pd.date_range('2020-01-06', periods=42, freq='D')
    return pd.DataFrame({'StockCode': ['A']*42, 'Description': ['Book']*42,
                         'Category':['Books']*42, 'InvoiceDate':dates,
                         'Quantity': [1.]*42, 'Revenue': [10.]*42})


class AnalyticsTests(unittest.TestCase):
    def test_zero_actual_mape_and_wape(self):
        m = metrics([0, 10], [5, 8])
        self.assertEqual(m['MAPE_n'], 1)
        self.assertAlmostEqual(m['MAPE_pct'], 20.)
        self.assertAlmostEqual(m['WAPE_pct'], 70.)

    def test_boundary_weeks_and_interior_zero_week(self):
        df = transactions()
        df = df.loc[~df.InvoiceDate.between('2020-01-13','2020-01-19')]
        y, meta = weekly_series(df)
        self.assertEqual(len(y), 6)
        self.assertEqual(y.iloc[1], 0.)
        self.assertEqual(y.sum(), df.Quantity.sum())
        short = df.loc[df.InvoiceDate.between('2020-01-07','2020-01-10')]
        with self.assertRaises(ValueError):
            weekly_series(short)

    def test_holdout_cannot_change_selection_or_interval_calibration(self):
        idx = pd.date_range('2020-01-05', periods=51, freq='W-SUN')
        y = pd.Series(100 + np.arange(51)*.4 + np.sin(np.arange(51))*5, index=idx)
        first = evaluate_models(y)
        changed = y.copy()
        changed.iloc[-8:] = changed.iloc[-8:]*100
        second = evaluate_models(changed)
        self.assertEqual(first['winner'], second['winner'])
        pd.testing.assert_frame_equal(first['cv'], second['cv'])
        self.assertEqual(first['radius'], second['radius'])
        self.assertNotEqual(first['test_scores'].RMSE.sum(), second['test_scores'].RMSE.sum())

    def test_recursive_rf_reproducible_and_nonnegative(self):
        idx = pd.date_range('2020-01-05', periods=30, freq='W-SUN')
        y = pd.Series(np.arange(30)+1., index=idx)
        a, b = forecast_model('Random Forest', y, 12), forecast_model('Random Forest', y.copy(), 12)
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a >= 0).all())

    def test_zero_days_and_one_sku_abc(self):
        df = transactions().iloc[[0, 41]].copy()
        df.Quantity = [2., 4.]
        df.Revenue = [20., 40.]
        out, meta = inventory_table(df, lookback_days=90)
        row = out.iloc[0]
        dense = np.zeros(42)
        dense[[0, 41]] = [2., 4.]
        self.assertAlmostEqual(row.MeanDaily, dense.mean())
        self.assertAlmostEqual(row.StdDaily, dense.std(ddof=1))
        self.assertEqual(row.ABC, 'A')
        expected = NormalDist().inv_cdf(.95) * dense.std(ddof=1) * np.sqrt(14)
        self.assertAlmostEqual(row.SafetyStock, expected)
        self.assertEqual(row.ReorderPoint, np.ceil(14*dense.mean()+expected))
        more, _ = inventory_table(df, lead_std_days=3)
        self.assertGreater(more.iloc[0].SafetyStock, row.SafetyStock)

    def test_stockcode_identity_and_weekly_zeros(self):
        df = transactions().iloc[[0, 41]].copy()
        df.Description = ['Old title','New title']
        out, _ = inventory_table(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0].Description, 'New title')
        weekly = np.array([1.,0,0,0,0,1.])
        self.assertAlmostEqual(out.iloc[0].CV, weekly.std(ddof=1)/weekly.mean())

    def test_one_day_has_no_fabricated_standard_deviation(self):
        out, _ = inventory_table(transactions().iloc[[0]])
        self.assertTrue(pd.isna(out.iloc[0].StdDaily))
        self.assertTrue(pd.isna(out.iloc[0].ReorderPoint))
        self.assertEqual(out.iloc[0].XYZ, 'Unclassified')

    def test_short_history_rejected_for_model_comparison(self):
        with self.assertRaises(ValueError):
            evaluate_models(pd.Series(np.ones(12), index=pd.date_range('2020-01-05',periods=12,freq='W-SUN')))

    def test_band_contains_point_and_widens(self):
        band = prediction_band([100.,100.,100.], 10.)
        self.assertTrue((band[:,0] <= 100).all() and (band[:,1] >= 100).all())
        self.assertTrue(np.all(np.diff(band[:,1]-band[:,0]) > 0))


if __name__ == '__main__':
    unittest.main()
