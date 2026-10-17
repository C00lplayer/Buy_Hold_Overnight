# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion


class OvernightDipBuyAlgorithm(QCAlgorithm):
    """
    SPY overnight "buy the dip" strategy (2010 to present).

    Base trade (same execution skeleton as the earlier overnight runs):
      - Long SPY overnight only: buy ~1 minute before the close, sell right
        after the next session open. Flat during the trading day.

    Dip definition (decided once per day, ~15:59 ET, from ONE batch of daily
    bars that ends on the PREVIOUS session - no look-ahead):
      1. Trend filter (long-term MA): previous close > 200-day SMA, i.e. only
         buy pullbacks while the primary trend is up.
      2. Pullback trigger (short-term MA): previous close < 5-day EMA, i.e.
         price has slipped below its very short-term average.
      3. Negative closing flow: the previous session was a down close
         (close < prior close), i.e. the pullback is actively in progress.

    All three must hold on the decision date to buy that night; otherwise the
    account stays in cash. Sizing: 100% of portfolio value per dip night.
    """

    _TICKER = "SPY"
    _LOOKBACK_DAYS = 400     # calendar days of daily bars fetched per decision
    _MIN_BARS = 210          # need at least 200 for the SMA + one prior close

    def initialize(self) -> None:
        self.set_start_date(2010, 1, 1)
        self.set_cash(100_000)

        self._spy = self.add_equity(self._TICKER, Resolution.MINUTE).symbol

        self._entry_price = None
        self._trade_returns = []    # frictionless per-round-trip returns
        self._equity_values = []    # 15:59 ET daily equity snapshots (net curve)
        self._dip_nights = 0
        self._signal_days = 0

        # Dip decision + entry, one minute before the close (fills at the
        # 15:59 bar, the closest available approximation of the close).
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 1),
            self._decide_and_enter,
        )
        # Exit at the next session open.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.after_market_open(self._spy, 0),
            self._exit_at_open,
        )

    # ------------------------------------------------------------------ #
    def _dip_signal(self) -> bool:
        """Evaluate the dip rule on daily bars ending the PREVIOUS session."""
        end = self.time.date() - timedelta(days=1)
        start = end - timedelta(days=self._LOOKBACK_DAYS)
        bars = self.history(self._spy, start, end, Resolution.DAILY)
        if bars is None or bars.empty:
            return False

        closes = bars["close"].to_numpy(dtype=float)
        if closes.size < self._MIN_BARS:
            return False

        last = closes[-1]
        sma200 = float(np.mean(closes[-200:]))
        ema5 = float(pd.Series(closes).ewm(span=5, adjust=False).mean().iloc[-1])
        down_close = last < closes[-2]

        trend_up = last > sma200
        below_short_ma = last < ema5
        return bool(trend_up and below_short_ma and down_close)

    # ------------------------------------------------------------------ #
    def _decide_and_enter(self) -> None:
        # Daily equity snapshot for the net-of-fees curve (before today's
        # entry order is submitted, so pending orders never distort it).
        value = float(self.portfolio.total_portfolio_value)
        if value > 0:
            self._equity_values.append(value)

        self._signal_days += 1
        if self.portfolio.invested:
            return
        if not self._dip_signal():
            return

        self._dip_nights += 1
        self.set_holdings(self._spy, 1.0)

    # ------------------------------------------------------------------ #
    def _exit_at_open(self) -> None:
        if self.portfolio.invested:
            self.liquidate(self._spy)

    # ------------------------------------------------------------------ #
    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED:
            return
        if order_event.symbol != self._spy:
            return
        if order_event.fill_price <= 0:
            return

        if self._entry_price is None and order_event.quantity > 0:
            self._entry_price = order_event.fill_price
        elif self._entry_price is not None and order_event.quantity < 0:
            self._trade_returns.append(
                order_event.fill_price / self._entry_price - 1.0
            )
            self._entry_price = None

    # ------------------------------------------------------------------ #
    def _describe_series(self, values, label: str, periods: int) -> None:
        if values.size < 2:
            self.log(f"[STATS] {label}: insufficient data")
            return
        cumulative = float(np.prod(1.0 + values) - 1.0)
        running_max = np.maximum.accumulate(np.cumprod(1.0 + values))
        drawdowns = np.cumprod(1.0 + values) / running_max - 1.0
        max_dd = float(drawdowns.min())
        std = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        sharpe = mean / std * np.sqrt(periods) if std > 0 else 0.0
        self.log(
            f"[STATS] {label}: n={values.size} cum={cumulative * 100:.2f}% "
            f"mean={mean * 100:.4f}% sharpe={sharpe:.2f} maxDD={max_dd * 100:.2f}%"
        )

    # ------------------------------------------------------------------ #
    def on_end_of_algorithm(self) -> None:
        rets = np.array(self._trade_returns, dtype=float)
        wins = int((rets > 0).sum()) if rets.size else 0
        self.log(
            f"[SUMMARY] signal_days={self._signal_days} dip_nights={self._dip_nights} "
            f"round_trips={rets.size} win_rate={wins / rets.size * 100:.2f}%"
            if rets.size
            else "[SUMMARY] no round trips"
        )
        self._describe_series(rets, "overnight dip trades (frictionless)", 252)

        values = np.array(self._equity_values, dtype=float)
        if values.size > 1:
            net_daily = values[1:] / values[:-1] - 1.0
            self._describe_series(net_daily, "equity curve daily (net of fees)", 252)
            self.log(
                f"[SUMMARY] final_value={values[-1]:.2f} "
                f"net_cum={values[-1] / values[0] * 100 - 100:.2f}%"
            )

        # SPY buy & hold benchmark and its overnight/intraday decomposition
        # over the same backtest window.
        bh = self.history(self._spy, datetime(2010, 1, 1), self.time, Resolution.DAILY)
        if bh is not None and not bh.empty:
            close = bh["close"].to_numpy(dtype=float)
            open_ = bh["open"].to_numpy(dtype=float)
            overnight = open_[1:] / close[:-1] - 1.0
            intraday = close[1:] / open_[1:] - 1.0
            self.log(
                f"[BENCH] SPY buy&hold={close[-1] / close[0] * 100 - 100:.2f}% "
                f"overnight_comp={float(np.prod(1 + overnight) - 1) * 100:.2f}% "
                f"intraday_comp={float(np.prod(1 + intraday) - 1) * 100:.2f}%"
            )
