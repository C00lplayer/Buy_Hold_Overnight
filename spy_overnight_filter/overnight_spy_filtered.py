# region imports
from AlgorithmImports import *
import math
import statistics
# endregion


class OvernightSpyFilteredAlgorithm(QCAlgorithm):
    """
    Overnight SPY strategy with a liquidity + volatility gate.

    Baseline behavior (buy 100% SPY at the regular close, sell at the next
    regular open) is kept, but the overnight position is only taken when the
    gate passes; otherwise nothing happens that day (stay flat).

    Gate, evaluated once per day 30 minutes before the close from daily bars
    that end on the previous session (no look-ahead):
      - High liquidity: last completed session's dollar volume (close x
        volume) >= its trailing 63-session median.
      - Low volatility: 21-session realized volatility (annualized) as of the
        last completed session <= its trailing 252-session median.

    Regular session only, 100% allocation, no leverage, no indicators.
    """

    LIQUIDITY_LOOKBACK = 63
    VOL_LOOKBACK = 21
    VOL_MEDIAN_LOOKBACK = 252
    HISTORY_BARS = VOL_LOOKBACK + VOL_MEDIAN_LOOKBACK + 10

    def initialize(self) -> None:
        # Same window as the baseline: 2010-01-01 -> present, ~2 orders/day.
        self.set_start_date(2010, 1, 1)
        self.set_cash(100_000)

        self._spy = self.add_equity("SPY", Resolution.MINUTE).symbol

        # Compute the gate once per day, 30 minutes before the close.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 30),
            self._refresh_filter,
        )

        # Entry: as close as practically possible to the regular close.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 1),
            self._enter_overnight,
        )

        # Exit: as close as practically possible to the next regular open.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.after_market_open(self._spy, 0),
            self._exit_overnight,
        )

        # Sample end-of-day equity (flat, cash) just before the entry event,
        # so each row reflects that morning's exit fill.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 5),
            self._record_equity,
        )

        self._filter_passes = False
        self._liquidity_ok = False
        self._volatility_ok = False
        self._filter_pass_days = 0
        self._filter_block_days = 0
        self._entry_price = 0.0
        self._overnight_returns: list[float] = []
        self._daily_equity: list[tuple[str, float]] = []

    def _refresh_filter(self) -> None:
        # One batch daily history call; during the session the last completed
        # daily bar is the previous session, so no same-day information leaks.
        bars = self.history(self._spy, self.HISTORY_BARS, Resolution.DAILY)
        min_bars = self.LIQUIDITY_LOOKBACK + self.VOL_LOOKBACK
        if bars.empty or len(bars) < min_bars:
            self._filter_passes = False
            self._filter_block_days += 1
            return

        closes = bars["close"].tolist()
        volumes = bars["volume"].tolist()

        # Liquidity gate: last completed session's dollar volume vs its
        # trailing median.
        dollar_volumes = [c * v for c, v in zip(closes, volumes)]
        recent_dv = dollar_volumes[-self.LIQUIDITY_LOOKBACK:]
        self._liquidity_ok = recent_dv[-1] >= statistics.median(recent_dv)

        # Volatility gate: current 21-day realized vol (annualized) vs its
        # trailing 252-day median.
        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        vols = [
            statistics.stdev(returns[i - self.VOL_LOOKBACK : i]) * math.sqrt(252)
            for i in range(self.VOL_LOOKBACK, len(returns) + 1)
        ]
        recent_vols = vols[-self.VOL_MEDIAN_LOOKBACK:]
        self._volatility_ok = recent_vols[-1] <= statistics.median(recent_vols)

        self._filter_passes = self._liquidity_ok and self._volatility_ok
        if self._filter_passes:
            self._filter_pass_days += 1
        else:
            self._filter_block_days += 1

    def _enter_overnight(self) -> None:
        # Nothing happens when the gate failed (or a position is already open).
        if self.portfolio.invested or not self._filter_passes:
            return
        if self.securities[self._spy].price <= 0:
            return
        self.set_holdings(self._spy, 1.0)

    def _exit_overnight(self) -> None:
        if self.portfolio.invested:
            self.liquidate(self._spy)

    def _record_equity(self) -> None:
        # Flat (cash) at this point on every day, so the sample equals the
        # end-of-day equity including this morning's exit fill.
        self._daily_equity.append(
            (self.time.date().isoformat(), float(self.portfolio.total_portfolio_value))
        )

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED or order_event.symbol != self._spy:
            return
        if order_event.quantity > 0:
            # Entry fill near the close -> remember the price.
            self._entry_price = order_event.fill_price
        elif self._entry_price > 0:
            # Exit fill near the next open -> one overnight round trip.
            self._overnight_returns.append(order_event.fill_price / self._entry_price - 1)
            self._entry_price = 0.0

    def on_end_of_algorithm(self) -> None:
        self.log(
            f"[FILTER] pass_days={self._filter_pass_days} block_days={self._filter_block_days}"
        )

        returns = self._overnight_returns
        if not returns:
            return

        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        mean = statistics.mean(returns)
        std = statistics.stdev(returns) if n > 1 else 0.0
        sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0

        # Compounded frictionless curve of the taken overnight legs.
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            cumulative *= 1.0 + r
            peak = max(peak, cumulative)
            max_dd = min(max_dd, cumulative / peak - 1.0)

        self.log(
            f"[OVERNIGHT-FILTERED] round_trips={n} win_rate={wins / n:.2%} "
            f"mean_per_trade={mean:.6f} median_per_trade={statistics.median(returns):.6f} "
            f"cumulative={cumulative - 1:.2%} sharpe_annualized={sharpe:.2f} "
            f"max_drawdown={max_dd:.2%}"
        )

        # Decompose SPY daily bars into overnight (close t -> open t+1) and
        # intraday (open t -> close t) components; adjusted data so dividends
        # are included. Same reference block as the baseline.
        daily = self.history(self._spy, self.start_date, self.time, Resolution.DAILY)
        if daily.empty:
            return

        opens = daily["open"].tolist()
        closes = daily["close"].tolist()

        overnight = 1.0
        for i in range(1, len(closes)):
            overnight *= 1.0 + opens[i] / closes[i - 1] - 1.0

        daytime = 1.0
        for o, c in zip(opens, closes):
            daytime *= 1.0 + c / o - 1.0

        bnh = closes[-1] / closes[0] - 1.0
        years = (self.time - self.start_date).total_seconds() / (365.25 * 24 * 3600)
        bnh_cagr = (closes[-1] / closes[0]) ** (1.0 / years) - 1.0

        self.log(
            f"[SPY-DAILY] days={len(closes)} buy_hold_total={bnh:.2%} "
            f"buy_hold_cagr={bnh_cagr:.2%} overnight_component={overnight - 1:.2%} "
            f"daytime_component={daytime - 1:.2%} "
            f"check_product={overnight * daytime - 1:.2%}"
        )

        # Free-plan organization: no Object Store, and bulk series data stays
        # in the cloud. Chart the engine's own daily equity curve instead; the
        # results page is the sanctioned place to view exact backtest output.
        for _d, _v in self._daily_equity:
            self.plot("Overnight Gate Equity", "Equity", _v)

        self.log(f"[EQUITY] days={len(self._daily_equity)}")
