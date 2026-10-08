# region imports
from AlgorithmImports import *
import math
import statistics
# endregion


class EtfState:
    """Rolling daily filter inputs and overnight trade stats for one ETF."""

    def __init__(self, symbol: Symbol) -> None:
        self.symbol = symbol
        self._closes: list[float] = []
        self._dollar_volumes: list[float] = []
        self.trades = 0
        self.wins = 0
        self.cum = 1.0

    def reset(self) -> None:
        self._closes = []
        self._dollar_volumes = []

    def update(self, close: float, volume: float) -> None:
        self._closes.append(float(close))
        self._dollar_volumes.append(float(close) * float(volume))

    @property
    def ready(self) -> bool:
        # 63 days for ADV63 and 22 closes for 21 returns, with buffer.
        return len(self._closes) >= 64

    def adv63(self) -> float:
        return statistics.mean(self._dollar_volumes[-63:])

    def vol21(self) -> float:
        closes = self._closes[-22:]
        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        return statistics.stdev(returns) * math.sqrt(252.0)


class OvernightFilteredEtfAlgorithm(QCAlgorithm):
    """
    Overnight ETF strategy with liquidity and volatility filters (adaptation
    of the OvernightSpyBaselineAlgorithm).

    Mechanics are unchanged from the baseline: buy at the regular close
    (market order scheduled one minute before the close, fills at the open of
    the 15:59 bar), sell at the next regular open, regular session only, 100%
    allocation, no leverage.

    Adaptation: each day ~90 minutes before the close, one batch daily history
    request refreshes the 63-day average dollar volume (ADV63) and the 21-day
    annualized close-to-close volatility (vol21) of the ETF universe. ETFs in
    the top LIQUIDITY_FRACTION by ADV63 form the liquid pool; the single ETF
    traded overnight is the pool member with the highest (VOL_TILT="high") or
    lowest (VOL_TILT="low") vol21. The filter window ends at the previous
    session, so there is no look-ahead. Trading one ETF per day keeps the run
    at ~2 orders/day (~8.4k over the 2010+ window), inside the order quota
    that the baseline already noted; multi-ETF baskets would exceed it.

    on_end_of_algorithm logs compounded overnight-curve statistics (cumulative
    return, CAGR, annualized Sharpe, max drawdown, win rate), a per-ETF
    breakdown of what the filters selected, and the SPY overnight/intraday
    decomposition. The results page shows the same Sharpe/drawdown net of
    fees from the actual equity curve.
    """

    ETF_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "GLD"]
    LIQUIDITY_FRACTION = 0.5  # keep the top half of the universe by ADV63
    VOL_TILT = "low"  # "high" or "low" realized-volatility preference

    def initialize(self) -> None:
        # Same window as the baseline (2010-01-01 -> present).
        self.set_start_date(2010, 1, 1)
        self.set_cash(100_000)

        self._states = {}
        self._spy = None
        for ticker in self.ETF_TICKERS:
            symbol = self.add_equity(ticker, Resolution.MINUTE).symbol
            self._states[symbol] = EtfState(symbol)
            if ticker == "SPY":
                self._spy = symbol

        self._target = None
        self._entry_symbol = None
        self._entry_price = 0.0
        self._overnight_returns: list[float] = []
        self._logged_selections = 0

        # Daily filter refresh before the close (tracks early-close days too).
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 90),
            self._select_target,
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

    def _select_target(self) -> None:
        symbols = list(self._states.keys())
        history = self.history(symbols, 80, Resolution.DAILY)
        universe_index = history.index.get_level_values(0) if not history.empty else []
        for symbol in symbols:
            state = self._states[symbol]
            state.reset()
            if symbol not in universe_index:
                continue
            bars = history.loc[symbol]
            for close, volume in zip(bars["close"], bars["volume"]):
                state.update(close, volume)

        ready = [s for s in self._states.values() if s.ready]
        if not ready:
            self._target = None
            return

        ranked = sorted(ready, key=lambda s: s.adv63(), reverse=True)
        pool_size = max(1, int(round(len(ranked) * self.LIQUIDITY_FRACTION)))
        pool = ranked[:pool_size]
        if self.VOL_TILT == "low":
            self._target = min(pool, key=lambda s: s.vol21())
        else:
            self._target = max(pool, key=lambda s: s.vol21())

        if self._logged_selections < 3:
            self._logged_selections += 1
            tickers = ", ".join(s.symbol.value for s in pool)
            self.log(
                f"[SELECT] {self.time:%Y-%m-%d} pool=[{tickers}] "
                f"target={self._target.symbol.value} vol21={self._target.vol21():.2%}"
            )

    def _enter_overnight(self) -> None:
        if self._target is None:
            return
        if self.portfolio.invested:
            # Safety net: the book is flat by construction after each open.
            self.liquidate()
            return
        self.set_holdings(self._target.symbol, 1.0)

    def _exit_overnight(self) -> None:
        if self.portfolio.invested:
            self.liquidate()

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED:
            return
        state = self._states.get(order_event.symbol)
        if state is None:
            return
        if order_event.quantity > 0:
            # Entry fill near the close -> remember symbol and price.
            self._entry_symbol = order_event.symbol
            self._entry_price = order_event.fill_price
        elif self._entry_symbol == order_event.symbol and self._entry_price > 0:
            # Exit fill near the next open -> one overnight round trip.
            r = order_event.fill_price / self._entry_price - 1.0
            self._overnight_returns.append(r)
            state.trades += 1
            state.wins += 1 if r > 0 else 0
            state.cum *= 1.0 + r
            self._entry_symbol = None
            self._entry_price = 0.0

    def on_end_of_algorithm(self) -> None:
        returns = self._overnight_returns
        if not returns:
            return

        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        cumulative = 1.0
        for r in returns:
            cumulative *= 1.0 + r

        mean = statistics.mean(returns)
        std = statistics.stdev(returns) if n > 1 else 0.0
        sharpe = mean / std * math.sqrt(252.0) if std > 0 else 0.0

        # Max drawdown on the compounded overnight curve (flat intraday).
        peak = 1.0
        equity = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            max_dd = max(max_dd, peak / equity - 1.0)

        years = (self.time - self.start_date).total_seconds() / (365.25 * 24 * 3600)
        cagr = cumulative ** (1.0 / years) - 1.0 if cumulative > 0 else float("nan")

        self.log(
            f"[OVERNIGHT-FILTERED tilt={self.VOL_TILT}] round_trips={n} "
            f"win_rate={wins / n:.2%} mean_per_trade={mean:.6f} "
            f"median_per_trade={statistics.median(returns):.6f} "
            f"cumulative={cumulative - 1:.2%} cagr={cagr:.2%} "
            f"sharpe={sharpe:.2f} max_drawdown={max_dd:.2%}"
        )

        for state in self._states.values():
            if state.trades:
                self.log(
                    f"[SELECTIONS] {state.symbol.value} trades={state.trades} "
                    f"win_rate={state.wins / state.trades:.2%} "
                    f"cumulative_overnight={state.cum - 1:.2%}"
                )

        # Decompose SPY daily bars into overnight (close t -> open t+1) and
        # intraday (open t -> close t) components; adjusted data so dividends
        # are included. Same reference stats as the baseline.
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
